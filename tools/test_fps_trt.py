import os
import sys
import time
import argparse

import torch
import psutil
import tensorrt as trt
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data.dataloader import DataLoader, BatchImageCollateFuncion
from src import zoo

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TensorRTInfer:
    """
    A class to handle TensorRT engine building and inference.
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found at {engine_path}")
            
        print(f"Loading TensorRT engine from: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = torch.cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = np.empty(size, dtype=dtype)
            device_mem = torch.empty(size, dtype=torch.from_numpy(np.dtype(dtype))).cuda()
            
            bindings.append(device_mem.data_ptr())
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, bindings, stream

    def __call__(self, x: torch.Tensor):
        # Transfer input data to the page-locked host buffer
        np.copyto(self.inputs[0]['host'], x.cpu().numpy().ravel())
        
        # Transfer input data from host to device
        torch.cuda.memcpy_async(self.inputs[0]['device'].data_ptr(), self.inputs[0]['host'].ctypes.data, self.inputs[0]['host'].nbytes, torch.cuda.memcpy_host_to_device, self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.native_stream)
        
        # Transfer predictions back from device to host
        for out in self.outputs:
            torch.cuda.memcpy_async(out['host'].ctypes.data, out['device'].data_ptr(), out['device'].element_size() * out['device'].nelement(), torch.cuda.memcpy_device_to_host, self.stream)
            
        self.stream.synchronize()
        
        # Reshape the flattened output to the correct shape
        batch_size = x.shape[0]
        reshaped_outputs = []
        for i, out in enumerate(self.outputs):
            shape = (batch_size, ) + self.engine.get_binding_shape(i + 1) # +1 because input is at index 0
            reshaped_outputs.append(out['host'][:trt.volume(shape)].reshape(shape))
            
        return reshaped_outputs

@torch.no_grad()
def benchmark(trt_infer, data_loader, device):
    timings = []
    process = psutil.Process(os.getpid())
    
    print(f"Performing warm-up runs...")
    warmup_loader = iter(data_loader)
    for _ in range(20):
        try:
            samples, _ = next(warmup_loader)
            _ = trt_infer(samples.to(device))
        except StopIteration:
            break
    
    print("Warm-up complete. Starting benchmark...")
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    process.cpu_percent(interval=None) 
    ram_usages_mb, gpu_mem_usages_mb = [], []
    
    image_count = 0
    for samples, _ in data_loader:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = trt_infer(samples.to(device))
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)
        image_count += samples.size(0)
        
        ram_usages_mb.append(process.memory_info().rss / (1024 * 1024))
        if device.type == 'cuda':
            gpu_mem_usages_mb.append(torch.cuda.memory_allocated(device) / (1024 * 1024))

    cpu_usage_percent = process.cpu_percent(interval=None)
    avg_ram_mb = sum(ram_usages_mb) / len(ram_usages_mb) if ram_usages_mb else 0
    peak_ram_mb = max(ram_usages_mb) if ram_usages_mb else 0
    avg_gpu_mem_mb = sum(gpu_mem_usages_mb) / len(gpu_mem_usages_mb) if gpu_mem_usages_mb else 0
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0

    total_time = sum(timings)
    fps = image_count / total_time

    print("\n" + "="*40)
    print("--- TensorRT Performance Test Results ---")
    print(f"Total images tested: {image_count} (entire validation set)")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("-" * 20)
    print("--- Resource Usage ---")
    print(f"Average GPU Memory: {avg_gpu_mem_mb:.2f} MB")
    print(f"Peak GPU Memory: {peak_gpu_mem_mb:.2f} MB")
    print(f"Average RAM Usage: {avg_ram_mb:.2f} MB")
    print(f"Peak RAM Usage: {peak_ram_mb:.2f} MB")
    print(f"Average CPU Usage: {cpu_usage_percent:.2f}%")
    print("="*40)

def build_engine(onnx_path, engine_path, use_fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

    if use_fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 mode for TensorRT engine build.")
        config.set_flag(trt.BuilderFlag.FP16)

    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError("Failed to parse the ONNX file.")
    
    builder.max_batch_size = 16 # Set max batch size
    
    print("Building serialized TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine.")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to: {engine_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("TensorRT requires a NVIDIA GPU.")

    if not os.path.exists(args.engine_path):
        print("Engine file not found, building a new one...")
        build_engine(args.onnx_path, args.engine_path, args.fp16)

    trt_infer = TensorRTInfer(args.engine_path)

    val_dataset = zoo.dataset.coco_val_dataset(
        img_folder=os.path.join(args.dataset_dir, "val2017"),
        ann_file=os.path.join(args.dataset_dir, "annotations/instances_val2017.json")
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False, 
        collate_fn=BatchImageCollateFuncion()
    )

    benchmark(trt_infer, val_dataloader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TensorRT FPS Test script')
    
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the ONNX model file')
    parser.add_argument('--engine_path', type=str, required=True, help='Path to save/load the TensorRT engine file')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--fp16', type=str2bool, default=True, help='Enable FP16 mode for TensorRT')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)
