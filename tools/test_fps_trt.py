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

# --- Imports from your project ---
from src.data.dataloader import DataLoader, BatchImageCollateFuncion
from src import zoo

# --- Utility function ---
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
    def __init__(self, onnx_path, engine_path, use_fp16=True):
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.use_fp16 = use_fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_or_build_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def load_or_build_engine(self):
        if os.path.exists(self.engine_path):
            print(f"Loading existing TensorRT engine from: {self.engine_path}")
            with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print(f"Building new TensorRT engine from: {self.onnx_path}")
            return self.build_engine()

    def build_engine(self):
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        config = builder.create_builder_config()

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

        if self.use_fp16 and builder.platform_has_fast_fp16:
            print("Enabling FP16 mode.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("FP16 not supported or not requested. Using FP32.")

        if not parser.parse_from_file(self.onnx_path):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX file.")
        
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 640, 640), (8, 3, 640, 640), (16, 3, 640, 640)) 
        config.add_optimization_profile(profile)

        print("Building serialized network...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build the TensorRT engine.")

        os.makedirs(os.path.dirname(self.engine_path), exist_ok=True)
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
        
        with trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(serialized_engine)

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = torch.cuda.Stream()

        dtype_map = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int8: torch.int8,
        }

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            shape_info = self.engine.get_tensor_profile_shape(name, 0)
            max_shape = shape_info[2]
            
            numpy_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = dtype_map.get(numpy_dtype)
            if torch_dtype is None:
                raise TypeError(f"Unsupported numpy dtype {numpy_dtype} for tensor {name}")

            # --- FIX: Removed the invalid 'size=' keyword argument ---
            device_mem = torch.empty(trt.volume(max_shape), dtype=torch_dtype).cuda()
            bindings.append(device_mem.data_ptr())

            if is_input:
                inputs.append({'name': name, 'buffer': device_mem})
            else:
                outputs.append({'name': name, 'buffer': device_mem})
                
        return inputs, outputs, bindings, stream

    def __call__(self, x: torch.Tensor):
        self.context.set_input_shape(self.inputs[0]['name'], x.shape)
        self.inputs[0]['buffer'].copy_(x.contiguous().flatten())
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
        
        results = []
        for output in self.outputs:
            output_shape = self.context.get_tensor_shape(output['name'])
            output_volume = trt.volume(output_shape)
            results.append(output['buffer'][:output_volume].view(output_shape))
            
        self.stream.synchronize()
        return results

@torch.no_grad()
def benchmark(trt_infer, data_loader, device):
    """
    Main function to run the FPS benchmark using a TensorRT engine.
    """
    timings = []
    process = psutil.Process(os.getpid())
    
    # --- Warm-up Phase ---
    print(f"Performing warm-up runs...")
    warmup_loader = iter(data_loader)
    for _ in range(20):
        try:
            samples, _ = next(warmup_loader)
            samples = samples.to(device)
            _ = trt_infer(samples)
        except StopIteration:
            break
    
    print("Warm-up complete. Starting benchmark on the entire validation set...")
    
    # --- Benchmark Phase ---
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    process.cpu_percent(interval=None) 
    ram_usages_mb = []
    gpu_mem_usages_mb = []
    
    image_count = 0
    for samples, _ in data_loader:
        samples = samples.to(device)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = trt_infer(samples)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)
        image_count += samples.size(0)
        
        ram_usages_mb.append(process.memory_info().rss / (1024 * 1024))
        if device.type == 'cuda':
            gpu_mem_usages_mb.append(torch.cuda.memory_allocated(device) / (1024 * 1024))

    # --- Finalize Resource Metrics ---
    cpu_usage_percent = process.cpu_percent(interval=None)
    avg_ram_mb = sum(ram_usages_mb) / len(ram_usages_mb) if ram_usages_mb else 0
    peak_ram_mb = max(ram_usages_mb) if ram_usages_mb else 0
    avg_gpu_mem_mb = sum(gpu_mem_usages_mb) / len(gpu_mem_usages_mb) if gpu_mem_usages_mb else 0
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0

    # --- Results Calculation ---
    total_time = sum(timings)
    fps = image_count / total_time

    print("\n" + "="*40)
    print("--- TensorRT Performance Test Results ---")
    print(f"Precision: {'FP16' if trt_infer.use_fp16 else 'FP32'}")
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

def main(args):
    """
    Sets up the dataset and TensorRT engine, then calls the benchmark function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("TensorRT requires a NVIDIA GPU.")

    # --- Setup TensorRT Engine ---
    trt_infer = TensorRTInfer(args.onnx_path, args.engine_path, args.fp16)

    # --- COCO Dataset Creation ---
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

    # --- Run Benchmark ---
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
