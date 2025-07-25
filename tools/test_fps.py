import os
import sys
import time
import argparse
import psutil
import torch

# Add the project root directory to the Python path, just like in your train.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src import zoo
from src.data.coco.coco_dataset import CocoDetection
from src.data.dataloader import DataLoader, BatchImageCollateFuncion
from utils import fit, val, str2bool

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@torch.no_grad()
def benchmark(model, data_loader, device, args):
    """
    Main function to run the FPS benchmark.
    It includes a warm-up phase and uses CUDA synchronization for accurate timing.
    It also measures system resource usage (CPU, RAM, GPU Memory).
    """
    model.eval()
    timings = []
    
    # --- Resource Monitoring Setup ---
    # process = psutil.Process(os.getpid())
    
    # --- Warm-up Phase ---
    print(f"Performing {args.warmup_runs} warm-up runs...")
    warmup_loader = iter(data_loader)
    for i in range(args.warmup_runs):
        try:
            samples, _ = next(warmup_loader)
            samples = samples.to(device)
            _ = model(samples)
        except StopIteration:
            print("Warm-up runs are more than the dataset size. Stopping warm-up.")
            break
    
    print("Warm-up complete. Starting benchmark on the entire validation set...")
    
    # --- Benchmark Phase ---
    # Reset GPU memory stats and initialize CPU/RAM/GPU tracking before the main loop
    # if device.type == 'cuda':
    #     torch.cuda.reset_peak_memory_stats(device)
    
    # process.cpu_percent(interval=None) 
    # ram_usages_mb = []
    # gpu_mem_usages_mb = []
    
    image_count = 0
    for samples, _ in data_loader:
        samples = samples.to(device)
        
        # Synchronize before starting the timer
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # The actual model inference
        _ = model(samples)
        
        # Synchronize again to wait for the inference to complete
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)
        image_count += samples.size(0)
        
        # Record RAM and GPU usage after each step
        # ram_usages_mb.append(process.memory_info().rss / (1024 * 1024))
        # if device.type == 'cuda':
        #     gpu_mem_usages_mb.append(torch.cuda.memory_allocated(device) / (1024 * 1024))

    # --- Finalize Resource Metrics ---
    # cpu_usage_percent = process.cpu_percent(interval=None)
    
    # if ram_usages_mb:
    #     avg_ram_mb = sum(ram_usages_mb) / len(ram_usages_mb)
    #     peak_ram_mb = max(ram_usages_mb)
    # else:
    #     avg_ram_mb = peak_ram_mb = 0

    # if gpu_mem_usages_mb:
    #     avg_gpu_mem_mb = sum(gpu_mem_usages_mb) / len(gpu_mem_usages_mb)
    #     peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    # else:
    #     avg_gpu_mem_mb = peak_gpu_mem_mb = 0

    # --- Results Calculation ---
    if not timings:
        print("Error: No images were processed. Cannot calculate FPS.")
        return

    total_time = sum(timings)
    fps = image_count / total_time

    print("\n" + "="*40)
    print("--- Performance Test Results ---")
    print(f"Model: {args.model_type}")
    print(f"Total images tested: {image_count} (entire validation set)")
    print(f"Batch size: {args.batch_size}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    # print("-" * 20)
    # print("--- Resource Usage ---")
    # print(f"Average GPU Memory: {avg_gpu_mem_mb:.2f} MB")
    # print(f"Peak GPU Memory: {peak_gpu_mem_mb:.2f} MB")
    # print(f"Average RAM Usage: {avg_ram_mb:.2f} MB")
    # print(f"Peak RAM Usage: {peak_ram_mb:.2f} MB")
    # print(f"Average CPU Usage (over benchmark duration): {cpu_usage_percent:.2f}%")
    print("="*40)


def main(args):
    """
    Sets up the model and COCO dataset based on your train.py, then calls the benchmark function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Correct Model Creation (mirrors your train.py) ---
    print(f"Creating model: {args.model_type}")
    model = getattr(zoo.model, args.model_type)()
    model.to(device)

    # --- Load Weights (mirrors logic from your utils.py `val` function) ---
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found at {args.weight_path}")
    
    print(f"Loading weights from: {args.weight_path}")
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    
    # This logic correctly loads the standard or EMA model state from the checkpoint
    if args.ema and 'ema' in checkpoint and checkpoint['ema'] is not None:
        print("Loading EMA model state.")
        model.load_state_dict(checkpoint['ema']['module'])
    else:
        print("Loading standard model state.")
        model.load_state_dict(checkpoint['model'])
    
    print("Weights loaded successfully.")

    val_dataset = zoo.coco_val_dataset(
        img_folder=os.path.join(args.dataset_dir, "val2017"),
        ann_file=os.path.join(args.dataset_dir, "annotations/instances_val2017.json"), 
        dataset_class=CocoDetection)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, 
                                collate_fn=BatchImageCollateFuncion())

    # --- Run Benchmark ---
    benchmark(model, val_dataloader, device, args)


if __name__ == '__main__':
    # --- Argument Parser (mirrors your train.py) ---
    parser = argparse.ArgumentParser('RT-DETR FPS Test script')
    
    parser.add_argument('--weight_path', '-w', type=str, required=True, help='Path to the weight file')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_type', type=str, required=True, choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd', 'r50vd_segm'], help='Choose the model type')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--ema', type=str2bool, default=True, help='Use Exponential Moving Average model state')

    # --- Arguments specific to the FPS test script ---
    parser.add_argument('--num_images', type=int, default=500, help="Number of images to use for the FPS test.")
    parser.add_argument('--warmup_runs', type=int, default=20, help="Number of initial runs to discard for GPU warm-up.")
    
    args = parser.parse_args()
    main(args)
