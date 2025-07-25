import os
import sys
import time
import argparse

import torch
from torch.utils.data import DataLoader

# Add the project root directory to the Python path.
# This is crucial for allowing the script to import modules from the 'src' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Correct Imports based on your provided files ---
# Importing the specific model creation functions from your model.py
from src.zoo.model import r50vd_segm, r50vd, r18vd, r34vd, r50vd_m, r101vd
# Importing the specific dataset creation function from your dataset.py
from src.zoo.dataset import coco_val_dataset

def get_args_parser():
    """
    Defines all necessary command-line arguments directly within this script.
    """
    parser = argparse.ArgumentParser('RT-DETR FPS Test script', add_help=False)
    
    # --- Core Arguments for FPS Test ---
    parser.add_argument('--model_type', type=str, required=True, help="Model type (e.g., 'r50vd_segm')")
    parser.add_argument('--weight_path', type=str, required=True, help="Path to the model checkpoint (.pth) file")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the root COCO dataset directory")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size for inference")
    parser.add_argument('--num_workers', default=2, type=int, help="Number of data loading workers")
    parser.add_argument('--device', default='cuda', help='Device to use for testing (e.g., "cpu" or "cuda")')
    parser.add_argument('--num_images', type=int, default=500, help="Number of images to use for the FPS test.")
    parser.add_argument('--warmup_runs', type=int, default=20, help="Number of initial runs to discard for GPU warm-up.")

    return parser

@torch.no_grad()
def benchmark(model, data_loader, device, args):
    """
    Main function to run the FPS benchmark.
    It includes a warm-up phase and uses CUDA synchronization for accurate timing.
    """
    model.eval()
    timings = []
    
    # --- Warm-up Phase ---
    print(f"Performing {args.warmup_runs} warm-up runs...")
    for i, (samples, _) in enumerate(data_loader):
        if i >= args.warmup_runs:
            break
        samples = samples.to(device)
        _ = model(samples)
    
    print("Warm-up complete. Starting benchmark.")
    
    # --- Benchmark Phase ---
    image_count = 0
    for samples, _ in data_loader:
        if image_count >= args.num_images:
            break
            
        samples = samples.to(device)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = model(samples)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)
        image_count += samples.size(0)

    # --- Results Calculation ---
    if not timings:
        print("Error: No images were processed. Cannot calculate FPS.")
        return

    total_time = sum(timings)
    fps = image_count / total_time

    print("\n" + "="*40)
    print("--- FPS Test Results ---")
    print(f"Model: {args.model_type}")
    print(f"Images tested: {image_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("="*40)


def main(args):
    """
    Sets up the model and COCO dataset, then calls the benchmark function.
    """
    device = torch.device(args.device)

    # --- Correct Model Creation ---
    # A dictionary to map the model_type string to the actual function call.
    MODEL_MAP = {
        'r50vd_segm': r50vd_segm,
        'r50vd': r50vd,
        'r18vd': r18vd,
        'r34vd': r34vd,
        'r50vd_m': r50vd_m,
        'r101vd': r101vd,
    }
    
    if args.model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model_type '{args.model_type}'. Available options are: {list(MODEL_MAP.keys())}")
    
    print(f"Creating model: {args.model_type}")
    model = MODEL_MAP[args.model_type]()
    model.to(device)

    # --- Load Weights ---
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found at {args.weight_path}")
    
    print(f"Loading weights from: {args.weight_path}")
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    
    # This logic correctly handles different ways checkpoints might be saved.
    state_dict = checkpoint.get('model_ema') or checkpoint.get('model') or checkpoint
    model.load_state_dict(state_dict)
    print("Weights loaded successfully.")

    # --- Correct COCO Dataset Creation ---
    val_img_folder = os.path.join(args.dataset_dir, 'val2017')
    val_ann_file = os.path.join(args.dataset_dir, 'annotations', 'instances_val2017.json')

    # Calls the coco_val_dataset function from your dataset.py
    val_dataset = coco_val_dataset(img_folder=val_img_folder, ann_file=val_ann_file)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn # Use the collate_fn from the dataset
    )

    # --- Run Benchmark ---
    benchmark(model, val_loader, device, args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
