import os
import sys
import argparse

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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

def main(args):
    """
    Main function to handle the ONNX export process with multiple named outputs.
    """
    print(f"Creating model: {args.model_type}")
    model = getattr(zoo.model, args.model_type)()
    
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found at {args.weight_path}")

    print(f"Loading weights from: {args.weight_path}")
    checkpoint = torch.load(args.weight_path, map_location='cpu')

    if args.ema and 'ema' in checkpoint and checkpoint['ema'] is not None:
        print("Loading EMA model state.")
        model.load_state_dict(checkpoint['ema']['module'])
    else:
        print("Loading standard model state.")
        model.load_state_dict(checkpoint['model'])

    model.eval()
    print("Weights loaded successfully and model set to evaluation mode.")

    dummy_input = torch.randn(1, 3, args.height, args.width)
    
    # --- FIX: Define the correct output names based on the model's forward pass ---
    output_names = ["pred_logits", "pred_boxes", "pred_masks"]
    
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'pred_logits': {0: 'batch_size'},
        'pred_boxes': {0: 'batch_size'},
        'pred_masks': {0: 'batch_size'},
    }
    
    output_dir = os.path.dirname(args.save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting model to ONNX at: {args.save_path}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.save_path,
            opset_version=16,
            do_constant_folding=True,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print("ONNX export completed successfully!")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RT-DETR ONNX Export script')
    parser.add_argument('--weight_path', '-w', type=str, required=True, help='Path to the PyTorch checkpoint file (.pth)')
    parser.add_argument('--model_type', type=str, required=True, choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd', 'r50vd_segm'], help='The model architecture type')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the exported ONNX model file (.onnx)')
    parser.add_argument('--ema', type=str2bool, default=True, help='Export the EMA model state')
    parser.add_argument('--height', type=int, default=640, help='Input image height')
    parser.add_argument('--width', type=int, default=640, help='Input image width')
    args = parser.parse_args()
    main(args)
