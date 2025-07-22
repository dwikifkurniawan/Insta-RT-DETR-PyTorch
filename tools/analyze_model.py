import torch
import argparse
import sys
import os
from thop import profile, clever_format

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.zoo import model as zoo_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="An accurate analysis tool for RT-DETR models using thop.",
        epilog="""
Example:
$ python tools/analyze_model_accurate.py --model_type r50vd_segm
"""
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='r50vd_segm',
        choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd', 'r50vd_segm'],
        help='Choose the model type to analyze.'
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the model weights (.pth) file to load."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = getattr(zoo_model, args.model_type)().to(device)

    if args.weights:
        print(f"Loading weights from: {args.weights}")
        try:
            state_dict = torch.load(args.weights, map_location=device)['model']
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)
    else:
        print("Analyzing model with initialized (random) weights.")

    model.eval()

    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    print(f"Analyzing model: {args.model_type} with input shape: {dummy_input.shape}")
    print("\n" + "="*20 + " Detailed Analysis Table " + "="*20)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=True)
    print("="*65)

    macs, params = clever_format([macs, params], "%.2f")
    
    gflops = (float(macs[:-1]) * 2)

    print(f"\n--- Summary ---")
    print(f"Model:           {args.model_type}")
    print(f"Parameters:      {params}")
    print(f"MACs:            {macs}")
    print(f"GFLOPs:          {gflops:.2f}G")
    print("-----------------")