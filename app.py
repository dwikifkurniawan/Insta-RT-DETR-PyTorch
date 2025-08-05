import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template

from src.zoo.model import r50vd_segm
from src.nn.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor

# config
MODEL_WEIGHTS = 'checkpoint/model/50.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 80
CONF_THRESHOLD = 0.5

app = Flask(__name__, template_folder='templates')

def load_model_and_components():
    """Load the model and postprocessor"""
    print(f"Loading model from {MODEL_WEIGHTS} on {DEVICE}...")
    try:
        model = r50vd_segm()
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        postprocessor = RTDETRPostProcessor(num_classes=NUM_CLASSES, use_focal_loss=True)
        print("Model and postprocessor loaded successfully")
        return model, postprocessor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# load model
model, postprocessor = load_model_and_components()

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not postprocessor:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = img_rgb.shape[:2]
        print(f"Original image: {original_h}×{original_w} (H×W)")
        
        # Resize with aspect ratio preservation
        scale = min(640 / original_w, 640 / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # Create padded image
        img_padded = np.zeros((640, 640, 3), dtype=np.uint8)
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
        
        print(f"Resized to: {new_w}×{new_h}, Padding: ({pad_x}, {pad_y}), Scale: {scale:.3f}")
        
        # Store preprocessing info for coordinate correction
        preprocess_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'new_w': new_w,
            'new_h': new_h
        }
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
        
        model_input_size = torch.tensor([[640, 640]], device=DEVICE)
        results = postprocessor(outputs, model_input_size)
        
        print(f"Postprocessor input: 640×640 (model input size)")
        print(f"Will transform coordinates back to: {original_w}×{original_h}")
        
        # Filter results
        result = results[0]
        scores = result['scores']
        keep = scores > CONF_THRESHOLD
        
        boxes = result['boxes'][keep]
        labels = result['labels'][keep]
        masks = result['masks'][keep]
        final_scores = scores[keep]
        
        print(f"Found {len(final_scores)} detections")
        
        # Process predictions
        predictions = []
        for i in range(len(final_scores)):
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(float)
            
            print(f"Raw coordinates from postprocessor: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # Transform coordinates from 640x640 model space back to original image space
            # Remove padding offset
            x1_unpad = x1 - preprocess_info['pad_x']
            y1_unpad = y1 - preprocess_info['pad_y']
            x2_unpad = x2 - preprocess_info['pad_x']
            y2_unpad = y2 - preprocess_info['pad_y']
            
            # Scale back to original size
            x1_orig = x1_unpad / preprocess_info['scale']
            y1_orig = y1_unpad / preprocess_info['scale']
            x2_orig = x2_unpad / preprocess_info['scale']
            y2_orig = y2_unpad / preprocess_info['scale']
            
            print(f"After coordinate transformation: [{x1_orig:.1f}, {y1_orig:.1f}, {x2_orig:.1f}, {y2_orig:.1f}]")
            
            x1 = int(max(0, min(x1_orig, original_w)))
            y1 = int(max(0, min(y1_orig, original_h)))
            x2 = int(max(x1 + 1, min(x2_orig, original_w)))
            y2 = int(max(y1 + 1, min(y2_orig, original_h)))
            
            print(f"Final coordinates: [{x1}, {y1}, {x2}, {y2}] for {COCO_CLASSES[labels[i]]}")
            
            mask_contours = []
            if len(masks) > i:
                mask = masks[i].cpu().numpy()
                
                if len(mask.shape) == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                
                print(f"Mask shape from postprocessor: {mask.shape}")
                
                if mask.shape == (640, 640):
                    # Step 1: Remove padding from mask
                    mask_unpadded = mask[preprocess_info['pad_y']:preprocess_info['pad_y'] + preprocess_info['new_h'],
                                       preprocess_info['pad_x']:preprocess_info['pad_x'] + preprocess_info['new_w']]
                    
                    # Step 2: Resize back to original image size
                    mask_orig = cv2.resize(mask_unpadded.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                    print(f"Transformed mask to original size: {mask_orig.shape}")
                else:
                    print(f"Unexpected mask shape: {mask.shape}, expected (640, 640)")
                    continue
                
                # Extract mask region
                if mask_orig.shape == (original_h, original_w):
                    mask_region = mask_orig[y1:y2, x1:x2]
                    
                    if mask_region.size > 0:
                        # Convert to binary
                        binary_mask = (mask_region > 0.5).astype(np.uint8) * 255
                        
                        if binary_mask.sum() > 0:
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                contour = max(contours, key=cv2.contourArea)
                                if len(contour.shape) == 3:
                                    contour = contour.squeeze(1)
                                
                                contour_points = []
                                for point in contour:
                                    x_coord = int(point[0] + x1)
                                    y_coord = int(point[1] + y1)
                                    contour_points.append([x_coord, y_coord])
                                
                                mask_contours = contour_points
                                print(f"Generated {len(mask_contours)} contour points")
            
            predictions.append({
                'label': COCO_CLASSES[int(labels[i])],
                'score': float(final_scores[i]),
                'box': [x1, y1, x2, y2],
                'mask': mask_contours
            })
        
        return jsonify({
            'predictions': predictions,
            'image_info': {
                'width': int(original_w),   
                'height': int(original_h),  
                'preprocessing': {
                    'scale': float(preprocess_info['scale']),
                    'padding': [int(preprocess_info['pad_x']), int(preprocess_info['pad_y'])],
                    'resized_to': [int(preprocess_info['new_w']), int(preprocess_info['new_h'])]
                }
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5123, debug=True)