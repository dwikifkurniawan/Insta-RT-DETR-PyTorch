# Insta-RT-DETR: Efficient Real-Time Instance Segmentation

This repository contains the official PyTorch implementation of the paper: **Insta-RT-DETR: Modifying RT-DETR for Efficient Real-Time Instance Segmentation**

---

## 📍 Overview

The primary challenge in real-time instance segmentation is achieving an optimal balance between accuracy and latency. Insta-RT-DETR addresses this trade-off by strategically integrating components from high-accuracy models (Mask DINO) into the efficient RT-DETR framework.

Specifically, we incorporate:

- A lightweight mask prediction branch.
- Hybrid matching strategies.
- Unified denoising training.

This is achieved while preserving RT-DETR's Efficient Hybrid Encoder and Uncertainty-Minimal Query Selection. The result is a model that outperforms existing real-time baselines in accuracy while maintaining high inference speeds.

---

## 📊 Performance

Insta-RT-DETR establishes a new state-of-the-art trade-off on the COCO val2017 dataset (benchmarked on a Tesla V100 GPU):

| Model                    | Backbone | Input Size | Mask AP  | FPS  |
| ------------------------ | -------- | ---------- | -------- | ---- |
| Mask DINO                | R50      | 640        | 41.4     | 20.5 |
| FastInst-D3              | R50      | 640        | 40.5     | 32.5 |
| SparseInst               | R50      | 608        | 37.9     | 46.5 |
| **Insta-RT-DETR (Ours)** | R50      | 640        | **42.5** | 31.8 |

**Highlight:** Our model achieves **42.5% Mask AP at 31.8 FPS**, outperforming the re-evaluated Mask DINO baseline (41.4% AP @ 20.5 FPS) and FastInst-D3 (40.5% AP @ 32.5 FPS).

---

## 🛠️ Installation

### Requirements

- Python 3.10+ (Tested on 3.11)
- PyTorch >= 2.0.1
- Torchvision >= 0.17

### Steps

```bash
# Clone the repository
git clone https://github.com/dwikifkurniawan/Insta-RT-DETR-PyTorch.git
cd Insta-RT-DETR-PyTorch

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Data Preparation

Download the COCO 2017 dataset and organize it as follows:

```
dataset/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   ├── 000000000009.jpg
    │   └── ...
    └── val2017/
        ├── 000000000139.jpg
        └── ...
```

---

## 🚀 Usage

### 1. Training

To train the model on a single GPU (or multi-GPU using `torchrun`):

```bash
# Basic training command
python tools/train.py \
    --dataset_dir dataset/coco \
    --batch_size 4 \
    --model_type r50vd_segm \
    --save_dir output/insta_rtdetr_r50 \
    --epoch 50
```

**Key Arguments:**

- `--model_type`: Default is `r50vd_segm` (ResNet-50 backbone with segmentation).
- `--use_wandb`: Enable Weights & Biases logging (recommended).
- `--amp`: Enable Automatic Mixed Precision (default: True).

### 2. Evaluation

To evaluate a trained model on the validation set:

```bash
python tools/train.py \
    --val True \
    --weight_path output/insta_rtdetr_r50/best.pth \
    --dataset_dir dataset/coco \
    --batch_size 8
```

### 3. Web Demo (Inference)

We provide a Flask-based web application for easy inference on custom images.

1. Ensure you have a trained model weight file (e.g., `checkpoint/model/50.pth`).
2. Run the app:

```bash
python app.py
```

3. Open your browser at `http://localhost:5123` and upload an image to see detected bounding boxes and segmentation masks.

**Note:** The app automatically handles aspect-ratio preserving resizing (padding to 640x640) and maps coordinates back to the original image size.

---

## 🔗 Citation

The paper has been accepted and is currently undergoing minor revision.

---

## 👏 Acknowledgements

This code is built upon the excellent work of:

- [RT-DETR (Lywenyu)](https://github.com/lyuwenyu/RT-DETR)
- [RTDETR-PyTorch (int11)](https://github.com/int11/RTDETR-PyTorch)
- [Mask DINO](https://github.com/IDEA-Research/MaskDINO)

---

## 📄 License

This project is released under the license specified in the `LICENSE` file.

