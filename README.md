# 🌆 Cityscapes Semantic Segmentation using SegFormer-B2

This repository implements the **SegFormer-B2** architecture for semantic segmentation on the **Cityscapes** dataset.  
The model was trained for **20 epochs** and achieved an **accuracy of 86%**.

📦 **Model Weights:**  
The trained model weights have been uploaded under the **[v1.0.0 release](https://github.com/vinith-369/SegFormer/releases/tag/v1.0.0)**.  
You can download the `.pth` file from there to directly test or fine-tune the model.

---

## 📘 Overview

SegFormer is a **transformer-based semantic segmentation model** introduced in the paper  
> [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/pdf/2105.15203)  
by Xie et al., 2021.

Unlike traditional CNN-based methods, SegFormer combines **hierarchical Transformers** with an **MLP decoder**, leading to efficient feature aggregation and excellent performance without positional encodings.

---

## 🧠 Model Architecture

The implementation includes the following key modules:

- `Attention.py` — Implements the **Efficient Self-Attention** mechanism used in SegFormer.  
- `Head.py` — Defines the **MLP-based Decoder Head**.  
- `modules.py` — Contains building blocks like **Mix FFN**, **Overlap Patch Embedding**, and **Transformer Encoder Layers**.  
- `segformer.py` — Integrates encoder and decoder into a complete SegFormer-B2 architecture.  
- `utils.py` — Includes helper functions for dataset preprocessing, visualization, and evaluation.  
- `__init__.py` — Makes the modules importable as a package.

---

## 📊 Dataset

**Dataset:** [Cityscapes Dataset (Kaggle Mirror)](https://www.kaggle.com/datasets/xiaose/cityscapes)  
The dataset consists of **high-resolution urban street scenes** with pixel-level semantic annotations.  
It includes 19 semantic classes such as road, car, pedestrian, building, and vegetation.

---

## ⚙️ Training Details

| Parameter | Value |
|------------|--------|
| **Model** | SegFormer-B2 |
| **Dataset** | Cityscapes |
| **Epochs** | 20 |
| **Accuracy** | 86% |
| **Optimizer** | AdamW |
| **Learning Rate** | 6e-5 |
| **Batch Size** | 8 |
| **Loss Function** | Cross-Entropy Loss |
| **Hardware** | GPU (Recommended) |

---

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/vinith-369/SegFormer.git
cd SegFormer
