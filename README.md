# CS444 Final Project — Sign Language Recognition
**Spring 2026 | UIUC CS444: Deep Learning for Computer Vision**

This repository contains our exploration of multiple deep learning approaches for **American Sign Language (ASL) recognition**, ranging from ViT vs. CNN comparisons to real-time webcam-based letter-to-word translation. The project covers model reproduction, fine-tuning on ASL datasets, keypoint-based recognition, and video-based classification.

---

## Library
All libraries needed for this project is in requirement.txt. To install it, please run
```
pip install requirement.txt
```

---

## Repository Structure

### `231n/` — ViT Validation & Comparison
Reproduces the Vision Transformer (ViT) architecture and benchmarks it against CNN baselines.

- Implemented ViT from scratch and compared it with CNN models on **Sign MNIST** and **CIFAR-10**
- Verification notebook: [`Transformer_Captioning.ipynb`](https://github.com/hd322/CS444_Final_Project/blob/main/231n/Transformer_Captioning.ipynb)

---

### `fine_tuning/` — S3D & VAE Fine-Tuning on WLASL
Fine-tuning pre-trained video and generative models on the **WLASL (Word-Level American Sign Language)** dataset.

- **S3D** (pre-trained on Kinetics-400): fine-tuned with Full Fine-Tuning (FFT) and LoRA
- **VAE** (pre-trained on Kinetics-400 and SSv2): fine-tuned with FFT and LoRA on WLASL

---

### `SPOTER/` — Keypoint-Based Sign Recognition
Extracts hand keypoints from WLASL videos and trains sequence models on the extracted features.

- Hand keypoint extraction from WLASL video clips
- Training with **SPOTER** (Sign Pose-based Transformer) and **LSTM** on the processed keypoint data
- Utility scripts are located in the `utils/` folder

---

### `mediapipe/` — Real-Time ASL Fingerspelling Demo
A real-time ASL letter recognition pipeline built for webcam deployment.

- Started with a **ResNet-18** classifier trained on Sign MNIST, but webcam prediction performance was limited
- Replaced with a **MediaPipe + SVM** pipeline, significantly improving real-time accuracy
- Added a **stop gesture** (open palm / five fingers extended) to trigger letter-to-word translation

**How to run:**
```bash
# Step 1: Collect MediaPipe keypoint dataset
python collect.py

# Step 2: Train the SVM classifier
python train_v2.py

# Step 3: Run the real-time demo
python demo_v2.py
```

> 🎥 Demo video: *Coming soon*

---

### `I3D/` — I3D Video Classification on WLASL
A reproduction of the **I3D (Inflated 3D ConvNet)** paper applied to WLASL sign language classification.

- Reproduces the I3D architecture and trains it for ASL word classification
- Note: Due to the large model size, inference cannot be run locally; this module remains at the testing stage without a live demo

**How to run:**
```bash
# Train the model
python train.py

# Evaluate on test set
python test.py
```

---

## 📄 Report
For full details on methodology, experiments, and results, please refer to our project report.

> 📎 https://github.com/hd322/CS444_Final_Project/blob/main/Project_Report.pdf

---

## 👥 Team Members

| Name | NetID |
|------|------------|
|Hao Dong |     haod6      |
|  Ziyi Han    |    ziyihan2        |
|      |            |
|      |            |

---

## 🔗 References
- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [SPOTER](https://github.com/matyasbohacek/spoter)
- [I3D: Quo Vadis, Action Recognition?](https://arxiv.org/abs/1705.07750)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
