<h2 align="center">HccePose (BF)</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2510.10177">
    <img src="https://img.shields.io/badge/arXiv-2510.10177-B31B1B.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/SEU-WYL/HccePose">
    <img src="https://img.shields.io/badge/HuggingFace-HccePose-FFD21E.svg?logo=huggingface&logoColor=white" alt="HuggingFace">
  </a>
</p>

<p align="center">
  <a href="./README.md">English</b> | <a href="./README_CN.md">中文</a>
</p>
<!-- 
<img src="/show_vis/VID_20251011_215403.gif" width=100%>
<img src="/show_vis/VID_20251011_215255.gif" width=100%> -->

## 🧩 Introduction
HccePose represents the state-of-the-art method for 6D object pose estimation based on a single RGB image. It introduces a **Hierarchical Continuous Coordinate Encoding (HCCE)** scheme, which encodes the three coordinate components of object surface points into hierarchical continuous codes. Through this hierarchical encoding, the neural network can effectively learn the correspondence between 2D image features and 3D surface coordinates of the object.

In the pose estimation process, the network trained with HCCE predicts the 3D surface coordinates of the object from a single RGB image, which are then used in a **Perspective-n-Point (PnP)** algorithm to solve for the 6D pose. Unlike traditional methods that only learn the visible front surface of objects, **HccePose(BF)** additionally learns the 3D coordinates of the back surface, thereby constructing denser 2D–3D correspondences and significantly improving pose estimation accuracy.

It is noteworthy that **HccePose(BF)** not only achieves high-precision 6D pose estimation but also delivers state-of-the-art performance in 2D segmentation from a single RGB image. The continuous and hierarchical nature of HCCE enhances the network’s ability to learn accurate object masks, offering substantial advantages over existing methods.
### <img src="/show_vis/fig2.jpg" width=100%>
## 🚀 Features
### 🔹 Object Preprocessing
- Object renaming and centering  
- Rotation symmetry calibration (8 symmetry types) based on [**KASAL**](https://github.com/WangYuLin-SEU/KASAL)  
- Export to [**BOP format**](https://github.com/thodan/bop_toolkit)

### 🔹 Training Data Preparation
- Synthetic data generation and rendering using [**BlenderProc**](https://github.com/DLR-RM/BlenderProc)

### 🔹 2D Detection
- Label generation and model training using [**Ultralytics**](https://github.com/ultralytics)

### 🔹 6D Pose Estimation
- Preparation of **front** and **back** surface 3D coordinate labels  
- Distributed training (DDP) implementation of **HccePose**  
- Testing and visualization via **Dataloader**  
- **HccePose (YOLOv11)** inference and visualization on:
  - Single RGB images  
  - RGB videos  

## 🔧 Environment Setup

```bash
apt-get update && apt-get install -y wget software-properties-common gnupg2 python3-pip

apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev

python3 -m pip install --upgrade setuptools pip

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

apt-get update apt-get install pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl ninja-build

pip install ultralytics==8.3.70 fvcore==0.1.5.post20221221 pybind11==2.12.0 trimesh==4.2.2 ninja==1.11.1.1 kornia==0.7.2 open3d==0.19.0 transformations==2024.6.1 numpy==1.26.4 opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80

pip install scipy kiwisolver matplotlib imageio pypng Cython PyOpenGL triangle glumpy Pillow vispy imgaug mathutils pyrender pytz tqdm tensorboard kasal-6d
```

## 🏆 BOP LeaderBoards
### <img src="/show_vis/bop-6D-loc.png" width=100%>
### <img src="/show_vis/bop-2D-seg.png" width=100%>

