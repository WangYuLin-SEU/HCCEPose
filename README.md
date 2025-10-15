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
### Download the HccePose Project and Unzip BOP-related Toolkits
```bash
# Clone the project
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose

# Unzip toolkits
unzip bop_toolkit.zip
unzip blenderproc.zip
```
### Configure Ubuntu System Environment

⚠️ A GPU driver with EGL support must be pre-installed.
```bash
apt-get update && apt-get install -y wget software-properties-common gnupg2 python3-pip

apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev

python3 -m pip install --upgrade setuptools pip

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

apt-get update apt-get install pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl ninja-build

pip install ultralytics==8.3.70 fvcore==0.1.5.post20221221 pybind11==2.12.0 trimesh==4.2.2 ninja==1.11.1.1 kornia==0.7.2 open3d==0.19.0 transformations==2024.6.1 numpy==1.26.4 opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80

pip install scipy kiwisolver matplotlib imageio pypng Cython PyOpenGL triangle glumpy Pillow vispy imgaug mathutils pyrender pytz tqdm tensorboard kasal-6d
```

## ✏️ Quick Start

This project provides a simple **HccePose-based** application example for the **Bin-Picking** task.  
To reduce reproduction difficulty, both the objects (3D printed with standard white PLA material) and the camera (Xiaomi smartphone) are easily accessible devices.

You can:
- Print the sample object multiple times  
- Randomly place the printed objects  
- Capture photos freely using your phone  
- Directly perform **2D detection**, **2D segmentation**, and **6D pose estimation** using the pretrained weights provided in this project  

---

### 📦 Example Files  
> Please keep the folder hierarchy unchanged.

| Type | Resource Link |
|------|----------------|
| 🎨 Object 3D Models | [models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/models) |
| 📁 YOLOv11 Weights | [yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/yolo11) |
| 📂 HccePose Weights | [HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/HccePose) |
| 🖼️ Test Images | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs) |
| 🎥 Test Videos | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos) |

> ⚠️ Note:  
Files beginning with `train_` are only required for training.  
For this **Quick Start** section, only the above test files are needed.

---

### ⏳ Model and Loader
During testing, import the following modules:
- `HccePose.tester` → Integrated testing module covering **2D detection**, **segmentation**, and **6D pose estimation**.  
- `HccePose.bop_loader` → BOP-format dataset loader for loading object models and training data.

---

### 📸 Example Test
The following image shows the experimental setup:  
Several white 3D-printed objects are placed inside a bowl on a white table, then photographed with a mobile phone.  

Example input image 👇  
<div align="center">
 <img src="/test_imgs/IMG_20251007_165718.jpg" width="40%">
</div>

Source image: [Example Link](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/test_imgs/IMG_20251007_165718.jpg)

You can directly use the following script for **6D pose estimation** and visualization:

```python
import cv2, os, sys
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester
if __name__ == '__main__':

    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_img_path = os.path.join(current_dir, 'test_imgs')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    
    for name in ['IMG_20251007_165718']:
        file_name = os.path.join(test_img_path, '%s.jpg'%name)
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_RGB2BGR)
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        results_dict = Tester_item.perdict(cam_K, image, [obj_id],
                                                        conf = 0.85, confidence_threshold = 0.85)
        cv2.imwrite(file_name.replace('.jpg','_show_2d.jpg'), results_dict['show_2D_results'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis0.jpg'), results_dict['show_6D_vis0'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis1.jpg'), results_dict['show_6D_vis1'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis2.jpg'), results_dict['show_6D_vis2'])
    pass
```

### 🎯 Visualization Results

2D Detection Result (_show_2d.jpg):

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_2d.jpg" width="40%"> </div>

---

Network Outputs:

- HCCE-based front and back surface coordinate encodings

- Object mask

- Decoded 3D coordinate visualizations

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_6d_vis0.jpg" width="100%"> 
<img src="/show_vis/IMG_20251007_165718_show_6d_vis1.jpg" width="100%"> </div>

---

## 📅 Update Plan

We are currently organizing and updating the following modules:

- 📁 HccePose weights for the seven core BOP datasets

- 🧪 BOP Challenge testing pipeline

- 🔁 6D pose inference via inter-frame tracking

- 🏷️ Real-world 6D pose dataset preparation based on HccePose

- ⚙️ PBR + Real training workflow

- 📘 Tutorials on object preprocessing, data rendering, and model training

All components are expected to be completed by the end of 2025, with continuous daily updates whenever possible.

---

## 🏆 BOP LeaderBoards
### <img src="/show_vis/bop-6D-loc.png" width=100%>
### <img src="/show_vis/bop-2D-seg.png" width=100%>


***
If you find our work useful, please cite it as follows: 
```bibtex
@ARTICLE{KASAL,
  author = {Yulin Wang, Mengting Hu, Hongli Li, and Chen Luo},
  title  = {HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation}, 
  journal= {2025 IEEE/CVF International Conference on Computer Vision}, 
  year   = {2025}
}
```