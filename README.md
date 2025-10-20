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
HccePose(BF) represents the state-of-the-art method for 6D object pose estimation based on a single RGB image. It introduces a **Hierarchical Continuous Coordinate Encoding (HCCE)** scheme, which encodes the three coordinate components of object surface points into hierarchical continuous codes. Through this hierarchical encoding, the neural network can effectively learn the correspondence between 2D image features and 3D surface coordinates of the object.

In the pose estimation process, the network trained with HCCE predicts the 3D surface coordinates of the object from a single RGB image, which are then used in a **Perspective-n-Point (PnP)** algorithm to solve for the 6D pose. Unlike traditional methods that only learn the visible front surface of objects, **HccePose(BF)** additionally learns the 3D coordinates of the back surface, thereby constructing denser 2D–3D correspondences and significantly improving pose estimation accuracy.

It is noteworthy that **HccePose(BF)** not only achieves high-precision 6D pose estimation but also delivers state-of-the-art performance in 2D segmentation from a single RGB image. The continuous and hierarchical nature of HCCE enhances the network’s ability to learn accurate object masks, offering substantial advantages over existing methods.
### <img src="/show_vis/fig2.jpg" width=100%>
## 🚀 Features
#### 🔹 Object Preprocessing
- Object renaming and centering  
- Rotation symmetry calibration (8 symmetry types) based on [**KASAL**](https://github.com/WangYuLin-SEU/KASAL)  
- Export to [**BOP format**](https://github.com/thodan/bop_toolkit)

#### 🔹 Training Data Preparation
- Synthetic data generation and rendering using [**BlenderProc**](https://github.com/DLR-RM/BlenderProc)

#### 🔹 2D Detection
- Label generation and model training using [**Ultralytics**](https://github.com/ultralytics)

#### 🔹 6D Pose Estimation
- Preparation of **front** and **back** surface 3D coordinate labels  
- Distributed training (DDP) implementation of **HccePose(BF)**  
- Testing and visualization via **Dataloader**  
- **HccePose(BF) (YOLOv11)** inference and visualization on:
  - Single RGB images  
  - RGB videos  

## 🔧 Environment Setup
Download the HccePose(BF) Project and Unzip BOP-related Toolkits
```bash
# Clone the project
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose

# Unzip toolkits
unzip bop_toolkit.zip
unzip blenderproc.zip
```
Configure Ubuntu System Environment

> ⚠️ A GPU driver with EGL support must be pre-installed.
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

This project provides a simple **HccePose(BF)-based** application example for the **Bin-Picking** task.  
To reduce reproduction difficulty, both the objects (3D printed with standard white PLA material) and the camera (Xiaomi smartphone) are easily accessible devices.

You can:
- Print the sample object multiple times  
- Randomly place the printed objects  
- Capture photos freely using your phone  
- Directly perform **2D detection**, **2D segmentation**, and **6D pose estimation** using the pretrained weights provided in this project  

---


> Please keep the folder hierarchy unchanged.

| Type | Resource Link |
|------|----------------|
| 🎨 Object 3D Models | [demo-bin-picking/models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/models) |
| 📁 YOLOv11 Weights | [demo-bin-picking/yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/yolo11) |
| 📂 HccePose Weights | [demo-bin-picking/HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/HccePose) |
| 🖼️ Test Images | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs) |
| 🎥 Test Videos | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos) |

> ⚠️ Note:  
Files beginning with `train_` are only required for training.  
For this **Quick Start** section, only the above test files are needed.

---

#### ⏳ Model and Loader
During testing, import the following modules:
- `HccePose.tester` → Integrated testing module covering **2D detection**, **segmentation**, and **6D pose estimation**.  
- `HccePose.bop_loader` → BOP-format dataset loader for loading object models and training data.

---

#### 📸 Example Test
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
---

#### 🎯 Visualization Results

2D Detection Result (_show_2d.jpg):

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_2d.jpg" width="40%"> </div>


Network Outputs:

- HCCE-based front and back surface coordinate encodings

- Object mask

- Decoded 3D coordinate visualizations

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_6d_vis0.jpg" width="100%"> 
<img src="/show_vis/IMG_20251007_165718_show_6d_vis1.jpg" width="100%"> </div>

--- 

#### 🎥 6D Pose Estimation in Videos

The single-frame pose estimation pipeline can be easily extended to video sequences, enabling continuous-frame 6D pose estimation, as shown in the following example:

```python
import cv2, os, sys
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester

if __name__ == '__main__':
    
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_video_path = os.path.join(current_dir, 'test_videos')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    for name in ['VID_20251009_141247']:
        file_name = os.path.join(test_video_path, '%s.mp4'%name)
        cap = cv2.VideoCapture(file_name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_1 = None
        out_2 = None
        cam_K = np.array([
            [1.63235512e+03, 0.00000000e+00, 9.74032712e+02],
            [0.00000000e+00, 1.64159967e+03, 5.14229781e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results_dict = Tester_item.perdict(cam_K, frame, [obj_id],
                                                            conf = 0.85, confidence_threshold = 0.85)
            fps_hccepose = 1 / results_dict['time']
            show_6D_vis1 = results_dict['show_6D_vis1']
            show_6D_vis1[show_6D_vis1 < 0] = 0
            show_6D_vis1[show_6D_vis1 > 255] = 255
            if out_1 is None:
                out_1 = cv2.VideoWriter(
                    file_name.replace('.mp4', '_show_1.mp4'),
                    fourcc,
                    fps,
                    (show_6D_vis1.shape[1], show_6D_vis1.shape[0])
                )
            out_1.write(show_6D_vis1.astype(np.uint8))
            show_6D_vis2 = results_dict['show_6D_vis2']
            show_6D_vis2[show_6D_vis2 < 0] = 0
            show_6D_vis2[show_6D_vis2 > 255] = 255
            if out_2 is None:
                out_2 = cv2.VideoWriter(
                    file_name.replace('.mp4', '_show_2.mp4'),
                    fourcc,
                    fps,
                    (show_6D_vis2.shape[1], show_6D_vis2.shape[0])
                )
            cv2.putText(show_6D_vis2, "FPS: {0:.2f}".format(fps_hccepose), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
            out_2.write(show_6D_vis2.astype(np.uint8))
        cap.release()
        out_1.release()
        out_2.release()
    pass
```

--- 

#### 🎯 Visualization Results

**Original Video:**
<img src="/show_vis/VID_20251009_141247.gif" width=100%>

**Detection Results:**
<img src="/show_vis/VID_20251009_141247_vis.gif" width=100%>

---

In addition, by passing a list of multiple object IDs to `HccePose.tester`, multi-object 6D pose estimation can also be achieved.  

> Please keep the folder hierarchy unchanged.

| Type | Resource Link |
|------|----------------|
| 🎨 Object 3D Models | [demo-tex-objs/models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/models) |
| 📁 YOLOv11 Weights | [demo-tex-objs/yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/yolo11) |
| 📂 HccePose Weights | [demo-tex-objs/HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/HccePose) |
| 🖼️ Test Images | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs) |
| 🎥 Test Videos | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos) |

> ⚠️ Note:  
Files beginning with `train_` are only required for training.  
For this **Quick Start** section, only the above test files are needed.

**Original Video:**
<img src="/show_vis/VID_20251009_141731.gif" width=100%>

**Detection Results:**
<img src="/show_vis/VID_20251009_141731_vis.gif" width=100%>

---

## 🧪 BOP Challenge Testing

You can use the script [`s4_p2_test_bf_pbr_bop_challenge.py`](/s4_p2_test_bf_pbr_bop_challenge.py) to evaluate **HccePose(BF)** across the seven core BOP datasets.

---

#### Pretrained Weights

| Dataset | Weights Link |
|----------|---------------|
| **LM-O** | [Hugging Face - LM-O](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/lmo/HccePose) |
| **YCB-V** | [Hugging Face - YCB-V](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/ycbv/HccePose) |
| **T-LESS** | [Hugging Face - T-LESS](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/tless/HccePose) |
| **TUD-L** | [Hugging Face - TUD-L](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/tudl/HccePose) |
| **HB** | [Hugging Face - HB](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/hb/HccePose) |
| **ITODD** | [Hugging Face - ITODD](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/itodd/HccePose) |
| **IC-BIN** | [Hugging Face - IC-BIN](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/icbin/HccePose) |

---

#### Example: LM-O Dataset

As an example, we evaluated **HccePose(BF)** on the widely used **LM-O dataset** from the BOP benchmark. We adopted the [default 2D detector](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task1.zip) (GDRNPP) from the **BOP 2023 Challenge** and obtained the following output files:

- 2D segmentation results: [seg2d_lmo.json](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/lmo/seg2d_lmo.json)
- 6D pose results: [det6d_lmo.csv](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/lmo/det6d_lmo.csv)

These two files were submitted on **October 20, 2025**. The results are shown below.  
The **6D localization score** remains consistent with the 2024 submission,  
while the **2D segmentation score** improved by **0.002**, thanks to the correction of minor implementation bugs.

### <img src="/show_vis/BOP-website-lmo.png" width=100%>

---

#### ⚙️ Notes

- If some pretrained weights show an iteration count of `0`, this is **not an error**. All **HccePose(BF)** weights are fine-tuned from the standard HccePose model trained using only the front surface. In some cases, the initial weights already achieve optimal performance.

---

## 📅 Update Plan

We are currently organizing and updating the following modules:

- 📁 ~~HccePose(BF) weights for the seven core BOP datasets~~

- 🧪 ~~BOP Challenge testing pipeline~~

- 🔁 6D pose inference via inter-frame tracking

- 🏷️ Real-world 6D pose dataset preparation based on HccePose(BF)

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