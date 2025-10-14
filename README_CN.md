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

## 🧩 简介
HccePose 是目前基于单幅 RGB 图像的最先进 6D 位姿估计方法。该方法提出了一种 **层次化连续坐标编码（Hierarchical Continuous Coordinate Encoding, HCCE）** 机制，将物体表面点的三个坐标分量分别编码为层次化的连续代码。通过这种层次化的编码方式，神经网络能够有效学习 2D 图像特征与物体 3D 表面坐标之间的对应关系。

在位姿估计过程中，经过 HCCE 训练的网络可根据单幅 RGB 图像预测物体的 3D 表面坐标，并结合 **Perspective-n-Point (PnP)** 算法求解 6D 位姿。与传统方法仅学习物体可见正表面不同，**HccePose(BF)** 还学习了物体背表面的 3D 坐标，从而建立了更稠密的 2D–3D 对应关系，显著提升了位姿估计精度。

值得注意的是，**HccePose(BF)** 不仅在 6D 位姿估计中实现了高精度结果，同时在基于单幅 RGB 图像的 2D 分割任务中也达到了当前最优性能。HCCE 的连续性与层次化特征显著增强了网络对物体掩膜的学习能力，相较现有方法具有显著优势。
### <img src="/show_vis/fig2.jpg" width=100%>
## 🚀 特点
### 🔹 物体预处理
- 物体的重命名与中心化处理
- 基于 [**KASAL**](https://github.com/WangYuLin-SEU/KASAL) 的旋转对称标定（支持 8 类旋转对称类型）
- 支持导出为 [**BOP format**](https://github.com/thodan/bop_toolkit) 格式

### 🔹 训练数据制备
- 基于 [**BlenderProc**](https://github.com/DLR-RM/BlenderProc) 的合成数据生成与物理渲染，用于高质量训练数据集的构建

### 🔹 2D 检测
- 基于 [**Ultralytics**](https://github.com/ultralytics) 的标签制备与检测模型训练

### 🔹 6D 位姿估计
- 生成物体 **正面** 与 **背面** 的 3D 坐标标签
- 提供基于分布式训练（DDP）的 **HccePose** 训练代码
- 支持基于 Dataloader 的测试与可视化模块
- **HccePose (YOLOv11)** 的推理与可视化:
  - 单幅 RGB 图像的推理与可视化
  - RGB 视频序列的推理与可视化

## 🔧 环境配置

```bash
apt-get update && apt-get install -y wget software-properties-common gnupg2 python3-pip

apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev

python3 -m pip install --upgrade setuptools pip

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

apt-get update apt-get install pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl ninja-build

pip install ultralytics==8.3.70 fvcore==0.1.5.post20221221 pybind11==2.12.0 trimesh==4.2.2 ninja==1.11.1.1 kornia==0.7.2 open3d==0.19.0 transformations==2024.6.1 numpy==1.26.4 opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80

pip install scipy kiwisolver matplotlib imageio pypng Cython PyOpenGL triangle glumpy Pillow vispy imgaug mathutils pyrender pytz tqdm tensorboard kasal-6d
```

## ✏️ 快速开始
针对 **Bin-Picking** 问题，本项目提供了一个基于 **HccePose** 的简易应用示例。  
为降低复现难度，示例使用的物体（由普通 3D 打印机以白色 PLA 材料打印）和相机（小米手机）均为常见易得设备。  

您可以：
- 多次打印示例物体
- 任意摆放打印物体
- 使用手机自由拍摄
- 直接利用本项目提供的权重完成 2D 检测、2D 分割与 6D 位姿估计
---
### 📦 示例文件资源  
> 请保持文件夹层级结构不变

| 类型             | 资源链接                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------ |
| 🎨 物体 3D 模型    | [models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/models)     |
| 📁 YOLOv11 权重  | [yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/yolo11)     |
| 📂 HccePose 权重 | [HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/HccePose) |
| 🖼️ 测试图片       | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs)                |
| 🎥 测试视频        | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos)            |

> ⚠️ 注意：
文件名以 train 开头的压缩包仅在训练阶段使用，快速开始部分只需下载上述测试文件。

---

### ⏳ 模型与加载器
测试时，需要从以下模块导入：
- `HccePose.tester` → 提供集成式测试器（2D 检测、分割、6D 位姿估计全流程）
- `HccePose.bop_loader` → 基于 BOP 格式的数据加载器，用于加载物体模型文件和训练数据

---

### 📸 示例测试
下图展示了实验场景：  
我们将多个白色 3D 打印物体放入碗中，并放置在白色桌面上，随后用手机拍摄。  
原始图像示例如下 👇  
<div align="center">
 <img src="/test_imgs/IMG_20251007_165718.jpg" width="40%">
</div>

该图像来自：[示例图片链接](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/test_imgs/IMG_20251007_165718.jpg)

随后，可直接使用以下脚本进行 6D 位姿估计与可视化：

```python
import cv2
import numpy as np
from HccePose.tester import Tester
from HccePose.bop_loader import bop_dataset
if __name__ == '__main__':
    dataset_path = '/root/xxxxxx/demo-bin-picking'
    bop_dataset_item = bop_dataset(dataset_path)
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    obj_id = 1
    for name in ['IMG_20251007_165718']:
        file_name = '/root/xxxxxx/test_imgs/%s.jpg'%name
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
### 🎯 可视化结果

2D 检测结果 (_show_2d.jpg)：

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_2d.jpg" width="40%"> </div>

---

网络输出结果：

- 基于 HCCE 的前后表面坐标编码

- 物体掩膜

- 解码后的 3D 坐标可视化

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_6d_vis0.jpg" width="100%"> 
<img src="/show_vis/IMG_20251007_165718_show_6d_vis1.jpg" width="100%"> </div> 

---

## 🏆 BOP榜单
<img src="/show_vis/bop-6D-loc.png" width=100%>
<img src="/show_vis/bop-2D-seg.png" width=100%>
