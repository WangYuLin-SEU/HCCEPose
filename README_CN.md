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
##### 🔹 物体预处理
- 物体的重命名与中心化处理
- 基于 [**KASAL**](https://github.com/WangYuLin-SEU/KASAL) 的旋转对称标定（支持 8 类旋转对称类型）
- 支持导出为 [**BOP format**](https://github.com/thodan/bop_toolkit) 格式

##### 🔹 训练数据制备
- 基于 [**BlenderProc**](https://github.com/DLR-RM/BlenderProc) 的合成数据生成与物理渲染，用于高质量训练数据集的构建

##### 🔹 2D 检测
- 基于 [**Ultralytics**](https://github.com/ultralytics) 的标签制备与检测模型训练

##### 🔹 6D 位姿估计
- 生成物体 **正面** 与 **背面** 的 3D 坐标标签
- 提供基于分布式训练（DDP）的 **HccePose** 训练代码
- 支持基于 Dataloader 的测试与可视化模块
- **HccePose (YOLOv11)** 的推理与可视化:
  - 单幅 RGB 图像的推理与可视化
  - RGB 视频序列的推理与可视化

## 🔧 环境配置
下载 HccePose 项目并解压BOP等工具包
```bash
# 克隆项目
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose

# 解压工具包
unzip bop_toolkit.zip
unzip blenderproc.zip
```
配置 Ubuntu 系统环境

⚠️ 需要提前安装 带有 EGL 支持的显卡驱动
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

> 请保持文件夹层级结构不变

| 类型             | 资源链接                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------ |
| 🎨 物体 3D 模型    | [demo-bin-picking/models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/models)     |
| 📁 YOLOv11 权重  | [demo-bin-picking/yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/yolo11)     |
| 📂 HccePose 权重 | [demo-bin-picking/HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/HccePose) |
| 🖼️ 测试图片       | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs)                |
| 🎥 测试视频        | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos)            |

> ⚠️ 注意：
文件名以 train 开头的压缩包仅在训练阶段使用，快速开始部分只需下载上述测试文件。

---

##### ⏳ 模型与加载器
测试时，需要从以下模块导入：
- `HccePose.tester` → 提供集成式测试器（2D 检测、分割、6D 位姿估计全流程）
- `HccePose.bop_loader` → 基于 BOP 格式的数据加载器，用于加载物体模型文件和训练数据

---

##### 📸 示例测试
下图展示了实验场景：  
我们将多个白色 3D 打印物体放入碗中，并放置在白色桌面上，随后用手机拍摄。  
原始图像示例如下 👇  
<div align="center">
 <img src="/test_imgs/IMG_20251007_165718.jpg" width="40%">
</div>

该图像来自：[示例图片链接](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/test_imgs/IMG_20251007_165718.jpg)

随后，可直接使用以下脚本进行 6D 位姿估计与可视化：

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

##### 🎯 可视化结果

2D 检测结果 (_show_2d.jpg)：

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_2d.jpg" width="40%"> </div>


网络输出结果：

- 基于 HCCE 的前后表面坐标编码

- 物体掩膜

- 解码后的 3D 坐标可视化

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_6d_vis0.jpg" width="100%"> 
<img src="/show_vis/IMG_20251007_165718_show_6d_vis1.jpg" width="100%"> </div> 

---
##### 🎥 视频的6D位姿估计
基于单帧图像的位姿估计流程，可以轻松扩展至视频序列，从而实现对连续帧的 6D 位姿估计，代码如下：
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

##### 🎯 可视化结果
**原始视频：**
<img src="/show_vis/VID_20251009_141247.gif" width=100%>

**检测结果：**
<img src="/show_vis/VID_20251009_141247_vis.gif" width=100%>

---

此外，通过向`HccePose.tester`传入多个物体的id列表，即可实现对多物体的 6D 位姿估计。

> 请保持文件夹层级结构不变

| 类型             | 资源链接                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------ |
| 🎨 物体 3D 模型    | [demo-tex-objs/models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/models)     |
| 📁 YOLOv11 权重  | [demo-tex-objs/yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/yolo11)     |
| 📂 HccePose 权重 | [demo-tex-objs/HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/HccePose) |
| 🖼️ 测试图片       | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs)                |
| 🎥 测试视频        | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos)            |

> ⚠️ 注意：
文件名以 train 开头的压缩包仅在训练阶段使用，快速开始部分只需下载上述测试文件。

**原始视频：**
<img src="/show_vis/VID_20251009_141731.gif" width=100%>

**检测结果：**
<img src="/show_vis/VID_20251009_141731_vis.gif" width=100%>

---

## 📅 更新计划

我们目前正在整理和更新以下模块：

- 📁 七个核心 BOP 数据集的 HccePose 权重文件

- 🧪 BOP 挑战测试流程

- 🔁 基于前后帧跟踪的 6D 位姿推理

- 🏷️ 基于 HccePose 的真实场景 6D 位姿数据集制备

- ⚙️ PBR + Real 训练流程

- 📘 关于物体预处理、数据渲染及模型训练的教程

预计所有模块将在 2025 年底前完成，并尽可能 每日持续更新。

---

## 🏆 BOP榜单
<img src="/show_vis/bop-6D-loc.png" width=100%>
<img src="/show_vis/bop-2D-seg.png" width=100%>


***
如果您觉得我们的工作有帮助，请按以下方式引用：
```bibtex
@ARTICLE{KASAL,
  author = {Yulin Wang, Mengting Hu, Hongli Li, and Chen Luo},
  title  = {HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation}, 
  journal= {2025 IEEE/CVF International Conference on Computer Vision}, 
  year   = {2025}
}
```