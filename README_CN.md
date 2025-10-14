### <center> HccePose(BF) </center>

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

<img src="/show_vis/VID_20251011_215403.gif" width=100%>
<img src="/show_vis/VID_20251011_215255.gif" width=100%>

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

## 🏆 BOP榜单
### <img src="/show_vis/bop-6D-loc.png" width=100%>
### <img src="/show_vis/bop-2D-seg.png" width=100%>
