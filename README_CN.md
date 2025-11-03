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
HccePose(BF) 提出了一种 **层次化连续坐标编码（Hierarchical Continuous Coordinate Encoding, HCCE）** 机制，将物体表面点的三个坐标分量分别编码为层次化的连续代码。通过这种层次化的编码方式，神经网络能够有效学习 2D 图像特征与物体 3D 表面坐标之间的对应关系，也显著增强了网络对物体掩膜的学习能力。与传统方法仅学习物体可见正表面不同，**HccePose(BF)** 还学习了物体背表面的 3D 坐标，从而建立了更稠密的 2D–3D 对应关系，显著提升了位姿估计精度。

### <img src="/show_vis/fig2.jpg" width=100%>

## ✨ 更新
--- 
- ⚠️ 注意：所有路径都必须使用绝对路径，以避免运行时错误。
- 2025.10.27: 我们发布了 cc0textures-512，这是原版 CC0Textures（44GB） 的轻量替代版本，体积仅 600MB。 👉 [点此下载](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/cc0textures-512.zip)
- 2025.10.28: s4_p1_gen_bf_labels.py 已更新。若数据集中不存在 camera.json，脚本将自动创建一个默认文件。
---
## 🔧 环境配置

<details>
<summary>配置细节</summary>

下载 HccePose(BF) 项目并解压BOP等工具包
```bash
# 克隆项目
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose

# 解压工具包
unzip bop_toolkit.zip
unzip blenderproc.zip
```
配置 Ubuntu 系统环境 (Python 3.10)

⚠️ 需要提前安装 带有 EGL 支持的显卡驱动
```bash
apt-get update && apt-get install -y wget software-properties-common gnupg2 python3-pip

apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

apt-get update && apt-get install pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl ninja-build

pip install ultralytics==8.3.70 fvcore==0.1.5.post20221221 pybind11==2.12.0 trimesh==4.2.2 ninja==1.11.1.1 kornia==0.7.2 open3d==0.19.0 transformations==2024.6.1 numpy==1.26.4 opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80

pip install scipy kiwisolver matplotlib imageio pypng Cython PyOpenGL triangle glumpy Pillow vispy imgaug mathutils pyrender pytz tqdm tensorboard kasal-6d rich h5py

pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

apt-get install libsm6 libxrender1 libxext-dev

python -c "import imageio; imageio.plugins.freeimage.download()"

pip install -U "huggingface_hub[hf_transfer]"

```

</details>

---


## 🧱 自定义数据集及训练

#### 🎨 物体预处理

<details>
<summary>点击展开</summary>

以 [**`demo-bin-picking`**](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking) 数据集为例，我们首先使用 **SolidWorks** 设计物体模型，并导出为 STL 格式的三维网格文件。  
STL 文件下载链接：🔗 https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/raw-demo-models/multi-objs/board.STL

<img src="/show_vis/Design-3DMesh.jpg" width=100%>

随后，在 **MeshLab** 中导入该 STL 文件，并使用 **`Vertex Color Filling`** 工具为模型表面着色。

<img src="/show_vis/color-filling.png" width=100%>
<img src="/show_vis/color-filling-2.png" width=100%>

接着，将物体模型以 **非二进制 PLY 格式** 导出，并确保包含顶点颜色与法向量信息。

<img src="/show_vis/export-3d-mesh-ply.png" width=100%>

导出的模型中心通常与坐标系原点不重合（如下图所示）：

<img src="/show_vis/align-center.png" width=100%>

为解决模型中心偏移问题，可使用脚本 **`s1_p1_obj_rename_center.py`**：该脚本会加载 PLY 文件，将模型中心对齐至坐标系原点，并根据 BOP 规范重命名文件。用户需手动设置非负整数参数 **`obj_id`**，每个物体对应唯一编号。  

例如：

| **`input_ply`** | **`obj_id`** | **`output_ply`** |
| :---: | :---: | :---: |
| **`board.ply`** | **`1`** | **`obj_000001.ply`** |
| **`board.ply`** | **`2`** | **`obj_000002.ply`** |


当所有物体完成中心化与重命名后，将这些文件放入名为 **`models`** 的文件夹中，目录结构如下：

```bash
数据集名称
|--- models
      |--- obj_000001.ply
      ...
      |--- obj_000015.ply
```

---

</details>

#### 🌀 物体旋转对称分析

<details>
<summary>点击展开</summary>

在位姿估计任务中，许多物体存在多种旋转对称性，如圆柱、圆锥或多面体旋转对称。对于这些旋转对称物体，需要使用 KASAL 工具生成符合 BOP 规范的旋转对称先验。

KASAL 项目地址：🔗 https://github.com/WangYuLin-SEU/KASAL

安装命令：

```bash
pip install kasal-6d
```

运行以下代码可启动 **KASAL 图形界面**：

```python
from kasal.app.polyscope_app import app
mesh_path = 'demo-bin-picking'
app(mesh_path)
```

KASAL 会自动遍历 **`mesh_path`** 文件夹下所有 PLY 或 OBJ 文件（不加载 **`_sym.ply`** 等效果文件）。

<img src="/show_vis/kasal-1.png" width=100%>

在使用界面中：
* 下拉 **`Symmetry Type`** 选择旋转对称类型
* 对于 n 阶棱锥或棱柱旋转对称，需设置 **`N (n-fold)`**
* 对纹理旋转对称物体，勾选 **`ADI-C`**
* 若结果不准确，可通过 **`axis xyz`** 手动强制拟合

KASAL 将旋转对称划分为 **8 种类型**。若选择错误类型，将在可视化中显示异常，从而可辅助判断设置是否正确。

<img src="/show_vis/kasal-2.png" width=100%>

点击 **`Cal Current Obj`** 可计算当前物体的旋转对称轴，旋转对称先验将保存为 **`_sym_type.json`** 文件，例如：
* 旋转对称先验文件：**`obj_000001_sym_type.json`**
* 可视化文件：**`obj_000001_sym.ply`**

---
</details>

#### 🧾 BOP 格式模型信息生成

<details>
<summary>点击展开</summary>

运行脚本 **`s1_p3_obj_infos.py`**，该脚本会遍历 **`models`** 文件夹下所有满足 BOP 规范的 **`ply`** 文件及其对应的旋转对称文件，并最终生成标准的 **`models_info.json`** 文件。

生成后的目录结构如下：

```bash
数据集名称
|--- models
      |--- models_info.json
      |--- obj_000001.ply
      ...
      |--- obj_000015.ply
```

---
</details>


#### 🔥 渲染 PBR 数据集

<details>
<summary>点击展开</summary>

在 **BlenderProc** 的基础上，我们改写了一个用于渲染新数据集的脚本 **`s2_p1_gen_pbr_data.py`**。直接通过 Python 调用该脚本可能会导致 **内存泄漏（memory leak）**，随着渲染周期的增长，内存占用会逐渐增加，从而显著降低渲染效率。为了解决这一问题，我们提供了一个 **Shell 脚本** —— **`s2_p1_gen_pbr_data.sh`**，用于循环调用 **`s2_p1_gen_pbr_data.py`**，以此有效缓解内存累积问题，并显著提升渲染效率。此外，我们还针对 BlenderProc 进行了部分代码微调，以更好地适配新数据集的 PBR 数据制备流程。  

---

#### 渲染前准备

在渲染 PBR 数据前，需要使用 **`s2_p0_download_cc0textures.py`** 下载 **CC0Textures** 材质库。下载完成后，文件夹结构应如下所示：
```
HCCEPose
|--- s2_p0_download_cc0textures.py
|--- cc0textures
```

---

**cc0textures** 约占用 **44GB** 硬盘空间，体积较大。
为降低存储需求，我们制作了一个轻量级替代版本 **cc0textures-512**，其大小仅约 **600MB**。
下载链接如下：
👉 https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/cc0textures-512.zip

在运行渲染脚本时，只需将 **`cc0textures`** 的路径替换为 **`cc0textures-512`**，即可直接使用该轻量材质库。
（可以仅下载 **`cc0textures-512`**，无需下载原始的 **`cc0textures`**。）

---

#### 渲染执行

**`s2_p1_gen_pbr_data.py`** 用于生成 PBR 数据，该脚本基于 [BlenderProc2](https://github.com/DLR-RM/BlenderProc) 进行了改写。

执行命令如下：

```bash
cd HCCEPose
chmod +x s2_p1_gen_pbr_data.sh
nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &
```

**文件结构说明**

按照上述命令运行后，程序会：
- 调用 **`xxx/xxx/cc0textures`** 中的材质库；
- 使用 **`xxx/xxx/demo-bin-picking/models`** 文件夹下的物体模型；
- 在 **`xxx/xxx/demo-bin-picking`** 文件夹下生成 **42 个文件夹**，每个文件夹包含 **1000 帧 PBR 渲染图像**。

最终生成的文件结构如下：
```
demo-bin-picking
|--- models
|--- train_pbr
      |--- 000000
      |--- 000001
      ...
      |--- 000041
```

---

</details>

#### 🚀 训练 2D 检测器

<details>
<summary>点击展开</summary>

在 6D 位姿估计任务中，通常需要首先通过 **2D 检测器** 来确定物体的包围盒区域，并基于包围盒图像进一步推断物体的 **6D 位姿**。相比直接从整幅图像中预测 6D 位姿，**“2D 检测 + 6D 位姿估计”** 的两阶段方法在精度与稳定性方面表现更优。因此，本项目为 **HccePose(BF)** 配备了一个基于 **YOLOv11** 的 2D 检测器。  

以下将介绍如何将 **BOP 格式的 PBR 训练数据** 转换为 YOLO 可用的数据格式，并进行 YOLOv11 的训练。

---

####  转换 BOP PBR 训练数据为 YOLO 训练数据

为实现 BOP 格式 PBR 数据与 YOLO 数据的自动转换，我们提供了 **`s3_p1_prepare_yolo_label.py`** 脚本。在指定路径 **`xxx/xxx/demo-bin-picking`** 后运行该脚本，程序将在 **`demo-bin-picking`** 文件夹下生成一个新的 **`yolo11`** 文件夹。

生成后的目录结构如下：
```
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
      |--- train_obj_s
            |--- images
            |--- labels
            |--- yolo_configs
                |--- data_objs.yaml
            |--- autosplit_train.txt
            |--- autosplit_val.txt
```

其中：  
- **`images`** → 存放 2D 训练图像  
- **`labels`** → 存放 2D BBox 标签文件  
- **`data_objs.yaml`** → YOLO 训练配置文件  
- **`autosplit_train.txt`** → 训练集样本列表  
- **`autosplit_val.txt`** → 验证集样本列表  

---

#### 训练 YOLOv11 检测器

为训练 YOLOv11 检测器，我们提供了 **`s3_p2_train_yolo.py`** 脚本。 在指定路径 **`xxx/xxx/demo-bin-picking`** 后运行该脚本，  程序将自动训练 YOLOv11，并保存最佳权重文件 **`yolo11-detection-obj_s.pt`**。  

训练完成后，文件结构如下：

```
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
      |--- train_obj_s
            |--- detection
                |--- obj_s
                    |--- yolo11-detection-obj_s.pt
            |--- images
            |--- labels
            |--- yolo_configs
                |--- data_objs.yaml
            |--- autosplit_train.txt
            |--- autosplit_val.txt
```

---

#### ⚠️ 注意事项
**`s3_p2_train_yolo.py`** 会循环扫描 **`detection`** 文件夹下的 **`yolo11-detection-obj_s.pt`** 文件。
此机制能够在训练程序因意外中断后自动恢复训练，特别适用于云服务器等不便实时监控训练进度的环境，可避免设备长时间空置造成的资源浪费。
但若需要重新开始训练，请务必先删除 **`yolo11-detection-obj_s.pt`** 文件，否则该文件的存在会使程序继续从中断点恢复训练，而无法重新初始化。

---
</details>



#### 🧩 物体正背面标签制备

<details>
<summary>点击展开</summary>

在 **HccePose(BF)** 中，网络同时学习物体的 **正表面 3D 坐标** 与 **背表面 3D 坐标**。为生成这些正背面标签，我们分别渲染物体的正面和背面深度图。

在渲染物体正面深度图时，通过设置 **`gl.glDepthFunc(gl.GL_LESS)`** 保留最小深度值（即距离相机最近的表面），这些表面被定义为物体的 **正面**，该定义参考了渲染流程中“正背面剔除”的概念。相应地，在渲染背面深度图时，设置 **`gl.glDepthFunc(gl.GL_GREATER)`** 保留最大的深度值（即距离相机最远的表面），这些表面被定义为物体的 **背面**。最终，基于深度图与物体的 6D 位姿真值，可生成正背面的 **3D 坐标标签图**。

---

#### 旋转对称处理与真值校正

对于旋转对称物体，我们将 **离散** 与 **连续旋转对称** 统一表示为旋转对称矩阵集合，并基于该集合与物体真值位姿计算新的真值位姿集合。为保持 6D 位姿标签的唯一性，从中选取与单位矩阵 **L2 距离最小** 的真值位姿作为最终标签。

此外，依据相机成像原理，当物体发生平移而旋转不变时，在固定视角下会产生“**视觉上的旋转**”。对于旋转对称物体，这种视觉旋转会导致错误的 3D 坐标标签图。为修正此类误差，我们根据渲染得到的深度图计算物体的 3D 坐标，并使用 **RANSAC PnP** 对旋转进行校正。

---

#### 批量标签生成

基于上述思路，我们实现了 **`s4_p1_gen_bf_labels.py`**，该脚本支持多进程渲染，能够批量生成物体正背面的 3D 坐标标签图。指定数据集路径 **`/root/xxxxxx/demo-bin-picking`** 以及其中的文件夹 **`train_pbr`**，运行脚本后将生成两个新文件夹：  

- **`train_pbr_xyz_GT_front`**：存储正面 3D 坐标标签图  
- **`train_pbr_xyz_GT_back`**：存储背面 3D 坐标标签图  

目录结构如下：

```
demo-bin-picking
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```

以下示例展示了三张对应的图像：  
原始渲染图、正面 3D 坐标标签图、背面 3D 坐标标签图。
<p align="center">
  <img src="/show_vis/000000.jpg" width="32%">
  <img src="/show_vis/000000_000000-f.png" width="32%">
  <img src="/show_vis/000000_000000-b.png" width="32%">
</p>

---

</details>

#### 🚀 训练 HccePose(BF)

<details>
<summary>点击展开</summary>

在训练 **HccePose(BF)** 时，需要为每个物体单独训练一个对应的权重模型。  
通过 **`s4_p2_train_bf_pbr.py`** 脚本，可以实现 **批量物体的多卡训练**。

以 `demo-tex-objs` 数据集为例，训练完成后的文件夹结构如下：
```
demo-tex-objs
|--- HccePose
    |--- obj_01
    ...
    |--- obj_10
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```

在使用 **`s4_p2_train_bf_pbr.py`** 时，可通过参数 **`ide_debug`** 切换单卡与多卡模式：  
- 当 `ide_debug=True` 时，仅使用 **单卡**，适合在 IDE 中调试；  
- 当 `ide_debug=False` 时，启用 **DDP（分布式数据并行）训练** 模式。  

在 VSCode 等 IDE 中直接挂起 DDP 训练可能会引发通讯问题，因此推荐使用以下命令在后台运行多卡训练：
```
screen -S train_ddp
nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
``` 

如果仅需单卡运行或调试，可直接使用：

```
nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
```  


---

#### 训练范围设置

若需训练多个物体，可通过 **`start_obj_id`** 与 **`end_obj_id`** 参数设置物体 ID 范围。 例如，`start_obj_id=1` 且 `end_obj_id=5` 时，脚本会依次训练 `obj_000001.ply` 至 `obj_000005.ply`。若仅训练单个物体，则将两者设置为相同的数字即可。

此外，可根据实际需求修改 **`total_iteration`**，其默认值为 `50000`。在 DDP 训练中，实际训练的样本数量可通过以下公式计算：
```
total samples = total iteration × batch size × GPU number
```

---

</details>

---


## ✏️ 快速开始
针对 **Bin-Picking** 问题，本项目提供了一个基于 **HccePose(BF)** 的简易应用示例。  
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

#### ⏳ 模型与加载器
测试时，需要从以下模块导入：
- **`HccePose.tester`** → 提供集成式测试器（2D 检测、分割、6D 位姿估计全流程）
- **`HccePose.bop_loader`** → 基于 BOP 格式的数据加载器，用于加载物体模型文件和训练数据

---

#### 📸 示例测试
下图展示了实验场景：  
<details>
<summary>点击展开</summary>
我们将多个白色 3D 打印物体放入碗中，并放置在白色桌面上，随后用手机拍摄。  
原始图像示例如下 👇  
<div align="center">
 <img src="/test_imgs/IMG_20251007_165718.jpg" width="40%">
</div>

该图像来自：[示例图片链接](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/test_imgs/IMG_20251007_165718.jpg)

</details>

随后，可直接使用以下脚本进行 6D 位姿估计与可视化：

<details>
<summary>点击展开代码</summary>

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
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    
    for name in ['IMG_20251007_165718']:
        file_name = os.path.join(test_img_path, '%s.jpg'%name)
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_RGB2BGR)
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        results_dict = Tester_item.predict(cam_K, image, [obj_id],
                                                        conf = 0.85, confidence_threshold = 0.85)
        cv2.imwrite(file_name.replace('.jpg','_show_2d.jpg'), results_dict['show_2D_results'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis0.jpg'), results_dict['show_6D_vis0'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis1.jpg'), results_dict['show_6D_vis1'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis2.jpg'), results_dict['show_6D_vis2'])
    pass
```

</details>

---

#### 🎯 可视化结果

2D 检测结果 (_show_2d.jpg)：

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_2d.jpg" width="40%"> </div>


网络输出结果：

- 基于 HCCE 的前后表面坐标编码

- 物体掩膜

- 解码后的 3D 坐标可视化

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_6d_vis0.jpg" width="100%"> 
<img src="/show_vis/IMG_20251007_165718_show_6d_vis1.jpg" width="100%"> </div> 

---

#### 💫 如果觉得本教程对你有帮助

欢迎给项目点个 ⭐️ 支持一下！你的 Star 是我们持续完善文档和更新代码的最大动力 🙌

---
#### 🎥 视频的6D位姿估计

<details>
<summary>具体内容</summary>

基于单帧图像的位姿估计流程，可以轻松扩展至视频序列，从而实现对连续帧的 6D 位姿估计，代码如下：
<details>
<summary>点击展开代码</summary>

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
            results_dict = Tester_item.predict(cam_K, frame, [obj_id],
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

</details>

---

#### 🎯 可视化结果
**原始视频：**
<img src="/show_vis/VID_20251009_141247.gif" width=100%>

**检测结果：**
<img src="/show_vis/VID_20251009_141247_vis.gif" width=100%>

---

此外，通过向**`HccePose.tester`**传入多个物体的id列表，即可实现对多物体的 6D 位姿估计。

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

</details>

---




## 🧪 BOP挑战测试

您可以使用脚本[**`s4_p2_test_bf_pbr_bop_challenge.py`**](/s4_p2_test_bf_pbr_bop_challenge.py)来测试 **HccePose** 在七个 BOP 核心数据集上的表现。

#### 训练权重文件

| 数据集 | 权重链接 |
|----------|---------------|
| **LM-O** | [Hugging Face - LM-O](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/lmo/HccePose) |
| **YCB-V** | [Hugging Face - YCB-V](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/ycbv/HccePose) |
| **T-LESS** | [Hugging Face - T-LESS](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/tless/HccePose) |
| **TUD-L** | [Hugging Face - TUD-L](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/tudl/HccePose) |
| **HB** | [Hugging Face - HB](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/hb/HccePose) |
| **ITODD** | [Hugging Face - ITODD](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/itodd/HccePose) |
| **IC-BIN** | [Hugging Face - IC-BIN](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/icbin/HccePose) |

---

#### 示例：LM-O 数据集

以 BOP 中最广泛使用的 **LM-O 数据集** 为例，我们采用了 **BOP2023 挑战** 中的 [默认 2D 检测器](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task1.zip)（GDRNPP），对 **HccePose(BF)** 进行了测试，并保存了以下结果文件：

- 2D 分割结果：[seg2d_lmo.json](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/lmo/seg2d_lmo.json)
- 6D 位姿结果：[det6d_lmo.csv](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/lmo/det6d_lmo.csv)

我们于 **2025 年 10 月 20 日** 提交了这两个文件。测试结果如下图所示。  
**6D 定位分数** 与 2024 年提交结果保持一致，  
**2D 分割分数** 提高了 **0.002**，这得益于我们修复了一些细微的程序 bug。
<details>
<summary>点击展开</summary>
### <img src="/show_vis/BOP-website-lmo.png" width=100%>
</details>

---

#### ⚙️ 说明

- 如果您发现某些权重文件的轮数为 **`0`**，这并不是错误。**HccePose(BF)** 的权重文件都是基于仅使用前表面训练的标准 HccePose 再训练得到的，在某些情况下，初始权重即能达到最佳性能。

---

## 📅 更新计划

我们目前正在整理和更新以下模块：

- 📁 ~~七个核心 BOP 数据集的 HccePose(BF) 权重文件~~

- 🧪 ~~BOP 挑战测试流程~~

- 🔁 基于前后帧跟踪的 6D 位姿推理

- 🏷️ 基于 HccePose(BF) 的真实场景 6D 位姿数据集制备

- ⚙️ PBR + Real 训练流程

- 📘 关于~~物体预处理~~、~~数据渲染~~、~~YOLOv11标签制备与训练~~以及HccePose(BF)的~~标签制备~~与~~训练~~的教程

预计所有模块将在 2025 年底前完成，并尽可能 每日持续更新。

---

## 🏆 BOP榜单
<img src="/show_vis/bop-6D-loc.png" width=100%>
<img src="/show_vis/bop-2D-seg.png" width=100%>


***
如果您觉得我们的工作有帮助，请按以下方式引用：
```bibtex
@InProceedings{HccePose_BF,
    author    = {Wang, Yulin and Hu, Mengting and Li, Hongli and Luo, Chen},
    title     = {HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {7166-7175}
}
```