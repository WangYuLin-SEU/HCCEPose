# 数据加载接口

BOP 加载器读取物体模型及其元数据。用 [`Tester`](tester-api.md) 处理自己的图片时，准备好模型信息和配套权重即可。需要读取 BOP 场景与标注时再调用 `load_folder()`。

## 模型元数据

```python
from HccePose.bop_loader import bop_dataset

bop_dataset(dataset_path, model_name='models', local_rank=0)
```

`dataset_path` 是数据集根目录，`model_name` 指定其中的模型子目录，`local_rank` 控制提示信息输出。加载器读取 `<dataset_path>/<model_name>/models_info.json`，并按六位编号构造 `obj_000001.ply` 形式的模型路径。

| 属性 | 含义 |
| --- | --- |
| `dataset_path`、`model_path` | 数据集根路径和选定的模型目录。 |
| `model_info` | 以字符串物体 ID 为键的模型元数据，包括尺寸、最小坐标、直径及可选对称信息。 |
| `obj_id_list` | 按元数据遍历顺序取得的整数物体 ID。 |
| `obj_model_list` | 与 `obj_id_list` 对齐的 PLY 路径。 |
| `obj_info_list` | 同一顺序的元数据条目。 |

YOLO 类别从 0 开始编号。`Tester` 将类别索引用作 `obj_id_list` 的下标来查找物体 ID。更换检测器或整理模型元数据时，务必保留训练时的类别顺序。

数据集根目录或模型目录不存在时，加载器会打印提示并提前返回。缺少 PLY 文件也会打印提示。创建 `Tester` 或训练数据集前，请检查提示中列出的路径。

Linux 下导入 `bop_loader` 会设置 `PYOPENGL_PLATFORM=egl`。即使只做推理，它仍会导入仓库内的 BOP 工具包和增强依赖。从项目根目录运行 `python -m zipfile -e bop_toolkit.zip .`，解压后应存在 `bop_toolkit/bop_toolkit_lib/`。

源码：[导入与 EGL 设置](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py)、[模型加载器](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py)、[检测类别映射](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/tester.py)。

## 场景元数据

```python
dataset.load_folder(folder_name, scene_num=200, vis=0.0)
```

- `folder_name` 是相对数据集的目录名，如 `train_pbr` 或 `test`。
- `scene_num` 按连续的六位场景编号扫描，从 `000000` 到编号 `scene_num - 1`。
- `vis` 保留满足 `visib_fract >= vis` 的标注实例。

按 BOP 目录结构准备 `scene_camera.json`、`scene_gt.json`、`scene_gt_info.json`，以及 `rgb/` 或 `gray/`、`depth/`、`mask/` 和 `mask_visib/`。加载器能处理部分标注缺失的情况，但仍会按深度和掩码文件的后缀构造路径。普通 RGB 图片请直接传给 `Tester`。

```python
{
    'img_info': {...},        # 键如 "000000_000001"，表示场景号_图像号
    'obj_info': {...},        # 键如 "obj_000001"，表示物体编号
    'scene_path_list': [...], # 已存在的数字编号场景目录
}
```

`img_info` 按图像组织路径、相机信息和标注实例，`obj_info` 按物体组织实例记录。此调用建立元数据索引，图像像素由后续数据集包装器读取。目标文件夹不存在时返回 `None`。

源码：[`load_folder`](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py)。

## 训练与评测包装器

```python
train_bop_dataset_back_front(
    bop_dataset_item, folder_name,
    padding_ratio=1.5, crop_size_img=256, aug_op='imgaug',
)

test_bop_dataset_back_front(
    bop_dataset_item, folder_name,
    bbox_2D=None, test_targets_bop19=None,
    bbox_2D_score_threshold=0.0,
    padding_ratio=1.5, crop_size_img=256, ratio=1.0,
)
```

将任一数据集交给 `DataLoader` 前，先调用 `update_obj_id(obj_id, obj_path)` 选择物体。两种数据集都按可见比例阈值 `0.2` 加载场景。初始化时可能创建 `<folder_name>_xyz_GT_front/` 和 `<folder_name>_xyz_GT_back/` 目录。

训练样本返回 `(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)`。评测样本追加 `(Bbox, cam_K, cam_R_m2c, cam_t_m2c)`；使用外部检测时再追加 `(scene_id, image_id, score)`。裁剪图为 BGR 归一化张量。现有训练脚本已处理批次、当前物体选择和 checkpoint 元数据。

缺失的正背面标签图会被零填充。训练前请检查生成的标签，并抽查几个样本。

外部 bbox JSON 条目使用 `scene_id`、`image_id`、`category_id`、`bbox` 和 `score`，分数筛选条件为严格的 `score > bbox_2D_score_threshold`。BOP 挑战脚本提供了现成接入示例。

源码：[训练包装器](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py)、[评测包装器](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py)、[样本返回值](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py)。

RGB-D 帧所需的三个文件及深度换算方法见[位姿精修接口](refinement-api.md)。各训练和评测脚本的用途见[工具与进阶接口](utilities-api.md)。
