# 训练 2D 检测器

本页使用[公开上游源码](https://github.com/WangYuLin-SEU/HCCEPose/tree/f75cc4d5db2fc298fc7b853f8c75273757872382)中的准备、训练或评测脚本。请先按[安装说明](installation.md#公开训练与渲染源码)获取该版本，并在上游源码根目录执行本页命令；当前项目中的 `examples/inference/` 用于应用推理。

HccePose(BF) 先用 YOLOv11 找到物体，再根据每个检测框内的图像估计 6D 位姿。要为自己的物体训练检测器，先将 [PBR 渲染数据](rendering.md)中的图像和 BOP 标注转换为 YOLO 格式。

## 准备图像和检测框标签

公开 `s3_p1_prepare_yolo_label.py` 展示转换过程。该固定版本的入口路径拼接会重复加入数据集根目录，直接只替换占位字符串可能找不到文件。可将下面脚本保存在上游源码根目录，显式保留数据根、PBR 和输出三个不同路径；将 `dataset_root` 改为自己的数据位置。

```python
import json
import os
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from yolo_train.label import convert_train_pbr_2_yolo, generate_yaml

if __name__ == '__main__':
    dataset_root = os.path.join(project_dir, 'data', 'demo-bin-picking')
    pbr_path = os.path.join(dataset_root, 'train_pbr')
    output_path = os.path.join(dataset_root, 'yolo11', 'train_obj_s')
    info_path = os.path.join(dataset_root, 'models', 'models_info.json')
    with open(info_path, encoding='utf-8') as handle:
        object_ids = list(json.load(handle))
    convert_train_pbr_2_yolo(pbr_path, output_path, object_ids)
    generate_yaml(output_path, object_ids)
```

转换结果位于 `yolo11/train_obj_s/` 下。`images/` 和 `labels/` 分别保存图像和检测框标注。`data_objs.yaml` 是训练配置，两个 `autosplit_*.txt` 文件分别列出训练和验证样本。

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

## 训练和读取权重

`s3_p2_train_yolo.py` 默认使用 8 张 GPU，训练 100 个 epoch。根据机器配置设置 `gpu_num` 后运行：

```bash
python s3_p2_train_yolo.py
```

推理使用 `yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt`。训练脚本看到 `detection/obj_s/` 已存在就会跳过训练，即使里面还没有最终权重。若上一次训练中断，重跑前先检查该目录。只有外层脚本实际启动训练时，`yolo_train/train.py` 才会检查 `yolo11/train_obj_s/train/weights/last.pt`，并在找到时请求 `resume=True`。

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

检测器就绪后，接下来为位姿网络[生成正背面坐标标签](coordinate-labels.md)。

公开训练入口的输出路径也应检查：`save_dir` 已由 `data_objs_path` 定位，最终权重应使用 `os.path.join(save_dir, model_name)`；相对数据路径不要重复拼接根目录。训练后确认最终权重文件存在，再交给推理加载器。
