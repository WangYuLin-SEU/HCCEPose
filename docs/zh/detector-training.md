# 训练 2D 检测器

本页使用本仓库根目录中的准备、训练或评测脚本。请按[安装说明](installation.md#公开训练与渲染源码)准备对应依赖，再从仓库根目录执行命令。应用推理使用根目录的 `s4_p3_*.py` 示例。

HccePose(BF) 先用 YOLOv11 找到物体，再根据每个检测框内的图像估计 6D 位姿。要为自己的物体训练检测器，先将 [PBR 渲染数据](rendering.md)中的图像和 BOP 标注转换为 YOLO 格式。

## 准备图像和检测框标签

在 `s3_p1_prepare_yolo_label.py` 中设置数据根目录后即可转换。也可将下面脚本保存在仓库根目录，显式设置数据根、PBR 和输出路径；将 `dataset_root` 改为自己的数据位置。

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

推理使用 `yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt`。只有该最终权重为非空文件时，包装脚本才跳过训练；空目录或零字节权重不会被当成完成。子进程失败或训练结束后缺少最终权重会明确报错。启动训练时，`yolo_train/train.py` 检查 `yolo11/train_obj_s/train/weights/last.pt`，并在找到时请求 `resume=True`。

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

相对路径按本页要求从仓库根目录运行时解析。数据根、PBR、配置和权重路径分别构造，不重复拼接数据根。最终权重存在并不代替实际推理验证。
