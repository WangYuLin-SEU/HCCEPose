# Train the 2D detector

This page uses preparation, training or evaluation scripts in this repository’s root. Prepare the dependencies described in [Installation](installation.md#public-training-and-rendering-source), then run the commands from the repository root. Application inference examples are in `examples/inference/`.

HccePose(BF) uses YOLOv11 to locate objects and estimates a 6D pose from each detected crop. To train a detector for your objects, convert the images and BOP annotations from your [PBR scenes](rendering.md) to YOLO format.

## Prepare images and bounding-box labels

Set the dataset root in `s3_p1_prepare_yolo_label.py` to run conversion. Alternatively, save this recipe in the repository root, keeping dataset, PBR and output paths separate; set `dataset_root` to your data.

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

The converter writes the dataset to `yolo11/train_obj_s/`. The `images/` and `labels/` directories hold the images and bounding boxes. The `data_objs.yaml` file contains the training configuration, and the two `autosplit_*.txt` files list the training and validation images.

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

## Train the detector and locate the weights

The example trains for 100 epochs on eight GPUs. Set `gpu_num` in `s3_p2_train_yolo.py` to match your machine, then run:

```bash
python s3_p2_train_yolo.py
```

Inference loads `yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt`. The wrapper skips training only when this final file is nonempty; an empty directory or zero-byte weight does not count as completion. A failed subprocess or missing final output raises an error. When training starts, `yolo_train/train.py` checks for `yolo11/train_obj_s/train/weights/last.pt` and requests `resume=True` when it finds that checkpoint.

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

Next, [generate the front and back coordinate labels](coordinate-labels.md) for the pose network.

Relative paths resolve from the repository root as required above. Dataset, PBR, configuration and weight paths are constructed separately without repeating the root. A final weight file still needs an actual inference check.
