# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
The script `s3_p1_prepare_yolo_label.py` is used to convert a BOP-format PBR training dataset into the YOLO format.  
After specifying the path `xxx/xxx/demo-bin-picking` and running `s3_p1_prepare_yolo_label.py`,  
a new folder named `yolo11` will be created under the `demo-bin-picking` directory.  
The following is an example of the resulting folder structure:

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

------------------------------------------------------    

脚本 `s3_p1_prepare_yolo_label.py` 的功能是将 BOP 格式的 PBR 训练数据集转换为 YOLO 格式的数据集。  
在指定 `xxx/xxx/demo-bin-picking` 路径后运行 `s3_p1_prepare_yolo_label.py`，  
程序会在 `demo-bin-picking` 文件夹下生成一个名为 `yolo11` 的文件夹。  
以下示例展示了生成后的文件夹结构：

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
'''

import os, json
from yolo_train.label import convert_train_pbr_2_yolo, generate_yaml

if __name__ == '__main__':
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_path = 'xxx/xxx/demo-bin-picking'
    
    # Keep one dataset root and derive PBR inputs and YOLO outputs separately.
    # 保留一个数据根目录，分别构造 PBR 输入与 YOLO 输出路径。
    dataset_root = dataset_path
    train_pbr_path = os.path.join(dataset_root, 'train_pbr')
    output_path = os.path.join(dataset_root, 'yolo11', 'train_obj_s')
    
    # Extract all object IDs from the models_info.json file.
    # 根据 models_info.json 文件提取所有物体 ID。
    obj_id_list = []
    with open(os.path.join(dataset_root, 'models', 'models_info.json'), "r") as f:
        scene_gt_data = json.load(f)
    for key_ in scene_gt_data:
        obj_id_list.append(key_)
        
    # Convert the BOP-format `train_pbr` dataset into a YOLO-format dataset.
    # 将 BOP 格式的 `train_pbr` 数据集转换为 YOLO 格式的数据集。
    convert_train_pbr_2_yolo(train_pbr_path, output_path, obj_id_list)
    generate_yaml(output_path, obj_id_list)
    print("[INFO] Dataset preparation complete!")
    pass
