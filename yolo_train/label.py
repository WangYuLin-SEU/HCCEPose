# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
label.py is used to prepare YOLO training labels.  
The original script was adapted from OpenCV BPC.  
Project link: https://github.com/opencv/bpc  
The original version trained a separate YOLO model for each object,  
while this modified version supports training a single YOLO model for multiple objects.

------------------------------------------------------    

脚本 `label.py` 用于生成 YOLO 的训练标签。  
原始脚本改编自 OpenCV BPC 项目。  
项目链接：https://github.com/opencv/bpc  
原版脚本为每个物体单独训练一个 YOLO 模型，  
本脚本将其修改为支持多个物体共用一个 YOLO 模型进行训练。
'''

import os
import json
import shutil
from tqdm import tqdm
from PIL import Image 
from ultralytics.data.utils import autosplit

def convert_train_pbr_2_yolo(train_pbr_path, output_path, obj_id_list):
    '''
    ---
    ---
    Convert BOP-format dataset to YOLO-format dataset.
    ---
    ---
    Converts a BOP-format dataset into the YOLO format, creating `images` and `labels` folders
    to store 2D images and corresponding 2D bounding box (BBox) annotations.
    The function automatically extracts images containing the specified object IDs (`obj_id_list`)
    from the PBR training dataset and generates corresponding 2D BBox labels.
    ---
    ---
    Args:
        - train_pbr_path: Path to the input PBR training dataset.
        - output_path: Directory for saving the generated `images` and `labels` folders.
        - obj_id_list: List of object IDs to be converted.

    Returns:
        - None
    ---
    ---
    将 BOP 格式的数据集转换为 YOLO 格式的数据集。
    ---
    ---
    将 BOP 格式的数据集转换为 YOLO 格式，生成 `images` 和 `labels` 文件夹，
    分别用于存储 2D 图像及其对应的 2D 边界框（BBox）标签。
    该函数会根据给定的物体 ID 列表（`obj_id_list`），
    从 PBR 训练数据集中自动提取包含这些物体的图像，并生成相应的 2D BBox 标签。
    ---
    ---
    参数:
        - train_pbr_path: 输入的 PBR 训练数据集路径。
        - output_path: 用于保存生成的 `images` 和 `labels` 文件夹的目录。
        - obj_id_list: 需要转换的物体 ID 列表。

    返回:
        - 无
    '''

    cameras = ["rgb"]
    camera_gt_map = {
        "rgb": "scene_gt.json",
    }
    camera_gt_info_map = {
        "rgb": "scene_gt_info.json",
    }
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    scene_folders = [
        d for d in os.listdir(train_pbr_path)
        if os.path.isdir(os.path.join(train_pbr_path, d)) and not d.startswith(".")
    ]
    scene_folders.sort() 
    for scene_folder in tqdm(scene_folders, desc="Processing train_pbr scenes"):
        scene_path = os.path.join(train_pbr_path, scene_folder)
        for cam in cameras:
            rgb_path = os.path.join(scene_path, cam)
            scene_gt_file = os.path.join(scene_path, camera_gt_map[cam])
            scene_gt_info_file = os.path.join(scene_path, camera_gt_info_map[cam])
            if not os.path.exists(rgb_path):
                print(f"Missing RGB folder for {cam} in {scene_folder}: {rgb_path}")
                continue
            if not os.path.exists(scene_gt_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_file}")
                continue
            if not os.path.exists(scene_gt_info_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_info_file}")
                continue
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)
            for img_id in scene_gt_data:
                img_key = img_id 
                img_file_jpg = os.path.join(rgb_path, f"{int(img_id):06d}.jpg")
                img_file_png = os.path.join(rgb_path, f"{int(img_id):06d}.png")
                img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png if os.path.exists(img_file_png) else None
                if img_file is None:
                    continue
                if img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    continue
                valid_bboxes = []
                for bbox_info, gt_info in zip(scene_gt_info_data[img_key], scene_gt_data[img_key]):
                    if str(gt_info["obj_id"]) in obj_id_list and bbox_info["visib_fract"] > 0.1 and bbox_info['px_count_visib'] > 100:
                        valid_bboxes.append([str(gt_info["obj_id"]), bbox_info["bbox_visib"]]) 
                if not valid_bboxes:
                    continue
                out_img_name = f"{scene_folder}_{cam}_{int(img_id):06d}.jpg"
                out_img_path = os.path.join(images_dir, out_img_name)
                shutil.copy(img_file, out_img_path)
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
                out_label_name = f"{scene_folder}_{cam}_{int(img_id):06d}.txt"
                out_label_path = os.path.join(labels_dir, out_label_name)
                with open(out_label_path, "w") as lf:
                    for obj_id, (x, y, w, h) in valid_bboxes:
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height
                        assert 0 <= x_center <= 1 and 0 <= y_center <= 1
                        assert 0 <= width <= 1 and 0 <= height <= 1
                        id_ = str(obj_id_list.index(obj_id))
                        lf.write(f"{id_} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    autosplit( 
            path=images_dir,
            weights=(1.00, 0.05, 0.0),
            annotated_only=False 
        )

def generate_yaml(output_path, obj_id_list):
    
    '''
    ---
    ---
    Generate the YAML configuration file required for YOLO training.
    ---
    ---
    This function creates `yolo_configs/data_objs.yaml` in the specified directory
    based on the provided list of object IDs.
    ---
    ---
    Args:
        - output_path: Directory where the generated `yolo_configs/data_objs.yaml` will be saved.
        - obj_id_list: List of object IDs to include in the YAML configuration.

    Returns:
        - None
    ---
    ---
    生成 YOLO 训练所需的 YAML 配置文件。
    ---
    ---
    该函数根据给定的物体 ID 列表，
    在指定目录下创建 `yolo_configs/data_objs.yaml` 文件。
    ---
    ---
    参数:
        - output_path: 用于保存生成的 `yolo_configs/data_objs.yaml` 的目录。
        - obj_id_list: 需要写入 YAML 配置文件的物体 ID 列表。

    返回:
        - 无
    '''

    yolo_configs_dir = os.path.join(output_path, "yolo_configs")
    os.makedirs(yolo_configs_dir, exist_ok=True)
    images_dir = os.path.join(output_path, "images")
    train_path = images_dir
    val_path = None
    yaml_path = os.path.join(yolo_configs_dir, f"data_objs.yaml")
    names = []
    for obj_id_ in obj_id_list:
        names.append(f"{obj_id_}")
    if val_path is not None:
        yaml_content = {
            "train": train_path,
            "val": val_path,
            "nc": len(obj_id_list),
            "names": names 
        }
    else:
        yaml_content = {
            "train": os.path.join(os.path.dirname(images_dir), 'autosplit_train.txt'),
            "val": os.path.join(os.path.dirname(images_dir), 'autosplit_val.txt'),
            "nc": len(obj_id_list),
            "names": names 
        }
    with open(yaml_path, "w") as f:
        for key, value in yaml_content.items():
            f.write(f"{key}: {value}\n")
    print(f"[INFO] Generated YAML file at: {yaml_path}\n")
    return yaml_path