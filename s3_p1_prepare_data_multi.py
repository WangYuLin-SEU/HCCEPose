

import os
import json
import shutil
from tqdm import tqdm
from PIL import Image 
from ultralytics.data.utils import autosplit

def prepare_train_pbr(train_pbr_path, output_path, obj_id_list, output_val_path):
    """
    Prepare the train_pbr dataset for YOLO, filtering by specific object ID.
    Creates an 'images' and 'labels' folder under output_path.
    """
    # Cameras to scan. Edit if needed.
    cameras = ["rgb"]
    # The corresponding ground-truth JSON files in each scene folder
    camera_gt_map = {
        "rgb": "scene_gt.json",
    }
    camera_gt_info_map = {
        "rgb": "scene_gt_info.json",
    }

    # Ensure the "images" and "labels" directories exist
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate over each scene (e.g. 000000, 000001, ...)
    scene_folders = [
        d for d in os.listdir(train_pbr_path)
        if os.path.isdir(os.path.join(train_pbr_path, d)) and not d.startswith(".")
    ]
    scene_folders.sort()  # optional: sort numerically

    for scene_folder in tqdm(scene_folders, desc="Processing train_pbr scenes"):
        scene_path = os.path.join(train_pbr_path, scene_folder)

        # For each camera, read bounding box info
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

            # Load the JSON files for ground truth + info
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            # Assume image IDs go from 0..N-1
            # num_imgs = len(scene_gt_data)  # or use max key from scene_gt_data
            for img_id in scene_gt_data:
                img_key = img_id # str(img_id)
                img_file_jpg = os.path.join(rgb_path, f"{int(img_id):06d}.jpg")
                img_file_png = os.path.join(rgb_path, f"{int(img_id):06d}.png")
                img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png if os.path.exists(img_file_png) else None

                if img_file is None:
                    continue

                if img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    # If there's no ground-truth info for this frame, skip
                    continue

                # Filter only bounding boxes for 'obj_id'
                # We also check if visibility fraction > 0 (you can adjust this threshold)
                valid_bboxes = []
                for bbox_info, gt_info in zip(scene_gt_info_data[img_key], scene_gt_data[img_key]):
                    if str(gt_info["obj_id"]) in obj_id_list and bbox_info["visib_fract"] > 0.1 and bbox_info['px_count_visib'] > 100:
                        valid_bboxes.append([str(gt_info["obj_id"]), bbox_info["bbox_visib"]])  # (x, y, w, h)

                if not valid_bboxes:
                    # No bounding boxes for our object in this image
                    continue

                # Copy the image to the YOLO "images/" folder
                out_img_name = f"{scene_folder}_{cam}_{int(img_id):06d}.jpg"
                out_img_path = os.path.join(images_dir, out_img_name)
                
                # img_read = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                # cv2.imwrite(out_img_path, img_read)
                
                shutil.copy(img_file, out_img_path)

                # Read real image dimensions
                with Image.open(img_file) as img:
                    img_width, img_height = img.size

                # Write YOLO label(s) for all bounding boxes in this image
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
                        
                        # YOLO format: class x_center y_center width height
                        # Here class is always '0' because we have only 1 object
                        id_ = str(obj_id_list.index(obj_id))
                        lf.write(f"{id_} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    if output_val_path is None:
        autosplit( 
                path=images_dir,
                weights=(1.00, 0.05, 0.0), # (train, validation, test) fractional splits
                annotated_only=False     # split only images with annotation file when True
            )

def generate_yaml(output_path, output_val_path, obj_id_list):
    """
    Generate a YOLO .yaml file for training/validation.
    Writes file as: bpc/yolo/configs/data_obj_{obj_id}.yaml
    """
    yolo_configs_dir = os.path.join(output_path, "yolo_configs")
    os.makedirs(yolo_configs_dir, exist_ok=True)

    # The 'images' directory under output_path
    images_dir = os.path.join(output_path, "images")
    train_path = images_dir
    val_path = None
    if output_val_path is not None:
        images_dir = os.path.join(output_val_path, "images")
        val_path = images_dir 
    yaml_path = os.path.join(yolo_configs_dir, f"data_objs.yaml")

    names = []
    for obj_id_ in obj_id_list:
        names.append(f"{obj_id_}")
    # YAML content (same as before)
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


def main():

    # dataset_name = 'demo-charuco-board'
    dataset_name = 'demo-bin-picking'
    
    dataset_path = '/root/xxxxxx/%s/train_pbr'%dataset_name 
    output_path  = '/root/xxxxxx/%s/yolo11/train_obj_s'%dataset_name 
    
    output_val_path = None
    
    obj_id_list = []
    
    with open('/root/xxxxxx/%s/models/models_info.json'%dataset_name, "r") as f:
        scene_gt_data = json.load(f)
    for key_ in scene_gt_data:
        obj_id_list.append(key_)

    # 1) Prepare YOLO images + labels
    prepare_train_pbr(dataset_path, output_path, obj_id_list, output_val_path)
    # 2) Generate .yaml file for YOLO
    generate_yaml(output_path, output_val_path, obj_id_list)

    print("[INFO] Dataset preparation complete!")


if __name__ == "__main__":
    main()