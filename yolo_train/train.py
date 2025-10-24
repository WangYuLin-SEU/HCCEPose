# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
The script `train.py` is used to train YOLOv11.  
The original script was adapted from the OpenCV BPC project.  
Project link: https://github.com/opencv/bpc  
We added several augmentation strategies to enhance YOLO's performance.

------------------------------------------------------    

脚本 `train.py` 用于训练 YOLOv11。  
原始脚本改编自 OpenCV BPC 项目。  
项目链接：https://github.com/opencv/bpc  
在此基础上，增加了多种数据增强策略，以提升 YOLO 的性能。
'''

import os
import sys
import argparse
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print(current_dir)

def train_yolo11(task, data_path, gpu_num, epochs, imgsz, batch):
    """
    Train YOLO11 for a specific task ("detection" or "segmentation")
    using Ultralytics YOLO with a single object class.

    Args:
        task (str): "detection" or "segmentation"
        data_path (str): Path to the YOLO .yaml file (e.g. data_obj_11.yaml).
        obj_id (int): The BOP object ID (e.g. 11).
        epochs (int): Number of training epochs.
        imgsz (int): Image size used for training.
        batch (int): Batch size.

    Returns:
        final_model_path (str): Path where the trained model is saved.
    """

    device = []
    for i_ in range(int(gpu_num)):
        device.append(i_)
    
    if task == "detection":
        pretrained_weights = "yolo11x.pt"
        task_suffix = "detection"
    elif task == "segmentation":
        pretrained_weights = "yolo11n-seg.pt"
        task_suffix = "segmentation"
    else:
        print("Invalid task. Must be 'detection' or 'segmentation'.")
        return None
    if not os.path.exists(data_path):
        print(f"Error: Dataset YAML file not found at {data_path}")
        return None
    print(f"Loading model {pretrained_weights} for {task_suffix} ...")
    
    last_pt = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'train', 'weights', 'last.pt')
    if os.path.exists(last_pt):
        model = YOLO(last_pt)
        resume = True
    else:
        model = YOLO(pretrained_weights)
        resume = False
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        val = True, 
        fraction = 1.00,
        workers=8,  
        save=True,  
        save_period = 1,
        project = os.path.dirname(os.path.dirname(data_path)),
        close_mosaic = 10,
        label_smoothing=0.0,  
        degrees=0.0,       
        translate=0.1,      
        scale=0.50,     
        shear=0.0,       
        perspective=0.0,    
        flipud=0.5,      
        fliplr=0.5,       
        mosaic=1.0,     
        mixup=1.0,         
        copy_paste = 1.0,
        copy_paste_mode = 'mixup',
        resume=resume,
        dropout = 0.2,
        auto_augment = 'AugMix',
        freeze=0, 
        multi_scale=True,
    )
    save_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), task_suffix, f"obj_s")
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"yolo11-{task_suffix}-obj_s.pt"
    final_model_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), save_dir, model_name)
    model.save(final_model_path)
    print(f"Model saved as: {final_model_path}")
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on a specific dataset and object.")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset YAML file (e.g. idp_codebase/yolo/configs/data_obj_11.yaml).")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation"], default="detection",
                        help="Task type (detection or segmentation).")

    args = parser.parse_args()

    train_yolo11(
        task=args.task,
        data_path=args.data_path,
        gpu_num = args.gpu_num,
        # obj_id=args.obj_id,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )


if __name__ == "__main__":
    main()
