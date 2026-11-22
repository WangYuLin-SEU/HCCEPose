# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''

Train YOLOv11. After training, the folder structure is:
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

------------------------------------------------------    

训练 YOLOv11。训练完成后，文件夹结构如下：
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
'''

import os
import subprocess
import sys
if __name__ == '__main__':

    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_path = 'xxx/xxx/demo-bin-picking'
    
    # Specify the number of GPUs and the number of training epochs.  
    # For example, use 8 GPUs to train for 100 epochs.
    # 指定 GPU 的数量以及训练轮数。  
    # 例如使用 8 张 GPU 进行 100 轮训练。
    gpu_num = 8
    epochs = 100
    
    # Train
    # 开始训练
    dataset_name = os.path.basename(dataset_path)
    task_suffix = 'detection'
    dataset_pbr_path = os.path.join(dataset_path, 'train_pbr')
    train_multi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_train', 'train.py')
    data_objs_path = os.path.join(os.path.dirname(dataset_pbr_path), 'yolo11', 'train_obj_s', 'yolo_configs', 'data_objs.yaml')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), task_suffix, f"obj_s")
    model_name = f"yolo11-{task_suffix}-obj_s.pt"
    final_model_path = os.path.join(save_dir, model_name)
    batch_size = 12*gpu_num
    # A directory can survive a failed save; only a nonempty final weight completes this step.
    # 保存失败也可能留下目录；只有非空最终权重文件才表示本步骤完成。
    if not os.path.isfile(final_model_path) or os.path.getsize(final_model_path) == 0:
        subprocess.run([
            sys.executable, train_multi_path, '--data_path', data_objs_path,
            '--epochs', str(epochs), '--imgsz', '640', '--batch', str(batch_size),
            '--gpu_num', str(gpu_num), '--task', task_suffix,
        ], check=True)
        if not os.path.isfile(final_model_path) or os.path.getsize(final_model_path) == 0:
            raise FileNotFoundError(f'Training finished without the final detector weights: {final_model_path}')
    pass
