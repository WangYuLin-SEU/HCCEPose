# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
Generate training labels for HccePose(BF).  
After label generation, the folder structure is as follows:
```
demo-bin-picking
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```

------------------------------------------------------    

生成 HccePose(BF)的训练标签。  
标签生成完成后，文件夹结构如下：
```
demo-bin-picking
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```
'''

import torch
from HccePose.bop_loader import bop_dataset, rendering_bop_dataset_back_front


if __name__ == '__main__':
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_path = '/root/xxxxxx/demo-bin-picking'
    
    # Create an instance for loading the BOP dataset.
    # 创建一个用于加载 BOP 数据集的实例。
    bop_dataset_item = bop_dataset(dataset_path)
    
    # Specify a folder within the dataset and load the data from it.
    # 指定数据集中的一个文件夹，并加载该文件夹中的数据。
    folder_name = 'train_pbr'
    rendering_bop_dataset_back_front_item = rendering_bop_dataset_back_front(bop_dataset_item, folder_name)
    
    # Iterate through all object IDs and their 3D model paths to generate label maps of front and back 3D coordinates for each object.
    # 遍历所有物体的 ID 及其 3D 模型路径，为每个物体生成正面和背面的 3D 坐标标签图。
    for (obj_id, obj_path) in zip(bop_dataset_item.obj_id_list, bop_dataset_item.obj_model_list):
        print(obj_path)
        
        # The `rendering_bop_dataset_back_front` function loads data for all objects and switches between them using the `update_obj_id` function.
        # `rendering_bop_dataset_back_front` 会加载所有物体的数据，并通过调用 `update_obj_id` 函数来切换不同的物体。

        rendering_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        
        # PyTorch multiprocessing is used to accelerate label map rendering.  
        # If the device has more CPU cores, increasing `num_workers` can further improve label generation speed.  
        # The `worker_init_fn` creates an independent VisPy renderer for each process, ensuring that rendering in different processes is performed independently without conflicts.
        # 使用 PyTorch 的多进程机制来加速标签图的渲染。  
        # 如果设备的 CPU 核心数量较多，可通过设置更高的 `num_workers` 来进一步提升标签生成速度。  
        # `worker_init_fn` 会为每个进程创建独立的 VisPy 渲染器，不同进程之间的渲染相互独立，不会产生冲突。
        batch_size = 16        
        data_gen_loader = torch.utils.data.DataLoader(rendering_bop_dataset_back_front_item, 
                                                batch_size=16, shuffle=False, num_workers=16, drop_last=False, 
                                                worker_init_fn=rendering_bop_dataset_back_front_item.worker_init_fn)
        for batch_idx, (cc_) in enumerate(data_gen_loader):
            if int(batch_idx%5) == 0:
                print(batch_idx)
            if batch_idx == int(rendering_bop_dataset_back_front_item.nSamples / batch_size) + 1:
                break
    pass