# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''

Train HccePose (BF).  
After training is completed, an `HccePose` folder will be created in the dataset directory  
to store the weight files for each object.  

Example:
```
demo-tex-objs
|--- HccePose
    |--- obj_01
    ...
    |--- obj_10
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```

------------------------------------------------------    

训练 HccePose (BF)。  
训练完成后，会在数据集文件夹下生成一个 `HccePose` 文件夹，  
用于保存每个物体的权重文件。  

示例：
```
demo-tex-objs
|--- HccePose
    |--- obj_01
    ...
    |--- obj_10
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```
'''

import os, torch, argparse
import itertools
import numpy as np
from tqdm import tqdm
from HccePose.bop_loader import bop_dataset, train_bop_dataset_back_front, test_bop_dataset_back_front
from HccePose.network_model import HccePose_BF_Net, HccePose_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.metric import add_s
from kasal.bop_toolkit_lib.inout import load_ply

def test(obj_ply, obj_info, net: HccePose_BF_Net, test_loader: torch.utils.data.DataLoader):
    net.eval()
    add_list_l = []
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c) in tqdm(enumerate(test_loader)):
        if torch.cuda.is_available():
            rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            cam_K = cam_K.cpu().numpy()
        with autocast():
            pred_results = net.inference_batch(rgb_c, Bbox)
            pred_mask = pred_results['pred_mask']
            coord_image = pred_results['coord_2d_image']
            pred_front_code_0 = pred_results['pred_front_code_obj']
            pred_back_code_0 = pred_results['pred_back_code_obj']
            pred_front_code = pred_results['pred_front_code']
            pred_back_code = pred_results['pred_back_code']
            pred_front_code_raw = pred_results['pred_front_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
            pred_back_code_raw = pred_results['pred_back_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
            pred_front_code = torch.cat([pred_front_code, pred_front_code_raw], dim=-1)
            pred_back_code = torch.cat([pred_back_code, pred_back_code_raw], dim=-1)
            '''
            vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path='show_vis.jpg')
            '''
            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
            pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
            coord_image_np = coord_image.detach().cpu().numpy()
            pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
            for (cam_R_m2c_i, cam_t_m2c_i, pred_m_bf_c_np_i) in zip(cam_R_m2c.detach().cpu().numpy(), cam_t_m2c.detach().cpu().numpy(), pred_m_bf_c_np):
                info_list = solve_PnP_comb(pred_m_bf_c_np_i, train=True)
                
                for info_id_, info_i in enumerate(info_list):
                    info_list[info_id_]['add'] = add_s(obj_ply, obj_info, [[cam_R_m2c_i, cam_t_m2c_i]], [[info_i['rot'], info_i['tvecs']]])[0]
                add_list = []
                for i_ in range(len(info_list)):
                    info_list_i = itertools.combinations(info_list, len(info_list) - i_)
                    for info_list_i_j in info_list_i:
                        best_add = 0
                        best_s = 0
                        for info_list_i_j_k in info_list_i_j:
                            if info_list_i_j_k['num'] > best_s:
                                best_s = info_list_i_j_k['num']
                                best_add = info_list_i_j_k['add']
                        add_list.append(best_add)
                add_list = np.array(add_list)
                add_list_l.append(add_list)
        torch.cuda.empty_cache()
    add_list_l = np.array(add_list_l)
    add_list_l = np.mean(add_list_l, axis=0)
    print(add_list_l)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    print('max acc id: ', max_acc_id)
    print('max acc: ', max_acc)
    net.train()
    return max_acc_id, max_acc, add_list_l

if __name__ == '__main__':
    '''
    When `ide_debug` is set to True, single-GPU mode is used, allowing IDE debugging.  
    When `ide_debug` is set to False, DDP (Distributed Data Parallel) training is enabled.  

    DDP Training:  
    screen -S train_ddp
    nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    Single-GPU Training:  
    nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    ------------------------------------------------------    
    
    当 `ide_debug` 为 True 时，仅使用单卡，可在 IDE 中进行调试。  
    当 `ide_debug` 为 False 时，启用 DDP（分布式数据并行）训练。  

    DDP 训练：  
    screen -S train_ddp
    nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    单卡训练：
    nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    '''
    
    ide_debug = True
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_path = '/root/xxxxxx/demo-tex-objs'
    
    # Specify the name of the subfolder in the dataset used for loading training data.
    # 指定数据集中用于加载训练数据的子文件夹名称。
    train_folder_name = 'train_pbr'
    
    # The range of object IDs for training.  
    # `start_obj_id` is the starting object ID, and `end_obj_id` is the ending object ID.
    # 训练的物体 ID 范围。  
    # `start_obj_id` 为起始物体 ID，`end_obj_id` 为终止物体 ID。
    start_obj_id = 1
    end_obj_id =5
    
    # Total number of training epochs.
    # 总训练轮数。
    total_iteration = 50001
    
    # Learning rate.
    # 学习率。
    lr = 0.0002
    
    # Number of samples per training epoch.
    # 每轮训练的样本数量。
    batch_size = 24
    
    # Number of worker processes used by the DataLoader.
    # DataLoader 的进程数量。
    num_workers = 12
    
    # The number of epochs between saving checkpoints.
    # 保存检查点的间隔轮数。
    log_freq = 500
    
    # Scaling ratio for 2D bounding boxes.
    # 2D 包围盒的缩放比例。
    padding_ratio = 1.5
    

    # Whether to enable EfficientNet.
    # 是否启用 EfficientNet。
    efficientnet_key = None
    
    
    
    parser = argparse.ArgumentParser()
    if ide_debug:
        parser.add_argument("--local-rank", default=0, type=int)
    else:
        parser.add_argument("--local-rank", default=-1, type=int)
    args = parser.parse_args()
    if not ide_debug:
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier() 
        world_size = torch.distributed.get_world_size()
    local_rank = args.local_rank
    if local_rank != 0:
        if ide_debug is True:
            pass
    CUDA_DEVICE = str(local_rank)
    np.random.seed(local_rank)
    bop_dataset_item = bop_dataset(dataset_path, local_rank=local_rank)
    train_bop_dataset_back_front_item = train_bop_dataset_back_front(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, )
    
    # ratio = 0.01 means selecting 1% of samples from the dataset for testing.
    # ratio = 0.01 表示从数据集中选择 1% 的样本作为测试数据。
    test_bop_dataset_back_front_item = test_bop_dataset_back_front(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, ratio=0.01)
        
    for obj_id in range(start_obj_id, end_obj_id + 1):
        
        
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        print(obj_path)
        obj_ply = load_ply(obj_path)
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        
        # Create the save path.
        # 创建保存路径。
        save_path = os.path.join(dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'))
        best_save_path = os.path.join(save_path, 'best_score')
        try: os.mkdir(os.path.join(dataset_path, 'HccePose')) 
        except: 1
        try: os.mkdir(save_path) 
        except: 1
        try: os.mkdir(best_save_path) 
        except: 1

        # Get the 3D dimensions of the object.
        # 获取物体的 3D 尺寸。
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        # Define the loss function and neural network.
        # 定义损失函数和神经网络。
        loss_net = HccePose_Loss()
        scaler = GradScaler()
        net = HccePose_BF_Net(
                efficientnet_key = efficientnet_key,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        net_test = HccePose_BF_Net(
                efficientnet_key = efficientnet_key,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        if torch.cuda.is_available():
            net=net.to('cuda:'+CUDA_DEVICE)
            net_test=net_test.to('cuda:'+CUDA_DEVICE)
        optimizer=optim.Adam(net.parameters(), lr=lr)

        # Attempt to load weights from an interrupted training session.
        # 尝试加载中断训练时保存的权重。
        best_score = 0
        iteration_step = 0
        try:
            checkpoint_info = load_checkpoint(save_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
            best_score = checkpoint_info['best_score']
            iteration_step = checkpoint_info['iteration_step']
        except:
            print('no checkpoint')
        
        if not ide_debug:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], )
        
        # Update the training and testing data loaders respectively.
        # 分别更新训练和测试数据加载器。
        train_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        train_loader = torch.utils.data.DataLoader(train_bop_dataset_back_front_item, batch_size=batch_size, 
                                                shuffle=True, num_workers=num_workers, drop_last=True) 
        test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        test_loader = torch.utils.data.DataLoader(test_bop_dataset_back_front_item, batch_size=batch_size, 
                                                shuffle=False, num_workers=num_workers, drop_last=False) 
        
        # Train
        # 训练
        while True:
            end_training = False
            for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce) in enumerate(train_loader):
                
                # Test and save checkpoints only in the process where `local_rank = 0`.
                # 仅在 `local_rank = 0` 的进程中执行测试并保存检查点。
                if args.local_rank == 0:
                    if (iteration_step)%log_freq == 0 and iteration_step > 0:
                        if isinstance(net, torch.nn.parallel.DataParallel):
                            state_dict = net.module.state_dict()
                        elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
                            state_dict = net.module.state_dict()
                        else:
                            state_dict = net.state_dict()
                        net_test.load_state_dict(state_dict)
                        max_acc_id, max_acc, add_list_l = test(obj_ply, obj_info, net_test, test_loader, )
                        if max_acc >= best_score:
                            best_score = max_acc
                            save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, keypoints_ = add_list_l)
                        loss_net.print_error_ratio()
                        save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, keypoints_ = add_list_l)

                
                if torch.cuda.is_available():
                    rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                with autocast():
                    pred_mask, pred_front_back_code = net(rgb_c)
                    pred_front_code = pred_front_back_code[:, :24, ...]
                    pred_back_code = pred_front_back_code[:, 24:, ...]
                    current_loss = loss_net(pred_front_code, pred_back_code, pred_mask, GT_Front_hcce, GT_Back_hcce, mask_vis_c)
                    
                    # Visualization.
                    # 可视化。
                    '''
                    mask_vis_c = net.activation_function(mask_vis_c).round().clamp(0,1)
                    GT_Front_hcce = net.hcce_decode(GT_Front_hcce.permute(0,2,3,1)) / 255
                    GT_Back_hcce = net.hcce_decode(GT_Back_hcce.permute(0,2,3,1)) / 255
                    vis_rgb_mask_Coord(rgb_c, mask_vis_c, GT_Front_hcce.clamp(0,1), GT_Back_hcce.clamp(0,1), img_path='save_numpy_gt.jpg')
                    '''
                    
                    l_l = [
                        3*torch.sum(current_loss['Front_L1Losses']),
                        3*torch.sum(current_loss['Back_L1Losses']) ,
                        current_loss['mask_loss'],
                    ]
                    loss = l_l[0] + l_l[1] + l_l[2] #

                
                if not ide_debug:
                    torch.distributed.barrier()  
                    nan_flag = torch.tensor([int(torch.isnan(loss).any())], device=loss.device)
                    dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
                    if nan_flag.item() > 0:
                        for m in net.model.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_running_stats()
                        continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                
                if args.local_rank == 0:
                    print('dataset:%s - obj%s'%(os.path.basename(dataset_path), str(obj_id).rjust(2, '0')), 
                        "iteration_step:", iteration_step, 
                        "loss_front:", torch.sum(current_loss['Front_L1Losses']).item(),  
                        "loss_back:", torch.sum(current_loss['Back_L1Losses']).item(),  
                        "loss_mask:", current_loss['mask_loss'].item(),  
                        "total_loss:", loss.item(),
                        flush=True
                    )
                    
                iteration_step = iteration_step + 1
                if iteration_step >=total_iteration:
                    end_training = True
                    break
            if end_training == True:
                if args.local_rank == 0:
                    print('end the training in iteration_step:', iteration_step)
                break
             