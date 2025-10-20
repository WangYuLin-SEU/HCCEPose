import os, sys, torch, time
import numpy as np
from HccePose.bop_loader import bop_dataset, test_bop_dataset_back_front
from HccePose.network_model import HccePose_BF_Net, load_checkpoint
from torch.cuda.amp import autocast as autocast
from kasal.bop_toolkit_lib.inout import load_ply
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.metric import add_s
if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    
    train_folder_name = 'train_pbr'
    obj_id = 1
    
    CUDA_DEVICE = '0'
    
    # vis_op = False
    vis_op = True
    
    pnp_op = 'ransac+comb' # ['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb']
    pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
    
    batch_size = 24
    num_workers = 12
    
    
    log_freq = 500
    padding_ratio = 1.5
    efficientnet_key = None
    
    bop_dataset_item = bop_dataset(dataset_path)
    obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
    print(obj_path)
    
    save_path = os.path.join(dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'))
    best_save_path = os.path.join(save_path, 'best_score')
    
    test_bop_dataset_back_front_item = test_bop_dataset_back_front(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, ratio=0.01)
    
    obj_ply = load_ply(obj_path)
    obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
    
    min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
    size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
    
    net = HccePose_BF_Net(
            efficientnet_key = efficientnet_key,
            input_channels = 3, 
            min_xyz = min_xyz,
            size_xyz = size_xyz,
        )
    checkpoint_info = load_checkpoint(best_save_path, net, CUDA_DEVICE=CUDA_DEVICE)
    best_score, iteration_step, keypoints_ = \
        checkpoint_info['best_score'], checkpoint_info['iteration_step'], checkpoint_info['keypoints_']
    if torch.cuda.is_available():
        net=net.to('cuda:'+CUDA_DEVICE)
        net.eval()
        
    test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
    test_loader = torch.utils.data.DataLoader(test_bop_dataset_back_front_item, batch_size=batch_size, 
                                               shuffle=False, num_workers=num_workers, drop_last=False) 
    
    gt_list = []
    pred_list = []
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c) in enumerate(test_loader):
        if torch.cuda.is_available():
            rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            cam_K = cam_K.cpu().numpy()
        with autocast():
            t1 = time.time()
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
            
            if vis_op is not None:
                vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path='show_vis.jpg')
            
            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
            pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
            results = []
            coord_image_np = coord_image.detach().cpu().numpy()
            
            if pnp_op in ['epnp', 'ransac', 'ransac+vvs']:
                pred_m_f_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
                for pred_m_f_c_np_i in pred_m_f_c_np:
                    result_i = solve_PnP(pred_m_f_c_np_i, pnp_op=pnp_op_l[1][pnp_op_l[0].index(pnp_op)])
                    results.append(result_i)
                    pred_list.append([result_i['rot'], result_i['tvecs']])
            else:
                pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
                for pred_m_bf_c_np_i in pred_m_bf_c_np:
                    if pnp_op == 'ransac+comb':
                        pnp_op_0 = 2
                    else:
                        pnp_op_0 = 1
                    result_i = solve_PnP_comb(pred_m_bf_c_np_i, keypoints_, pnp_op=pnp_op_0)
                    results.append(result_i)
                    pred_list.append([result_i['rot'], result_i['tvecs']])
            
            
            for (cam_R_m2c_i, cam_t_m2c_i) in zip(cam_R_m2c.detach().cpu().numpy(), cam_t_m2c.detach().cpu().numpy()):
                gt_list.append([cam_R_m2c_i, cam_t_m2c_i])
                
            t2 = time.time()
            print(t2 - t1)
            results
        torch.cuda.empty_cache()

    acc_add, add_passes_list, add_error_list = add_s(obj_ply, obj_info, gt_list, pred_list)
    print('acc add: ', acc_add)
    pass
            