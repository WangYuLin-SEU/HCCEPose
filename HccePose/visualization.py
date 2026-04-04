"""Visualization helpers for HccePose.

This module collects the image-level rendering and overlay utilities used by
HccePose test scripts and the tester pipeline. It de-normalizes BGR-order crops
(``IMAGENET_MEAN_BGR``) into uint8 **BGR** panels for OpenCV, builds mask/code
tiles, and draws pose overlays with BGR line colors.

HccePose 的可视化辅助函数。

该模块汇总了 HccePose 测试脚本与 Tester 推理流程中的图像级可视化：
将按 BGR + ``IMAGENET_MEAN_BGR`` 归一化的网络输出还原为 uint8 **BGR** 拼图（OpenCV），
并叠加 Mask、编码图与位姿投影（线条颜色按 BGR 解释）。
"""

import torch, cv2, kornia
import numpy as np

from HccePose.bop_loader import IMAGENET_MEAN_BGR, IMAGENET_STD_BGR


def vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path = None):
    '''
    ---
    ---
    Visualize cropped image (BGR tensor convention), mask, and front/back code maps.
    ---
    ---
    The function de-normalizes BGR crops to uint8 BGR (OpenCV layout), arranges the
    predicted mask together with front/back coordinate codes into one canvas,
    and optionally saves the panel to disk. It is mainly used for debugging the
    direct HccePose network outputs before PnP.

    Args:
        - rgb_c: Normalized cropped BGR tensor (OpenCV channel order, ImageNet stats permuted).
        - pred_mask: Predicted binary mask tensor.
        - pred_front_code: Predicted front-side code tensor.
        - pred_back_code: Predicted back-side code tensor.
        - img_path: Optional path for saving the visualization.

    Returns:
        - save_numpy: The composed visualization image in numpy format.
    ---
    ---
    可视化裁剪图（BGR 张量约定）、掩膜以及正背面编码图。
    ---
    ---
    该函数会将归一化后的 BGR 裁剪图恢复到 uint8 BGR（OpenCV 布局），并把预测得到的掩膜、
    正面编码和背面编码拼接成一张总览图；必要时还可以直接保存到磁盘。
    它主要用于在进入 PnP 之前调试 HccePose 网络的直接输出。

    参数:
        - rgb_c: 归一化后的裁剪 BGR tensor（OpenCV 通道顺序，ImageNet 统计量已按 BGR 重排）。
        - pred_mask: 预测得到的二值掩膜 tensor。
        - pred_front_code: 预测得到的正面编码 tensor。
        - pred_back_code: 预测得到的背面编码 tensor。
        - img_path: 可选的可视化保存路径。

    返回:
        - save_numpy: 拼接后的 numpy 可视化图像。
    '''
    text_list = ['RGB', 'Mask']
    mean = torch.tensor(IMAGENET_MEAN_BGR, device=rgb_c.device, dtype=torch.float32)
    std = torch.tensor(IMAGENET_STD_BGR, device=rgb_c.device, dtype=torch.float32)
    def reverse_normalize(tensor):
        if tensor.dim() == 4:
            mean_ = mean.view(1, 3, 1, 1) 
            std_ = std.view(1, 3, 1, 1)
        else: 
            mean_ = mean.view(3, 1, 1)
            std_ = std.view(3, 1, 1)
        return tensor * std_ + mean_  
    reversed_rgb_c = reverse_normalize(rgb_c)
    reversed_rgb_c = reversed_rgb_c * 255
    reversed_rgb_c = kornia.geometry.transform.resize(
        reversed_rgb_c, 
        (128, 128),
        interpolation='bilinear',
    )
    reversed_rgb_c = reversed_rgb_c.permute(0,2,3,1)
    pred_mask_s = pred_mask[...,None].repeat(1,1,1,3) * 255
    pred_front_code = pred_front_code.clone() * 255
    pred_back_code = pred_back_code.clone() * 255
    s0,s1,s2 = reversed_rgb_c.shape[0]*reversed_rgb_c.shape[1], reversed_rgb_c.shape[2], reversed_rgb_c.shape[3]
    reversed_rgb_c = reversed_rgb_c.reshape((s0,s1,s2))
    pred_mask_s = pred_mask_s.reshape((s0,s1,s2))
    pred_front_code_l, pred_back_code_l = [], []
    text_front_list = []
    text_back_list = []
    for i in range(int(pred_front_code.shape[-1]/3)): 
        pred_front_code_l.append(pred_front_code[..., i*3:(i+1)*3].reshape((s0,s1,s2)))
        pred_back_code_l.append(pred_back_code[..., i*3:(i+1)*3].reshape((s0,s1,s2)))
        if i == 0:
            text_front_list.append('3D(F)')
            text_back_list.append('3D(B)')
        else:
            text_front_list.append('C%s(F)'%str(i))
            text_back_list.append('C%s(B)'%str(i))
    text_list = text_list + text_front_list + text_back_list
    save_tensor = torch.cat([reversed_rgb_c, pred_mask_s] + pred_front_code_l + pred_back_code_l, dim = 1)
    save_numpy = save_tensor.detach().cpu().numpy()
    for i in range(len(text_list)):
        cv2.putText(save_numpy, text_list[i], (i * 128 + 10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0) , 2, cv2.LINE_AA)
    if img_path is not None:
        cv2.imwrite(img_path, save_numpy)
    return save_numpy

def zero_other_masks_by_conf(pred_mask, conf_s):
    '''
    ---
    ---
    Keep only the highest-confidence mask at each pixel location.
    ---
    ---
    When multiple instance masks overlap, this helper suppresses all masks
    except the one whose detection confidence is maximal at the current pixel.
    This makes the merged origin-space visualization cleaner and easier to read.

    Args:
        - pred_mask: Batched mask tensor.
        - conf_s: Detection confidence tensor for each instance.

    Returns:
        - processed_masks: Masks after confidence-based per-pixel suppression.
    ---
    ---
    在每个像素位置仅保留置信度最高的掩膜。
    ---
    ---
    当多个实例掩膜发生重叠时，该函数会抑制除当前像素处置信度最高实例以外的
    其他掩膜，从而让回到原图空间后的可视化更加整洁、易于阅读。

    参数:
        - pred_mask: 批量实例掩膜 tensor。
        - conf_s: 每个实例对应的检测置信度 tensor。

    返回:
        - processed_masks: 经过逐像素置信度抑制后的掩膜。
    '''
    pred_mask = pred_mask.permute(1,0,2,3)
    B, N, H, W = pred_mask.shape 
    conf_s = conf_s[None,:,None,None].repeat(1, 1, H, W) * pred_mask
    max_indices = torch.argmax(conf_s, dim=1) 
    mask_indices = torch.arange(N, device=pred_mask.device).view(1, N, 1, 1) 
    is_max_mask = (max_indices.unsqueeze(1) == mask_indices) 
    max_mask = is_max_mask.float() 
    processed_masks = pred_mask * max_mask
    return processed_masks.permute(1,0,2,3)

def vis_rgb_mask_Coord_origin(cam_K, obj_ids_l, obj_ids_all, BBox_3d_l, Rts_l, conf_s, rgb_c, pred_mask, pred_front_code, pred_back_code, img_path = None):
    '''
    ---
    ---
    Visualize masks, codes, and projected 6D poses in the original image space.
    ---
    ---
    This function fuses multiple object instances back to the full-resolution
    image, overlays mask/code responses, and draws projected 3D bounding boxes
    according to the recovered poses. It is mainly used as the final HccePose
    2D/6D qualitative visualization.

    Args:
        - cam_K: Camera intrinsic matrix.
        - obj_ids_l: Object ids for the predicted instances.
        - obj_ids_all: All dataset object ids.
        - BBox_3d_l: Precomputed 3D bounding-box edge list for every object.
        - Rts_l: Predicted 6D poses.
        - conf_s: Detection confidences.
        - rgb_c: Full image tensor in normalized BGR space (ImageNet stats permuted).
        - pred_mask: Warped masks in image space.
        - pred_front_code: Warped front code maps.
        - pred_back_code: Warped back code maps.
        - img_path: Reserved optional save path.

    Returns:
        - save_numpy: Combined code/pose visualization canvas.
        - save_numpy_2: Compact side-by-side BGR uint8 view with pose overlay (OpenCV layout).
    ---
    ---
    在原图空间中可视化掩膜、编码图与 6D 位姿投影。
    ---
    ---
    该函数会将多个实例重新映射回完整分辨率图像，并叠加显示掩膜与编码响应，
    同时依据恢复出的位姿绘制物体的 3D 包围盒投影。它主要用于生成 HccePose
    最终的 2D/6D 定性展示结果。

    参数:
        - cam_K: 相机内参矩阵。
        - obj_ids_l: 当前预测实例对应的物体 id 列表。
        - obj_ids_all: 数据集中的全部物体 id 列表。
        - BBox_3d_l: 每个物体预先计算好的 3D 包围盒边集合。
        - Rts_l: 预测得到的 6D 位姿。
        - conf_s: 检测置信度。
        - rgb_c: 归一化后的整图 BGR tensor（ImageNet 统计量已按 BGR 重排）。
        - pred_mask: 映射回原图空间的掩膜。
        - pred_front_code: 映射回原图空间的正面编码图。
        - pred_back_code: 映射回原图空间的背面编码图。
        - img_path: 预留的可选保存路径。

    返回:
        - save_numpy: 组合后的编码与位姿可视化大图。
        - save_numpy_2: 紧凑版 BGR uint8 与位姿叠加（OpenCV 布局）。
    '''
    
    mean = torch.tensor(IMAGENET_MEAN_BGR, device=rgb_c.device, dtype=torch.float32)
    std = torch.tensor(IMAGENET_STD_BGR, device=rgb_c.device, dtype=torch.float32)
    def reverse_normalize(tensor):
        if tensor.dim() == 4:
            mean_ = mean.view(1, 3, 1, 1) 
            std_ = std.view(1, 3, 1, 1)
        else: 
            mean_ = mean.view(3, 1, 1)
            std_ = std.view(3, 1, 1)
        return tensor * std_ + mean_  
    reversed_rgb_c = reverse_normalize(rgb_c)
    reversed_rgb_c = reversed_rgb_c * 255

    reversed_rgb_c = reversed_rgb_c.permute(0,2,3,1)
    
    pred_mask = zero_other_masks_by_conf(pred_mask, conf_s)
    
    
    pred_front_code = (pred_front_code * 255 * pred_mask.repeat(1, pred_front_code.shape[1],1,1)).permute(0,2,3,1)
    pred_back_code = (pred_back_code * 255 * pred_mask.repeat(1, pred_back_code.shape[1],1,1)).permute(0,2,3,1)
    
    pred_mask_s_copy = pred_mask.repeat(1,3,1,1).clone().permute(0,2,3,1)
    
    rand_RGB = torch.rand(size=(pred_mask.shape[0], 3),device=pred_mask.device)
    
    pred_mask_s = pred_mask.repeat(1,3,1,1) * rand_RGB[..., None,None] * 255
    pred_mask_s = pred_mask_s.permute(0,2,3,1)
    reversed_rgb_c_mask = (reversed_rgb_c.clone()[0] + 0.5 * pred_mask_s.sum(dim=0)) / (pred_mask_s_copy.sum(dim=0)*0.5 + 1)
    
    s0,s1,s2 = reversed_rgb_c.shape[0]*reversed_rgb_c.shape[1], reversed_rgb_c.shape[2], reversed_rgb_c.shape[3]

    line_1 = [reversed_rgb_c_mask]
    line_1_text = ['Mask & 6D Poses']
    pred_front_code_l, pred_back_code_l = [], []
    text_front_list = []
    text_back_list = []
    for i in range(int(pred_front_code.shape[-1]/3)): 
        if i == 0:
            line_1.append((reversed_rgb_c.clone()[0] + 10*pred_front_code[..., i*3:(i+1)*3].sum(dim=0).reshape((s0,s1,s2))) / (pred_mask_s_copy.sum(dim=0)*10.0 + 1))
            line_1.append((reversed_rgb_c.clone()[0] + 10*pred_back_code[..., i*3:(i+1)*3].sum(dim=0).reshape((s0,s1,s2))) / (pred_mask_s_copy.sum(dim=0)*10.0 + 1))
            line_1_text.append('3D (Front)')
            line_1_text.append('3D (Back)')
        else:
            pred_front_code_l.append((reversed_rgb_c.clone()[0] + 10*pred_front_code[..., i*3:(i+1)*3].sum(dim=0).reshape((s0,s1,s2))) / (pred_mask_s_copy.sum(dim=0)*10.0 + 1))
            pred_back_code_l.append((reversed_rgb_c.clone()[0] + 10*pred_back_code[..., i*3:(i+1)*3].sum(dim=0).reshape((s0,s1,s2))) / (pred_mask_s_copy.sum(dim=0)*10.0 + 1))
            text_front_list.append('C%s (Front)'%str(i))
            text_back_list.append('C%s (Back)'%str(i))
        if i > 5:
            break
    
    save_numpy_line_1 = torch.cat(line_1, dim = 1).detach().cpu().numpy()
    for i in range(len(line_1_text)):
        cv2.putText(save_numpy_line_1, line_1_text[i], (i * 640 + 20, 60), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0) , 4, cv2.LINE_AA)
    save_numpy_front = torch.cat(pred_front_code_l, dim = 1).detach().cpu().numpy()
    for i in range(len(text_front_list)):
        cv2.putText(save_numpy_front, text_front_list[i], (i * 640 + 20, 60), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0) , 4, cv2.LINE_AA)
    save_numpy_back = torch.cat(pred_back_code_l, dim = 1).detach().cpu().numpy()
    for i in range(len(text_back_list)):
        cv2.putText(save_numpy_back, text_back_list[i], (i * 640 + 20, 60), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0) , 4, cv2.LINE_AA)
    
    for i, (obj_id, Rt_i) in enumerate(zip(obj_ids_l, Rts_l)):
        BBox_3d = BBox_3d_l[obj_ids_all.index(obj_id)].copy().reshape((-1, 3))
        BBox_3d = (Rt_i[:3,:3] @ BBox_3d.T).T + Rt_i[:3,3:].reshape((-1, 3))
        BBox_3d[:, 0] = BBox_3d[:, 0] / BBox_3d[:, 2] * cam_K[0,0] + cam_K[0,2]
        BBox_3d[:, 1] = BBox_3d[:, 1] / BBox_3d[:, 2] * cam_K[1,1] + cam_K[1,2]
        BBox_2d = BBox_3d[:, :2].reshape((-1, 2, 2))
        
        rand_RGB_np = rand_RGB[i].clone().cpu().numpy() * 255
        rand_RGB_np = rand_RGB_np.astype(np.uint8)
        line_bgr = (int(rand_RGB_np[2]), int(rand_RGB_np[1]), int(rand_RGB_np[0]))
        for BBox_2d_i in BBox_2d:
            cv2.line(save_numpy_line_1, BBox_2d_i[0].astype(np.int32), BBox_2d_i[1].astype(np.int32), line_bgr, 2)
    
    save_numpy_2 = np.concatenate([reversed_rgb_c.clone().cpu().numpy()[0], save_numpy_line_1[:, :int(save_numpy_line_1.shape[1] / 3), :]], axis = 1)
    
    save_numpy_line_1 = cv2.resize(save_numpy_line_1, (save_numpy_line_1.shape[1] * 2, save_numpy_line_1.shape[0] * 2))
    
    save_numpy = np.concatenate([save_numpy_line_1, save_numpy_front, save_numpy_back], axis = 0)
    
    return save_numpy, save_numpy_2
