"""PnP solving utilities for HccePose.

This module converts HccePose front/back coordinate predictions into 6D poses
with OpenCV PnP solvers. It contains both the basic single-branch solver and
the combined front/back strategy used by the current inference pipeline.

HccePose 的 PnP 求解工具。

该模块负责将 HccePose 预测得到的正背面坐标结果转换为 6D 位姿，
底层依赖 OpenCV 的 PnP 求解器。模块同时包含基础单分支求解逻辑，
以及当前推理流程中使用的正背面组合求解策略。
"""

import cv2
import numpy as np 
import itertools

def solve_PnP(pred_m_f_c_np, pnp_op = 2, reprojectionError = 1.5, bfu = None, iterationsCount=150):
    '''
    ---
    ---
    Solve one PnP problem from predicted mask/coordinate correspondences.
    ---
    ---
    The function first extracts valid 2D-3D correspondences from the predicted
    mask, then optionally builds different front/back point combinations, and
    finally applies one of several OpenCV PnP variants to estimate the object
    pose.

    Args:
        - pred_m_f_c_np: Tuple that stores mask, predicted coordinates, image
          coordinates, and camera intrinsics.
        - pnp_op: PnP backend option.
        - reprojectionError: Inlier threshold in pixels.
        - bfu: Optional front/back point fusion mode.
        - iterationsCount: Maximum RANSAC iteration count.

    Returns:
        - return_info: Dictionary containing success flag, rotation, translation,
          and inlier indices.
    ---
    ---
    根据预测的掩膜与坐标对应关系求解单次 PnP。
    ---
    ---
    该函数会先从预测掩膜中提取有效的 2D-3D 对应点，再根据需要构造不同的
    正背面点组合，最后调用 OpenCV 的若干 PnP 变体来估计物体位姿。

    参数:
        - pred_m_f_c_np: 保存掩膜、预测坐标、图像坐标与相机内参的元组。
        - pnp_op: PnP 求解方式选项。
        - reprojectionError: 像素重投影内点阈值。
        - bfu: 可选的正背面点融合模式。
        - iterationsCount: RANSAC 最大迭代次数。

    返回:
        - return_info: 包含成功标记、旋转、平移和内点索引的字典。
    '''
    
    return_info = {
        'success' : False,
        'rot' : np.eye(3),
        'tvecs' : np.zeros((3, 1)),
        'inliers' : np.zeros((1)),
    }
    if len(pred_m_f_c_np) == 4:
        pred_mask_np, pred_front, coord_image_np, cam_K = pred_m_f_c_np
    else:
        pred_mask_np, pred_back, coord_image_np, cam_K, pred_front = pred_m_f_c_np
        pred_back = pred_back[pred_mask_np > 0, :].astype(np.float32)
    pred_front = pred_front[pred_mask_np > 0, :].astype(np.float32)
    coord_image_np = coord_image_np[pred_mask_np > 0, :].astype(np.float32)
    if bfu == 'bf' and coord_image_np.shape[0] != 0:
        index_l = np.arange(int(coord_image_np.shape[0]/2)) * 2
        Points_3D = pred_front.copy()
        Points_3D[index_l,:] = pred_back[index_l,:]
    elif bfu == 'bfu' and coord_image_np.shape[0] != 0:
        len_u = 10
        Points_3D = pred_front.copy()
        index_l = np.arange(int(pred_back.shape[0]/(len_u + 1))-1) * (len_u + 1)
        Points_3D[index_l + 1] = pred_back[index_l + 1]
        for r_i_ in range(len_u-1):
            r_i_i = (r_i_+1) / len_u
            Points_3D_tmp = r_i_i * pred_front + (1 - r_i_i) * pred_back
            Points_3D[index_l + 2 + r_i_] = Points_3D_tmp[index_l + 2 + r_i_]
    elif bfu == 'b' and coord_image_np.shape[0] != 0:
        Points_3D = pred_back
    else:
        Points_3D = pred_front
        
    if Points_3D.shape[0] <= 4:
        return return_info
    Points_3D = np.ascontiguousarray(Points_3D.astype(np.float32))
    coord_image_np = np.ascontiguousarray(coord_image_np.astype(np.float32))
    cam_K = np.ascontiguousarray(cam_K)
    if pnp_op == 0:
        success, rvecs, tvecs = cv2.solvePnP(Points_3D.astype(np.float32), coord_image_np.astype(np.float32),  
                                                cam_K, None, flags=cv2.SOLVEPNP_EPNP)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        if success is False:
            rot = np.eye(3)
            tvecs = np.zeros((3, 1))
        reprojection, _ = cv2.projectPoints(pred_front, rvecs, tvecs, cam_K, None)
        reprojection = reprojection.reshape((-1,2))
        error = np.linalg.norm(reprojection - coord_image_np, axis = 1)
        inliers = np.where(error < reprojectionError )[0].reshape((-1,1))
        return_info['success'] = success
        return_info['rot'] = rot
        return_info['tvecs'] = tvecs
        return_info['inliers'] = inliers
    elif pnp_op == 1:
        try:
            success, rvecs_1, tvecs_1, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                        coord_image_np.astype(np.float32), cam_K, distCoeffs=None,
                                                        reprojectionError=reprojectionError, confidence=0.995,
                                                        iterationsCount=iterationsCount, flags=cv2.SOLVEPNP_SQPNP)
        except:
            success, rvecs_1, tvecs_1, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                        coord_image_np.astype(np.float32), cam_K, distCoeffs=None,
                                                        reprojectionError=reprojectionError, confidence=0.995,
                                                        iterationsCount=iterationsCount, flags=cv2.SOLVEPNP_EPNP)
        reprojection_1, _ = cv2.projectPoints(Points_3D, rvecs_1, tvecs_1, cam_K, None)
        reprojection_1 = reprojection_1.reshape((-1,2))
        error_1 = np.linalg.norm(reprojection_1 - coord_image_np, axis = 1)
        inliers_1 = np.where(error_1 < reprojectionError )[0].reshape((-1,1))
        if success:
            rvecs_2, tvecs_2 = cv2.solvePnPRefineVVS(Points_3D[inliers[:, 0], :].astype(np.float32),
                                                coord_image_np[inliers[:, 0], :].astype(np.float32),
                                                cam_K, np.zeros((5)),rvecs_1,tvecs_1)
            reprojection_2, _ = cv2.projectPoints(Points_3D, rvecs_2, tvecs_2, cam_K, None)
            reprojection_2 = reprojection_2.reshape((-1,2))
            error_2 = np.linalg.norm(reprojection_2 - coord_image_np, axis = 1)
            inliers_2 = np.where(error_2 < reprojectionError )[0].reshape((-1,1))
        if success:
            if inliers_1.shape[0] > inliers_2.shape[0]:rvecs = rvecs_1; tvecs = tvecs_1
            else:rvecs = rvecs_2; tvecs = tvecs_2
        else:rvecs = rvecs_1; tvecs = tvecs_1
        
        reprojection, _ = cv2.projectPoints(pred_front, rvecs, tvecs, cam_K, None)
        reprojection = reprojection.reshape((-1,2))
        error = np.linalg.norm(reprojection - coord_image_np, axis = 1)
        inliers = np.where(error < reprojectionError )[0].reshape((-1,1))
        
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        return_info['success'] = success
        return_info['rot'] = rot
        return_info['tvecs'] = tvecs
        return_info['inliers'] = inliers
    elif pnp_op == 2:
        success, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                    coord_image_np.astype(np.float32), cam_K, distCoeffs=None,
                                                    reprojectionError=reprojectionError, 
                                                    iterationsCount=iterationsCount, flags=cv2.SOLVEPNP_EPNP)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        return_info['success'] = success
        return_info['rot'] = rot
        return_info['tvecs'] = tvecs
        
        reprojection, _ = cv2.projectPoints(pred_front, rvecs, tvecs, cam_K, None)
        reprojection = reprojection.reshape((-1,2))
        error = np.linalg.norm(reprojection - coord_image_np, axis = 1)
        inliers = np.where(error < reprojectionError )[0].reshape((-1,1))
        return_info['inliers'] = inliers
        
    if np.isnan(rot).any() or np.isinf(rot).any():
        return_info['rot'] = np.eye(3)
    if np.isnan(tvecs).any() or np.isinf(tvecs).any():
        return_info['tvecs'] = np.zeros((3, 1))
    return return_info


def solve_PnP_comb(pred_m_bf_c_np, keypoints_=None, pnp_op = 2, reprojectionError = 1.5, train =False, iterationsCount=150):
    '''
    ---
    ---
    Solve a combined front/back PnP problem and select the best candidate.
    ---
    ---
    The current HccePose inference path evaluates several front/back coordinate
    combinations, compares their inlier counts, and optionally uses the stored
    keypoint prior to pick the final pose candidate.

    Args:
        - pred_m_bf_c_np: Tuple with mask, front code, back code, image
          coordinates, and camera intrinsics.
        - keypoints_: Optional keypoint prior for candidate selection.
        - pnp_op: PnP backend option.
        - reprojectionError: Inlier threshold in pixels.
        - train: Whether to return all candidates for training-time use.
        - iterationsCount: Maximum RANSAC iteration count.

    Returns:
        - results_best / info_list: The selected best candidate during inference,
          or the full candidate list during training mode.
    ---
    ---
    求解正背面组合 PnP，并从多个候选中选择最佳位姿。
    ---
    ---
    当前 HccePose 推理流程会同时评估若干种正背面坐标组合，并比较它们的内点数，
    在需要时还会结合保存的关键点先验选择最终位姿候选。

    参数:
        - pred_m_bf_c_np: 包含掩膜、正面编码、背面编码、图像坐标与相机内参的元组。
        - keypoints_: 可选的关键点先验，用于候选筛选。
        - pnp_op: PnP 求解方式选项。
        - reprojectionError: 像素重投影内点阈值。
        - train: 是否在训练阶段返回全部候选。
        - iterationsCount: RANSAC 最大迭代次数。

    返回:
        - results_best / info_list: 推理阶段返回筛选出的最佳候选，训练阶段返回全部候选列表。
    '''
    
    np.random.seed(0)
    
    pred_mask_np, pred_front_code_0_np, pred_back_code_0_np, coord_image_np, cam_K = pred_m_bf_c_np
    
    input_f = (pred_mask_np, pred_front_code_0_np, coord_image_np, cam_K)
    input_bfu = (pred_mask_np, pred_back_code_0_np, coord_image_np, cam_K, pred_front_code_0_np)
    
    
    results_f = solve_PnP(input_f, pnp_op = pnp_op, reprojectionError = reprojectionError, iterationsCount=iterationsCount)
    results_b = solve_PnP(input_bfu, pnp_op = pnp_op, reprojectionError = reprojectionError, iterationsCount=iterationsCount, bfu='b')
    results_bfu = solve_PnP(input_bfu, pnp_op = pnp_op, reprojectionError = reprojectionError, iterationsCount=iterationsCount, bfu='bfu')
    results_bf = solve_PnP(input_bfu, pnp_op = pnp_op, reprojectionError = reprojectionError, iterationsCount=iterationsCount, bfu='bf')
    
    info_list = []
    if results_f['success']:
        info_list.append({'rot' : results_f['rot'], 'tvecs' : results_f['tvecs'], 'success' : results_f['success'], 'num' : results_f['inliers'].shape[0],})
    else:
        info_list.append({'rot' : results_f['rot'], 'tvecs' : results_f['tvecs'], 'success' : results_f['success'], 'num' : 0,})
    if results_b['success']:
        info_list.append({'rot' : results_b['rot'], 'tvecs' : results_b['tvecs'], 'success' : results_b['success'], 'num' : results_b['inliers'].shape[0],})
    else:
        info_list.append({'rot' : results_b['rot'], 'tvecs' : results_b['tvecs'], 'success' : results_b['success'], 'num' : 0,})
    if pnp_op == 1:
        if results_bfu['success']:
            info_list.append({'rot' : results_bfu['rot'], 'tvecs' : results_bfu['tvecs'], 'success' : results_bfu['success'], 'num' : results_bfu['inliers'].shape[0],})
        else:
            info_list.append({'rot' : results_bfu['rot'], 'tvecs' : results_bfu['tvecs'], 'success' : results_bfu['success'], 'num' : 0,})
        if results_bf['success']:
            info_list.append({'rot' : results_bf['rot'], 'tvecs' : results_bf['tvecs'], 'success' : results_bf['success'], 'num' : results_bf['inliers'].shape[0],})
        else:
            info_list.append({'rot' : results_bf['rot'], 'tvecs' : results_bf['tvecs'], 'success' : results_bf['success'], 'num' : 0,})
    else:
        if results_bf['success']:
            info_list.append({'rot' : results_bf['rot'], 'tvecs' : results_bf['tvecs'], 'success' : results_bf['success'], 'num' : results_bf['inliers'].shape[0],})
        else:
            info_list.append({'rot' : results_bf['rot'], 'tvecs' : results_bf['tvecs'], 'success' : results_bf['success'], 'num' : 0,})
        if results_bfu['success']:
            info_list.append({'rot' : results_bfu['rot'], 'tvecs' : results_bfu['tvecs'], 'success' : results_bfu['success'], 'num' : results_bfu['inliers'].shape[0],})
        else:
            info_list.append({'rot' : results_bfu['rot'], 'tvecs' : results_bfu['tvecs'], 'success' : results_bfu['success'], 'num' : 0,})
            
    if train is False and keypoints_ is not None:
        keypoints_max_id = np.argmax(keypoints_)
        i_c = 0
        results_best = {
            'rot' : np.eye(3), 'tvecs' : np.zeros((3, 1)), 'success' : False, 'num' : 0,
        }
        for i_ in range(len(info_list)):
            info_list_i = itertools.combinations(info_list, len(info_list) - i_)
            for info_list_i_j in info_list_i:
                if keypoints_max_id == i_c:
                    best_s = 0
                    for info_list_i_j_k in info_list_i_j:
                        if info_list_i_j_k['num'] > best_s:
                            best_s = info_list_i_j_k['num']
                            results_best = info_list_i_j_k
                i_c += 1
                                
        return results_best
    else:
        return info_list