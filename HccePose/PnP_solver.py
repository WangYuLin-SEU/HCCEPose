import cv2
import numpy as np 
import itertools

def solve_PnP(pred_m_f_c_np, pnp_op = 2, reprojectionError = 1.5, bfu = None):
    return_info = {
        'success' : False,
        'rot' : np.eye(3),
        'tvecs' : np.zeros((3, 1)),
        'inliers' : np.zeros((1)),
    }
    if len(pred_m_f_c_np) == 4:
        pred_mask_np, pred_code_0_np, coord_image_np, cam_K = pred_m_f_c_np
        pred_front_code_0_np_cs = None
    else:
        pred_mask_np, pred_code_0_np, coord_image_np, cam_K, pred_front_code_0_np_c = pred_m_f_c_np
        pred_front_code_0_np_cs = pred_front_code_0_np_c[pred_mask_np > 0, :].astype(np.float32)
    pred_code_0_np_s = pred_code_0_np[pred_mask_np > 0, :].astype(np.float32)
    coord_image_np_s = coord_image_np[pred_mask_np > 0, :].astype(np.float32)
    if bfu == 'bf' and coord_image_np_s.shape[0] != 0:
        index_l = np.arange(int(coord_image_np_s[0].shape[0]/2)) * 2
        Points_3D = pred_front_code_0_np_c.copy()
        Points_3D[index_l,:] = pred_code_0_np[index_l,:]
        pred_code_0_np = Points_3D
    if bfu == 'bfu' and coord_image_np_s.shape[0] != 0:
        len_u = 10
        Points_3D_ultra = pred_front_code_0_np_c.copy()
        index_l = np.arange(int(pred_code_0_np.shape[0]/(len_u + 1))-1) * (len_u + 1)
        Points_3D_ultra[index_l + 1] = pred_code_0_np[index_l + 1]
        for r_i_ in range(len_u-1):
            r_i_i = (r_i_+1) / len_u
            Points_3D_tmp = r_i_i * pred_front_code_0_np_c + (1 - r_i_i) * pred_code_0_np
            Points_3D_ultra[index_l + 2 + r_i_] = Points_3D_tmp[index_l + 2 + r_i_]
        pred_code_0_np = Points_3D_ultra
        
    if pred_code_0_np_s.shape[0] <= 6:
        return return_info
    if pnp_op == 0:
        success, rvecs, tvecs = cv2.solvePnP(pred_code_0_np_s.astype(np.float32),coord_image_np_s.astype(np.float32),  
                                                cam_K, None, flags=cv2.SOLVEPNP_EPNP)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        if success is False:
            rot = np.eye(3)
            tvecs = np.zeros((3, 1))
        if pred_front_code_0_np_cs is None:
            reprojection, _ = cv2.projectPoints(pred_code_0_np_s, rvecs, tvecs, cam_K, None)
        else:
            reprojection, _ = cv2.projectPoints(pred_front_code_0_np_cs, rvecs, tvecs, cam_K, None)
        reprojection = reprojection.reshape((-1,2))
        error = np.linalg.norm(reprojection - coord_image_np_s, axis = 1)
        inliers = np.where(error < reprojectionError )[0].reshape((-1,1))
        return_info['success'] = success
        return_info['rot'] = rot
        return_info['tvecs'] = tvecs
        if success:
            return_info['inliers'] = inliers
    elif pnp_op == 1:
        # implementation of HccePose(BF) paper, but slow
        success, rvecs_1, tvecs_1, inliers = cv2.solvePnPRansac(pred_code_0_np_s.astype(np.float32),
                                                    coord_image_np_s.astype(np.float32), cam_K, distCoeffs=None,
                                                    reprojectionError=reprojectionError, iterationsCount=150, flags=cv2.SOLVEPNP_EPNP)
        if success:
            rvecs_2, tvecs_2 = cv2.solvePnPRefineVVS(pred_code_0_np_s[inliers[:, 0], :].astype(np.float32),
                                                coord_image_np_s[inliers[:, 0], :].astype(np.float32),
                                                cam_K, np.zeros((5)),rvecs_1,tvecs_1)
            if pred_front_code_0_np_cs is None:
                reprojection_2, _ = cv2.projectPoints(pred_code_0_np_s, rvecs_2, tvecs_2, cam_K, None)
            else:
                reprojection_2, _ = cv2.projectPoints(pred_front_code_0_np_cs, rvecs_2, tvecs_2, cam_K, None)
            reprojection_2 = reprojection_2.reshape((-1,2))
            error_2 = np.linalg.norm(reprojection_2 - coord_image_np_s, axis = 1)
            inliers_2 = np.where(error_2 < reprojectionError )[0].reshape((-1,1))
        if success:
            if inliers.shape[0] > inliers_2.shape[0]:rvecs = rvecs_1; tvecs = tvecs_1
            else:rvecs = rvecs_2; tvecs = tvecs_2; inliers = inliers_2
        else:rvecs = rvecs_1; tvecs = tvecs_1
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        return_info['success'] = success
        return_info['rot'] = rot
        return_info['tvecs'] = tvecs
        if success:
            return_info['inliers'] = inliers
    elif pnp_op == 2:
        success, rvecs, tvecs, inliers = cv2.solvePnPRansac(pred_code_0_np_s.astype(np.float32),
                                                    coord_image_np_s.astype(np.float32), cam_K, distCoeffs=None,
                                                    reprojectionError=reprojectionError, iterationsCount=150, flags=cv2.SOLVEPNP_EPNP)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        return_info['success'] = success
        return_info['rot'] = rot
        return_info['tvecs'] = tvecs
        if success:
            return_info['inliers'] = inliers
        
    return return_info


def solve_PnP_comb(pred_m_bf_c_np, keypoints_=None, pnp_op = 2, reprojectionError = 1.5, train =False):
    
    np.random.seed(0)
    
    pred_mask_np, pred_front_code_0_np, pred_back_code_0_np, coord_image_np, cam_K = pred_m_bf_c_np
    
    input_f = (pred_mask_np, pred_front_code_0_np, coord_image_np, cam_K)
    input_bfu = (pred_mask_np, pred_back_code_0_np, coord_image_np, cam_K, pred_front_code_0_np)
    
    
    results_f = solve_PnP(input_f, pnp_op = pnp_op, reprojectionError = reprojectionError)
    results_b = solve_PnP(input_bfu, pnp_op = pnp_op, reprojectionError = reprojectionError)
    results_bf = solve_PnP(input_bfu, pnp_op = pnp_op, reprojectionError = reprojectionError, bfu='bf')
    results_bfu = solve_PnP(input_bfu, pnp_op = pnp_op, reprojectionError = reprojectionError, bfu='bfu')
    
    info_list = []
    info_list.append({'rot' : results_f['rot'], 'tvecs' : results_f['tvecs'], 'success' : results_f['success'], 'num' : results_f['inliers'].shape[0],})
    info_list.append({'rot' : results_b['rot'], 'tvecs' : results_b['tvecs'], 'success' : results_b['success'], 'num' : results_b['inliers'].shape[0],})
    info_list.append({'rot' : results_bf['rot'], 'tvecs' : results_bf['tvecs'], 'success' : results_bf['success'], 'num' : results_bf['inliers'].shape[0],})
    info_list.append({'rot' : results_bfu['rot'], 'tvecs' : results_bfu['tvecs'], 'success' : results_bfu['success'], 'num' : results_bfu['inliers'].shape[0],})
    
    if train is False and keypoints_ is not None:
        keypoints_max_id = np.argmax(keypoints_)
        i_c = 0
        results_best = None
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