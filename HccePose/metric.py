from HccePose.bop_loader import pose_error
import numpy as np


def add_s(obj_ply, obj_info, gt_list, pred_list):
    pts = obj_ply['pts']
    e_list = []
    for (gt_Rt, pred_Rt) in zip(gt_list, pred_list):
        if 'symmetries_discrete' in obj_info or 'symmetries_continuous' in obj_info:
            e = pose_error.adi(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        else:
            e = pose_error.add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        e_list.append(e)
    e_list = np.array(e_list)
    
    pass_list = e_list.copy()
    
    pass_list[pass_list < 0.1 * obj_info['diameter']] = 0
    pass_list[pass_list > 0] = -1
    pass_list += 1
    
    return np.mean(pass_list), pass_list, e_list