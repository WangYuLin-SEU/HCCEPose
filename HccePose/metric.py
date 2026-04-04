"""Evaluation metrics for HccePose pose estimation.

This module currently wraps the ADD / ADD-S style metric used in BOP-style
6D pose evaluation and converts per-sample pose errors into pass ratios.

HccePose 的位姿评估指标。

该模块目前封装了 BOP 风格 6D 位姿评估中使用的 ADD / ADD-S 指标，
并将逐样本位姿误差转换为通过率统计结果。
"""

from HccePose.bop_loader import pose_error
import numpy as np


def add_s(obj_ply, obj_info, gt_list, pred_list):
    '''
    ---
    ---
    Compute ADD or ADD-S pass statistics for one object.
    ---
    ---
    The function selects ADD-S for symmetric objects and ADD for asymmetric
    ones, then converts the raw pose error list into a binary pass list using
    the standard 10% diameter threshold.

    Args:
        - obj_ply: Loaded object model containing point coordinates.
        - obj_info: Object meta information, including diameter and symmetry.
        - gt_list: Ground-truth pose list.
        - pred_list: Predicted pose list.

    Returns:
        - mean_pass: Mean pass ratio over all compared poses.
        - pass_list: Binary pass/fail list.
        - e_list: Raw ADD / ADD-S pose error list.
    ---
    ---
    计算单个物体的 ADD 或 ADD-S 通过率统计。
    ---
    ---
    该函数会针对对称物体自动选择 ADD-S、针对非对称物体选择 ADD，
    然后再按照常用的 10% 物体直径阈值把原始位姿误差转换为二值通过列表。

    参数:
        - obj_ply: 已加载的物体模型点云。
        - obj_info: 物体元信息，包含直径和对称性信息。
        - gt_list: 真值位姿列表。
        - pred_list: 预测位姿列表。

    返回:
        - mean_pass: 全部样本的平均通过率。
        - pass_list: 二值化的通过/失败列表。
        - e_list: 原始 ADD / ADD-S 位姿误差列表。
    '''
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