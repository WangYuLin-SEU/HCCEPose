# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import math, kornia, torch, trimesh, cv2
import re
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
import numpy as np
import nvdiffrast.torch as dr
import torch.nn.functional as F

glcam_in_cvcam = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]]).astype(float)

@dataclass
class BatchPoseData:
        '''
        ---
        ---
        Container for paired pose-refinement training or inference tensors.
        ---
        ---
        The structure stores RGB, depth, pose, crop and geometry tensors used by
        FoundationPose refiner/scorer preprocessing and forward routines.
        ---
        ---
        用于成对位姿微调训练或推理张量的数据容器。
        ---
        ---
        该结构保存 FoundationPose 的 refiner/scorer 预处理与前向过程中
        需要的 RGB、深度、位姿、裁剪和几何张量。
        '''

        rgbs: torch.Tensor = None
        object_datas = None
        bboxes: torch.Tensor = None
        K: torch.Tensor = None
        depths: Optional[torch.Tensor] = None
        rgbAs = None
        rgbBs = None
        depthAs = None
        depthBs = None
        normalAs = None
        normalBs = None
        poseA = None  #(B,4,4)
        poseB = None
        targets = None  # Score targets, torch tensor (B)

        def __init__(self, rgbAs=None, rgbBs=None, depthAs=None, depthBs=None, normalAs=None, normalBs=None, maskAs=None, maskBs=None, poseA=None, poseB=None, xyz_mapAs=None, xyz_mapBs=None, tf_to_crops=None, Ks=None, crop_masks=None, model_pts=None, mesh_diameters=None, labels=None):
                self.rgbAs = rgbAs
                self.rgbBs = rgbBs
                self.depthAs = depthAs
                self.depthBs = depthBs
                self.normalAs = normalAs
                self.normalBs = normalBs
                self.poseA = poseA
                self.poseB = poseB
                self.maskAs = maskAs
                self.maskBs = maskBs
                self.xyz_mapAs = xyz_mapAs
                self.xyz_mapBs = xyz_mapBs
                self.tf_to_crops = tf_to_crops
                self.crop_masks = crop_masks
                self.Ks = Ks
                self.model_pts = model_pts
                self.mesh_diameters = mesh_diameters
                self.labels = labels


class PairH5Dataset(torch.utils.data.Dataset):
    '''
    ---
    ---
    Shared preprocessing dataset wrapper for FoundationPose pair inputs.
    ---
    ---
    The dataset object is reused as a light utility class to normalize RGB and
    depth-derived xyz maps before they are fed into the refiner or scorer.
    ---
    ---
    FoundationPose 成对输入的共享预处理数据封装。
    ---
    ---
    该 dataset 对象在这里被当作轻量工具类复用，用于在 refiner/scorer 前向前
    统一归一化 RGB 和由深度生成的 xyz map。
    '''
    def __init__(self, cache_data=None):
        self.n_perturb = None
        self.H_ori = None
        self.W_ori = None
        self.cache_data = cache_data

    def __len__(self):
        return 1

    def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori):
        bs = len(batch.rgbAs)
        H,W = batch.rgbAs.shape[-2:]
        mesh_radius = batch.mesh_diameters.to(batch.rgbAs.device)/2
        tf_to_crops = batch.tf_to_crops.to(batch.rgbAs.device)
        crop_to_oris = torch.linalg.inv(batch.tf_to_crops).to(batch.rgbAs.device)  #(B,3,3)
        batch.poseA = batch.poseA.to(batch.rgbAs.device)
        batch.Ks = batch.Ks.to(batch.rgbAs.device)

        if batch.xyz_mapAs is None:
            depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.to(batch.rgbAs.device).expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
            batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
            batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
        batch.xyz_mapAs = batch.xyz_mapAs.to(batch.rgbAs.device)
        invalid = batch.xyz_mapAs[:,2:3]<0.001
        batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
        batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
        batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

        if batch.xyz_mapBs is None and batch.depthBs is not None:
            depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.to(batch.rgbAs.device).expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
            batch.xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
            batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
        if batch.xyz_mapBs is not None:
            batch.xyz_mapBs = batch.xyz_mapBs.to(batch.rgbAs.device)
            invalid = batch.xyz_mapBs[:,2:3]<0.001
            batch.xyz_mapBs = batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
            batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
            invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
            batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

        return batch

    def transform_batch(self, batch:BatchPoseData, H_ori, W_ori):
        '''Transform the batch before feeding to the network
        !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
        '''
        batch.rgbAs = batch.rgbAs.float()/255.0
        batch.rgbBs = batch.rgbBs.float()/255.0

        batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori)
        return batch

class PoseRefinePairH5Dataset(PairH5Dataset):
    '''
    ---
    ---
    Pair preprocessing wrapper specialized for FoundationPose refinement.
    ---
    ---
    FoundationPose refinement 专用的成对预处理封装。
    '''
    def __init__(self, cache_data=None):
        super().__init__(cache_data=cache_data)

    def transform_batch(self, batch:BatchPoseData, H_ori, W_ori):
        '''Transform the batch before feeding to the network
        !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
        '''
        batch.rgbAs = batch.rgbAs/255.0
        batch.rgbBs = batch.rgbBs/255.0

        batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori)

        return batch

class PoseScorePairH5Dataset(PairH5Dataset):
    '''
    ---
    ---
    Pair preprocessing wrapper specialized for FoundationPose scoring.
    ---
    ---
    FoundationPose 打分阶段专用的成对预处理封装。
    '''
    def __init__(self, cache_data=None):
        super().__init__(cache_data=cache_data)

    def transform_batch(self, batch:BatchPoseData, H_ori, W_ori):
        if batch.rgbAs is not None:
            batch.rgbAs = batch.rgbAs/255.0
        if batch.rgbBs is not None:
            batch.rgbBs = batch.rgbBs/255.0

        batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori)
        return batch

def depth2xyzmap_batch(depths, Ks, zfar):
    '''
    ---
    ---
    Convert a batch of depth maps to xyz maps in camera coordinates.
    ---
    ---
    Args:
        - depths: Torch tensor with shape `(B, H, W)`.
        - Ks: Torch tensor with shape `(B, 3, 3)`.
        - zfar: Maximum valid depth value.

    Returns:
        - xyz_maps: Torch tensor with shape `(B, H, W, 3)`.
    ---
    ---
    将一批深度图转换为相机坐标系下的 xyz map。
    ---
    ---
    参数:
        - depths: 形状为 `(B, H, W)` 的 torch 张量。
        - Ks: 形状为 `(B, 3, 3)` 的 torch 张量。
        - zfar: 有效深度的最大值。

    返回:
        - xyz_maps: 形状为 `(B, H, W, 3)` 的 torch 张量。
    '''
    bs = depths.shape[0]
    invalid_mask = (depths<0.001) | (depths>zfar)
    H,W = depths.shape[-2:]
    vs,us = torch.meshgrid(torch.arange(0,H),torch.arange(0,W), indexing='ij')
    vs = vs.reshape(-1).float().to(depths.device)[None].expand(bs,-1)
    us = us.reshape(-1).float().to(depths.device)[None].expand(bs,-1)
    zs = depths.reshape(bs,-1)
    Ks = Ks[:,None].expand(bs,zs.shape[-1],3,3)
    xs = (us-Ks[...,0,2])*zs/Ks[...,0,0]  #(B,N)
    ys = (vs-Ks[...,1,2])*zs/Ks[...,1,1]
    pts = torch.stack([xs,ys,zs], dim=-1)  #(B,N,3)
    xyz_maps = pts.reshape(bs,H,W,3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps


def compute_crop_window_tf_batch(poses=None, K=None, crop_ratio=1.2, out_size=None, mesh_diameter=None):

    '''
    ---
    ---
    Compute crop transforms that tightly cover projected object centers.
    ---
    ---
    The function estimates a square crop around each pose by projecting a small
    radius around the object center and then building the affine transform that
    maps the crop to the requested output size.

    Args:
        - poses: Pose tensor with shape `(B, 4, 4)`.
        - K: Camera intrinsics.
        - crop_ratio: Expansion ratio applied to the crop radius.
        - out_size: Output crop size.
        - mesh_diameter: Object diameter used to estimate crop radius.

    Returns:
        - tf_to_crop: Crop transform tensor.
    ---
    ---
    计算能够紧致覆盖投影物体中心区域的裁剪变换。
    ---
    ---
    该函数会先围绕物体中心投影一个小半径区域，再构造把该裁剪区域映射到
    目标输出尺寸的仿射变换。

    参数:
        - poses: 形状为 `(B, 4, 4)` 的位姿张量。
        - K: 相机内参。
        - crop_ratio: 对裁剪半径施加的放大比例。
        - out_size: 输出裁剪尺寸。
        - mesh_diameter: 用于估计裁剪半径的物体直径。

    返回:
        - tf_to_crop: 裁剪变换张量。
    '''
    def compute_tf_batch(left, right, top, bottom):
        B = len(left)

        tf = torch.eye(3)[None].expand(B,-1,-1).contiguous()
        tf[:,0,2] = -left
        tf[:,1,2] = -top
        new_tf = torch.eye(3)[None].expand(B,-1,-1).contiguous()
        new_tf[:,0,0] = out_size[0]/(right-left)
        new_tf[:,1,1] = out_size[1]/(bottom-top)
        return new_tf@tf

    B = len(poses)
    radius = mesh_diameter*crop_ratio/2
    offsets = torch.tensor([0,0,0,
                            radius,0,0,
                            -radius,0,0,
                            0,radius,0,
                            0,-radius,0], device=poses.device,dtype=torch.float).reshape(-1,3)
    pts = poses[:,:3,3].reshape(-1,1,3)+offsets.reshape(1,-1,3)
    K = torch.as_tensor(K,device=poses.device)
    projected = (K@pts.reshape(-1,3).T).T
    projected[:, 2][projected[:, 2] == 0] = 1
    uvs = projected[:,:2]/projected[:,2:3]
    uvs = uvs.reshape(B, -1, 2)
    center = uvs[:,0]
    radius = torch.abs(uvs-center.reshape(-1,1,2)).reshape(B,-1).max(axis=-1)[0].reshape(-1)
    radius[radius == 0] = 1
    left = center[:,0]-radius
    right = center[:,0]+radius
    top = center[:,1]-radius
    bottom = center[:,1]+radius
    return compute_tf_batch(left, right, top, bottom).to(poses.device)

def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
    '''
    ---
    ---
    Convert camera intrinsics to an OpenGL-style projection matrix.
    ---
    ---
    This helper bridges standard pinhole intrinsics and the projection format
    expected by nvdiffrast rendering.
    ---
    ---
    将相机内参转换为 OpenGL 风格的投影矩阵。
    ---
    ---
    该函数连接了标准针孔相机内参与 nvdiffrast 渲染所需的投影格式。
    '''
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar

    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    if window_coords == 'y_up':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    elif window_coords == 'y_down':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])
    else:
        raise NotImplementedError

    return proj

def transform_pts(pts,tf):
    '''
    ---
    ---
    Apply a rigid transform to 3D points.
    ---
    ---
    对三维点应用刚体变换。
    '''
    """Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)
    """
    if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
        tf = tf[...,None,:,:]
    return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]

def to_homo_torch(pts):
    '''
    ---
    ---
    Convert point tensors to homogeneous coordinates.
    ---
    ---
    将点张量转换为齐次坐标形式。
    '''
    '''
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    '''
    ones = torch.ones((*pts.shape[:-1],1), dtype=torch.float, device=pts.device)
    homo = torch.cat((pts, ones),dim=-1)
    return homo

def transform_dirs(dirs,tf):
    '''
    ---
    ---
    Apply only the rotational part of a transform to direction vectors.
    ---
    ---
    仅使用变换的旋转部分来变换方向向量。
    '''
    """
    @dirs: (...,3)
    @tf: (...,4,4)
    """
    if len(tf.shape)>=3 and tf.shape[-3]!=dirs.shape[-2]:
        tf = tf[...,None,:,:]
    return (tf[...,:3,:3]@dirs[...,None])[...,0]

def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.2, extra={}, device='cuda:0'):
    '''
    ---
    ---
    Render RGB, depth and mask buffers with nvdiffrast.
    ---
    ---
    This is the shared low-level renderer used by FoundationPose preprocessing,
    inverse crop rendering and visualization utilities.
    ---
    ---
    使用 nvdiffrast 渲染 RGB、深度和掩膜缓冲。
    ---
    ---
    这是 FoundationPose 预处理、反裁剪渲染以及可视化工具共享的底层 renderer。
    '''
    '''Just plain rendering, not support any gradient
    @K: (3,3) np array
    @ob_in_cams: (N,4,4) torch tensor, openCV camera
    @projection_mat: np array (4,4)
    @output_size: (height, width)
    @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
    @light_dir: in cam space
    @light_pos: in cam space
    '''
    if glctx is None:
        if context == 'gl':
            glctx = dr.RasterizeGLContext()
        elif context=='cuda':
            glctx = dr.RasterizeCudaContext(device)
        else:
            raise NotImplementedError
    
    if mesh_tensors is None:
        mesh_tensors = make_mesh_tensors(mesh)
    pos = mesh_tensors['pos']
    vnormals = mesh_tensors['vnormals']
    pos_idx = mesh_tensors['faces']
    
    has_tex = 'tex' in mesh_tensors

    ob_in_glcams = torch.tensor(glcam_in_cvcam, device=device, dtype=torch.float32)[None]@ob_in_cams
    if projection_mat is None:
        projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100)
    projection_mat = torch.as_tensor(projection_mat.reshape(-1,4,4), device=device, dtype=torch.float)
    mtx = projection_mat@ob_in_glcams

    if output_size is None:
        output_size = np.asarray([H,W])

    pts_cam = transform_pts(pos, ob_in_cams)
    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:,None]@pos_homo[None,...,None])[...,0]
    if bbox2d is not None:
        l = bbox2d[:,0]
        t = H-bbox2d[:,1]
        r = bbox2d[:,2]
        b = H-bbox2d[:,3]
        tf = torch.eye(4, dtype=torch.float, device=device).reshape(1,4,4).expand(len(ob_in_cams),4,4).contiguous()
        tf[:,0,0] = W/(r-l)
        tf[:,1,1] = H/(t-b)
        tf[:,3,0] = (W-r-l)/(r-l)
        tf[:,3,1] = (H-t-b)/(t-b)
        pos_clip = pos_clip@tf
        
    rast_out, _ = dr.rasterize(glctx, torch.as_tensor(pos_clip, device=device, dtype=torch.float32), pos_idx, resolution=np.asarray(output_size))
    xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
    depth = xyz_map[...,2]
    if has_tex:
        texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out, mesh_tensors['uv_idx'])
        color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
    else:
        color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, pos_idx)

    if use_light:
        get_normal = True
    if get_normal:
        vnormals_cam = transform_dirs(vnormals, ob_in_cams)
        normal_map, _ = dr.interpolate(torch.as_tensor(vnormals_cam, device=device, dtype=torch.float32), rast_out, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
    else:
        normal_map = None

    if use_light:
        if light_dir is not None:
            light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device=device)
        else:
            light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device=device).reshape(1,1,3) - pts_cam
        diffuse_intensity = (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[...,None]
        diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)  # (N_pose, H, W, 1)
        if light_color is None:
            light_color = color
        else:
            light_color = torch.as_tensor(light_color, device=device, dtype=torch.float)
        color = color*w_ambient + diffuse_intensity_map*light_color*w_diffuse

    color = color.clip(0,1)
    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background using alpha
    color = torch.flip(color, dims=[1])   # Flip Y coordinates
    depth = torch.flip(depth, dims=[1])
    extra['xyz_map'] = torch.flip(xyz_map, dims=[1])
    
    
    return color, depth, normal_map


def make_crop_data_batch_train(ob_mask, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None, device = 'cuda:0'):
    '''
    ---
    ---
    Build paired training/inference crops for FoundationPose refinement.
    ---
    ---
    为 FoundationPose refinement 构造成对的训练/推理裁剪数据。
    '''
    H,W = depth.shape[1:3]
    input_resize = [160, 160]
    render_size = [160, 160]
    tf_to_crops = compute_crop_window_tf_batch(poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), mesh_diameter=mesh_diameter)

    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device=device)
    bs = 16
    rgb_rs = []
    depth_rs = []
    mask_rs = []
    xyz_map_rs = []

    bbox2d_crop = torch.as_tensor(np.array([0, 0, input_resize[0]-1, input_resize[1]-1]).reshape(2,2), device=device, dtype=torch.float)
    tf_to_crops = torch.as_tensor(tf_to_crops, device=bbox2d_crop.device, dtype=torch.float)

    identity_matrix = torch.eye(3, dtype=tf_to_crops.dtype, device=tf_to_crops.device)
    has_nan_inf = torch.isnan(tf_to_crops).any(dim=(1, 2)) | torch.isinf(tf_to_crops).any(dim=(1, 2))
    has_nan_inf = has_nan_inf.float().to(torch.float32)
    tf_to_crops[has_nan_inf>0, ...] = identity_matrix

    tf_to_crops_det = torch.linalg.det(tf_to_crops)
    tf_to_crops[tf_to_crops_det == 0] = identity_matrix
    bbox2d_ori = transform_pts(bbox2d_crop, torch.linalg.inv(tf_to_crops)).reshape(-1,4)
    xyz_map = xyz_map.detach().clone()

    for b in range(0,len(poseA),bs):
        extra = {}
        rgb_r, depth_r, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context=device, glctx=glctx, mesh_tensors=mesh_tensors, output_size=input_resize, bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra, device=device)
        rgb_rs.append(rgb_r)
        depth_rs.append(depth_r[...,None])
        mask_r = depth_r[...,None].detach().clone()
        mask_r[mask_r > 0] = 1
        mask_rs.append(mask_r)
        xyz_map_rs.append(extra['xyz_map'])
    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)
    Ks = torch.as_tensor(K, device=device, dtype=torch.float).reshape(1,3,3)
    mask_rs = torch.cat(mask_rs, dim=0).permute(0,3,1,2)
    mask_rs[mask_rs > 0] = 1
    ob_mask[ob_mask > 0] = 1
    if ob_mask.shape[0] != 1:
        maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(ob_mask[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    else:
        maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(ob_mask[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)

    maskAs = mask_rs.clone().detach()
    if rgb.shape[0] != 1:
        rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device=device).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    else:
        rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device=device).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)

    depthAs = depth_rs
    if depth.shape[0] != 1:
        depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    else:
        depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)

    rgbAs = rgb_rs
    xyz_mapAs = xyz_map_rs
    if xyz_map.shape[0] != 1:
        xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device=device, dtype=torch.float).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    else:
        xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device=device, dtype=torch.float).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)

    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device=device)*mesh_diameter
    pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=None, normalBs=None,
                           maskAs = maskAs, maskBs = maskBs,
                           poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W)
    return pose_data

def so3_exp_map(data):
    '''
    ---
    ---
    Map axis-angle residuals to SO(3) rotation matrices.
    ---
    ---
    将轴角残差映射到 SO(3) 旋转矩阵。
    '''
    mat = kornia.geometry.liegroup.So3.exp(data).matrix()

    return mat

def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
    '''
    ---
    ---
    Update poses with egocentric translation and rotation residuals.
    ---
    ---
    使用自体坐标系下的平移和旋转残差更新位姿。
    '''
    '''Used for Pose Refinement. Given the object's two poses in camera, convert them to relative poses in camera's egocentric view
    @A_in_cam: (B,4,4) torch tensor
    '''
    trans_delta = A_in_cam[:,:3,3] + trans_delta
    trans_delta = trans_delta[..., None]
    rot_mat_delta = rot_mat_delta@A_in_cam[:,:3,:3]
    RT_34 = torch.cat([rot_mat_delta, trans_delta], dim=2)
    RT_14 = torch.zeros_like(RT_34[:,0:1,:])
    RT_14[:, 0, 3] = 1
    RT_44 = torch.cat([RT_34, RT_14], dim=1)
    return RT_44


def make_crop_data_batch_train_score(ob_mask, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None, device = 'cuda:0'):
    '''
    ---
    ---
    Build paired crop data for FoundationPose scoring.
    ---
    ---
    为 FoundationPose 打分阶段构造成对裁剪数据。
    '''
    H,W = depth.shape[1:3]
    input_resize = [160, 160]
    render_size = [160, 160]
    tf_to_crops = compute_crop_window_tf_batch(poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), mesh_diameter=mesh_diameter)
    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device=device)
    bs = 16
    rgb_rs = []
    depth_rs = []
    mask_rs = []
    xyz_map_rs = []
    bbox2d_crop = torch.as_tensor(np.array([0, 0, input_resize[0]-1, input_resize[1]-1]).reshape(2,2), device=device, dtype=torch.float)
    tf_to_crops = torch.as_tensor(tf_to_crops, device=bbox2d_crop.device, dtype=torch.float)
    identity_matrix = torch.eye(3, dtype=tf_to_crops.dtype, device=tf_to_crops.device)
    has_nan_inf = torch.isnan(tf_to_crops).any(dim=(1, 2)) | torch.isinf(tf_to_crops).any(dim=(1, 2))
    has_nan_inf = has_nan_inf.float().to(torch.float32)
    tf_to_crops[has_nan_inf>0, ...] = identity_matrix
    tf_to_crops_det = torch.linalg.det(tf_to_crops)
    tf_to_crops[tf_to_crops_det == 0] = identity_matrix
    bbox2d_ori = transform_pts(bbox2d_crop, torch.linalg.inv(tf_to_crops)).reshape(-1,4)
    xyz_map = xyz_map.detach().clone()
    for b in range(0,len(poseA),bs):
        extra = {}
        rgb_r, depth_r, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context=device, glctx=glctx, mesh_tensors=mesh_tensors, output_size=input_resize, bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra, device=device)
        rgb_rs.append(rgb_r)
        depth_rs.append(depth_r[...,None])
        mask_r = depth_r[...,None].detach().clone()
        mask_r[mask_r > 0] = 1
        mask_rs.append(mask_r)
        xyz_map_rs.append(extra['xyz_map'])
    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)
    Ks = torch.as_tensor(K, device=device, dtype=torch.float).reshape(1,3,3)
    mask_rs = torch.cat(mask_rs, dim=0).permute(0,3,1,2)
    mask_rs[mask_rs > 0] = 1
    ob_mask[ob_mask > 0] = 1
    if ob_mask.shape[0] != 1:
        maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(ob_mask[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    else:
        maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(ob_mask[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    maskAs = mask_rs.clone().detach()
    if rgb.shape[0] != 1:
        rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device=device).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    else:
        rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device=device).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = depth_rs
    if depth.shape[0] != 1:
        depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    else:
        depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth[...,None], dtype=torch.float, device=device).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    rgbAs = rgb_rs
    xyz_mapAs = xyz_map_rs
    if xyz_map.shape[0] != 1:
        xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device=device, dtype=torch.float).permute(0, 3, 1, 2), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    else:
        xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device=device, dtype=torch.float).permute(0, 3, 1, 2).repeat(tf_to_crops.shape[0],1,1,1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device=device)*mesh_diameter
    pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=None, normalBs=None,
                           maskAs = maskAs, maskBs = maskBs,
                           poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W)
    return pose_data


def make_crop_data_batch_train_inv(ob_in_cams, mesh, rgb, K,glctx=None, mesh_tensors=None, device = 'cuda:0', mesh_diameter=None):
    '''
    ---
    ---
    Render inverse-crop masks back to image space for visualization.
    ---
    ---
    将反裁剪掩膜渲染回原图空间，供可视化使用。
    '''
    H,W = rgb.shape[1:3]
    input_resize = [160, 160]
    render_size = [160, 160]
    tf_to_crops = compute_crop_window_tf_batch(poses=ob_in_cams, K=K, crop_ratio=1.2, out_size=(render_size[1], render_size[0]), mesh_diameter=mesh_diameter)
    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device=device)
    bs = 512
    mask_rs = []
    bbox2d_crop = torch.as_tensor(np.array([0, 0, input_resize[0]-1, input_resize[1]-1]).reshape(2,2), device=device, dtype=torch.float)
    tf_to_crops = torch.as_tensor(tf_to_crops, device=bbox2d_crop.device, dtype=torch.float)
    identity_matrix = torch.eye(3, dtype=tf_to_crops.dtype, device=tf_to_crops.device)
    has_nan_inf = torch.isnan(tf_to_crops).any(dim=(1, 2)) | torch.isinf(tf_to_crops).any(dim=(1, 2))
    has_nan_inf = has_nan_inf.float().to(torch.float32)
    tf_to_crops[has_nan_inf>0, ...] = identity_matrix
    tf_to_crops_det = torch.linalg.det(tf_to_crops)
    tf_to_crops[tf_to_crops_det == 0] = identity_matrix
    bbox2d_ori = transform_pts(bbox2d_crop, torch.linalg.inv(tf_to_crops)).reshape(-1,4)
    for b in range(0,len(poseA),bs):
        _, depth_r, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context=device, glctx=glctx, mesh_tensors=mesh_tensors, output_size=input_resize, bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra={}, device=device)
        mask_r = depth_r[...,None].detach().clone()
        mask_r[mask_r > 0] = 1
        mask_rs.append(mask_r)
    mask_rs = torch.cat(mask_rs, dim=0).permute(0,3,1,2)
    mask_rs = kornia.geometry.transform.warp_perspective(mask_rs, torch.linalg.inv(tf_to_crops), dsize=[H,W], mode='nearest', align_corners=False)
    return mask_rs


class ConvBNReLU(nn.Module):
    '''
    ---
    ---
    Basic convolution-batchnorm-activation block used by FoundationPose networks.
    ---
    ---
    FoundationPose 网络中使用的基础卷积-归一化-激活模块。
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
        ]
        if norm_layer is not None:
          layers.append(norm_layer(C_out))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    '''
    ---
    ---
    Create a standard 3x3 convolution layer.
    ---
    ---
    创建一个标准的 3x3 卷积层。
    '''
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

class ResnetBasicBlock(nn.Module):
    '''
    ---
    ---
    Residual basic block used by the FoundationPose backbone.
    ---
    ---
    FoundationPose 主干网络使用的基础残差块。
    '''
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class PositionalEmbedding(nn.Module):
    '''
    ---
    ---
    Positional embedding layer used in the lightweight transformer heads.
    ---
    ---
    轻量 transformer 头部中使用的位置编码层。
    '''
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()[None]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        ---
        ---
        Add positional embedding to a sequence tensor.
        ---
        ---
        为序列张量添加位置编码。
        '''
        return x + self.pe[:, :x.size(1)]


class RefineNet(nn.Module):
    '''
    ---
    ---
    Neural network that predicts pose refinement residuals.
    ---
    ---
    用于预测位姿微调残差的神经网络。
    '''
    def __init__(self, c_in=6):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.encodeA = nn.Sequential(
        ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
        ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
        ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
        )
        self.encodeAB = nn.Sequential(
        ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
        ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
        ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
        )
        embed_dim = 512
        num_heads = 4
        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)
        self.trans_head = nn.Sequential(
        nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, 3),
        )
        rot_out_dim = 3
        self.rot_head = nn.Sequential(
        nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, rot_out_dim),
        )

    @torch.compiler.disable()
    def forward(self, A, B):
        """
        @A: (B,C,H,W)
        """
        bs = A.shape[0]
        output = {}
        x = torch.cat([A,B], dim=0)
        x = self.encodeA(x)
        a = x[:bs]
        b = x[bs:]
        ab = torch.cat((a,b),1).contiguous()
        ab = self.encodeAB(ab)  #(B,C,H,W)
        ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
        t_ = self.trans_head(ab)
        r_ = self.rot_head(ab)
        output['trans'] = torch.mean(t_,dim=1)
        output['rot'] = torch.mean(r_,dim=1)
        return output

class ScoreNet(nn.Module):
    '''
    ---
    ---
    Neural network that scores candidate poses.
    ---
    ---
    用于对候选位姿进行打分的神经网络。
    '''
    def __init__(self, c_in=4, L=0):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.L = L
        self.encoderA = nn.Sequential(
        ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
        ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
        ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
        )
        self.encoderAB = nn.Sequential(
        ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
        ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
        ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
        )
        embed_dim = 512
        num_heads = 4
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                         dropout=0.2, 
                                         bias=True, batch_first=True)
        self.att_cross = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                               dropout=0.2, 
                                               bias=True, batch_first=True)
        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)
        self.linear = nn.Linear(embed_dim, 1)

    def extract_feat(self, A, B):
        """
        @A: (B*L,C,H,W) L is num of pairs
        """
        bs = A.shape[0]  # B*L
        x = torch.cat([A,B], dim=0)
        x = self.encoderA(x)
        a = x[:bs]
        b = x[bs:]
        ab = torch.cat((a,b), dim=1)
        ab = self.encoderAB(ab)
        ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
        ab, _ = self.att(ab, ab, ab)
        return ab.mean(dim=1).reshape(bs,-1)

    def forward(self, A, B, L=5):
        """
        @A: (B*L,C,H,W) L is num of pairs
        @L: num of pairs
        """
        if self.L != 0:
            L = self.L
        output = {}
        bs = A.shape[0]//L
        feats = self.extract_feat(A, B)   #(B*L, C)
        x = feats.reshape(bs,L,-1)
        x, _ = self.att_cross(x, x, x)
        x = self.linear(x)
        output['score_logit'] = x.reshape(bs,L)
        return output

def _fp_parse_vis_stages(vis_stages, max_refine_stage):
    '''
    ---
    ---
    Parse FoundationPose visualization stage requests.
    ---
    ---
    解析 FoundationPose 可视化阶段请求。
    '''
    if vis_stages is None:
        return set(range(1, max_refine_stage + 1)) | {'score'}
    if isinstance(vis_stages, str):
        raw_items = [item.strip() for item in vis_stages.replace(';', ',').split(',') if item.strip()]
    elif np.isscalar(vis_stages):
        raw_items = [vis_stages]
    else:
        raw_items = list(vis_stages)

    stages = set()
    for raw_item in raw_items:
        if isinstance(raw_item, str):
            item = raw_item.strip().lower()
            if item == 'all':
                return set(range(1, max_refine_stage + 1)) | {'score'}
            if item == 'score':
                stages.add('score')
                continue
            item = item.replace('refine', '').replace('stage', '').replace('_', '').replace(' ', '')
            if item.isdigit():
                item = int(item)
            else:
                continue
        else:
            try:
                item = int(raw_item)
            except Exception:
                continue
        if 1 <= item <= max_refine_stage:
            stages.add(item)
    return stages


def _fp_add_title_bar(image, title, height=32, bg_value=20):
    '''
    ---
    ---
    Add a title bar above one FoundationPose visualization panel.
    ---
    ---
    在 FoundationPose 可视化面板上方添加标题栏。
    '''
    header = np.full((height, image.shape[1], 3), bg_value, dtype=np.uint8)
    cv2.putText(header, str(title), (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _fp_add_legend_bar(image, entries, height=44, bg_value=18):
    '''
    ---
    ---
    Add a legend bar below one FoundationPose visualization panel.
    ---
    ---
    在 FoundationPose 可视化面板下方添加图例栏。
    '''
    if entries is None or len(entries) == 0:
        return image
    legend = np.full((height, image.shape[1], 3), bg_value, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    item_h = 14
    left_pad = 8
    right_pad = 12
    row_y = [14, 32]
    x = left_pad
    row_id = 0
    for color, label in entries:
        label = str(label)
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        item_w = 14 + 8 + text_size[0] + 18
        if x + item_w > image.shape[1] - right_pad and row_id == 0:
            row_id = 1
            x = left_pad
        if x + item_w > image.shape[1] - right_pad:
            break
        y = row_y[row_id]
        cv2.rectangle(legend, (x, y - item_h + 2), (x + 14, y + 2), color, -1)
        x += 22
        cv2.putText(legend, label, (x, y), font, font_scale, (235, 235, 235), thickness, cv2.LINE_AA)
        x += text_size[0] + 18
    return np.concatenate([image, legend], axis=0)


def _fp_stack_cols(images, pad=4, bg_value=20):
    '''
    ---
    ---
    Stack multiple visualization tiles horizontally.
    ---
    ---
    将多个可视化 tile 横向拼接。
    '''
    images = [img for img in images if img is not None]
    if len(images) == 0:
        return None
    max_h = max(img.shape[0] for img in images)
    sep = np.full((max_h, pad, 3), bg_value, dtype=np.uint8)
    canvas = None
    for image in images:
        pad_h = max_h - image.shape[0]
        image_padded = cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(bg_value, bg_value, bg_value))
        if canvas is None:
            canvas = image_padded
        else:
            canvas = np.concatenate([canvas, sep, image_padded], axis=1)
    return canvas


def _fp_stack_rows(images, pad=4, bg_value=20):
    '''
    ---
    ---
    Stack multiple visualization tiles vertically.
    ---
    ---
    将多个可视化 tile 纵向拼接。
    '''
    images = [img for img in images if img is not None]
    if len(images) == 0:
        return None
    max_w = max(img.shape[1] for img in images)
    sep = np.full((pad, max_w, 3), bg_value, dtype=np.uint8)
    canvas = None
    for image in images:
        pad_w = max_w - image.shape[1]
        image_padded = cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(bg_value, bg_value, bg_value))
        if canvas is None:
            canvas = image_padded
        else:
            canvas = np.concatenate([canvas, sep, image_padded], axis=0)
    return canvas


def _fp_rgb_tensor_to_bgr(image_tensor, mask_tensor=None, size=128):
    image = image_tensor.detach().float().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.max() <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if mask_tensor is not None:
        mask = mask_tensor.detach().float().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask > 0.5
        image = image.copy()
        image[~mask] = (image[~mask].astype(np.float32) * 0.25).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image


def _fp_xyz_tensor_to_bgr(xyz_tensor, size=128):
    xyz = xyz_tensor.detach().float().cpu().numpy()
    if xyz.ndim == 3 and xyz.shape[0] >= 3:
        xyz = np.transpose(xyz[:3], (1, 2, 0))
    valid = np.linalg.norm(xyz, axis=-1) > 1e-6
    image = np.zeros_like(xyz, dtype=np.float32)
    if np.any(valid):
        scale = float(np.percentile(np.abs(xyz[valid]), 95))
        scale = max(scale, 1e-3)
        image = np.clip(xyz / (2.0 * scale) + 0.5, 0.0, 1.0)
        image[~valid] = 0
    image = (image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    return image


def _fp_mask_to_numpy(mask_tensor):
    if mask_tensor is None:
        return None
    mask = mask_tensor.detach().float().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask > 0.5


def _fp_mask_bbox(mask_tensor):
    mask = _fp_mask_to_numpy(mask_tensor)
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _fp_draw_bbox(image, bbox, color, thickness=2):
    if bbox is None:
        return image
    canvas = image.copy()
    x0, y0, x1, y1 = bbox
    cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness, cv2.LINE_AA)
    return canvas


def _fp_draw_mask_contour(image, mask_tensor, color, thickness=2):
    mask = _fp_mask_to_numpy(mask_tensor)
    if mask is None or not np.any(mask):
        return image
    canvas = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cv2.drawContours(canvas, contours, -1, color, thickness, cv2.LINE_AA)
    return canvas


def _fp_project_points_to_crop(points_obj, pose, K, tf_to_crop):
    points_obj = np.asarray(points_obj, dtype=np.float32)
    pose_np = pose.detach().float().cpu().numpy() if hasattr(pose, 'detach') else np.asarray(pose, dtype=np.float32)
    K_np = K.detach().float().cpu().numpy() if hasattr(K, 'detach') else np.asarray(K, dtype=np.float32)
    tf_np = tf_to_crop.detach().float().cpu().numpy() if hasattr(tf_to_crop, 'detach') else np.asarray(tf_to_crop, dtype=np.float32)
    points_cam = (pose_np[:3, :3] @ points_obj.T).T + pose_np[:3, 3][None]
    valid = points_cam[:, 2] > 1e-6
    uv_crop = np.full((points_obj.shape[0], 2), -1.0, dtype=np.float32)
    if np.any(valid):
        pts = points_cam[valid]
        uv = np.empty((pts.shape[0], 2), dtype=np.float32)
        uv[:, 0] = K_np[0, 0] * pts[:, 0] / pts[:, 2] + K_np[0, 2]
        uv[:, 1] = K_np[1, 1] * pts[:, 1] / pts[:, 2] + K_np[1, 2]
        uv_h = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=np.float32)], axis=1)
        uv_crop_h = (tf_np @ uv_h.T).T
        uv_crop_valid = uv_crop_h[:, :2] / np.clip(uv_crop_h[:, 2:3], 1e-6, None)
        uv_crop[valid] = uv_crop_valid
    return uv_crop, valid


def _fp_draw_projected_geometry(image, pose_data, sample_id, sample_points=None, bbox_corners=None, box_color=(0, 255, 0), point_color=(0, 220, 255), pose=None):
    canvas = image.copy()
    K = pose_data.Ks[0] if pose_data.Ks.shape[0] == 1 else pose_data.Ks[sample_id]
    tf_to_crop = pose_data.tf_to_crops[sample_id]
    if pose is None:
        pose = pose_data.poseA[sample_id]
    h, w = canvas.shape[:2]
    if bbox_corners is not None:
        bbox_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.int32)
        uv_bbox, valid_bbox = _fp_project_points_to_crop(bbox_corners, pose, K, tf_to_crop)
        for start_id, end_id in bbox_edges:
            if not (valid_bbox[start_id] and valid_bbox[end_id]):
                continue
            pt0 = tuple(np.round(uv_bbox[start_id]).astype(np.int32))
            pt1 = tuple(np.round(uv_bbox[end_id]).astype(np.int32))
            if (pt0[0] < -w or pt0[0] > 2 * w or pt0[1] < -h or pt0[1] > 2 * h or
                    pt1[0] < -w or pt1[0] > 2 * w or pt1[1] < -h or pt1[1] > 2 * h):
                continue
            cv2.line(canvas, pt0, pt1, box_color, 2, cv2.LINE_AA)
    if sample_points is not None and len(sample_points) > 0:
        uv_pts, valid_pts = _fp_project_points_to_crop(sample_points, pose, K, tf_to_crop)
        for uv in uv_pts[valid_pts]:
            x_i, y_i = np.round(uv).astype(np.int32)
            if 0 <= x_i < w and 0 <= y_i < h:
                cv2.circle(canvas, (x_i, y_i), 2, point_color, -1, cv2.LINE_AA)
    return canvas


def _fp_make_overlay(observed_rgb, rendered_rgb, render_mask_tensor=None):
    '''
    ---
    ---
    Blend observed and rendered images into one FoundationPose overlay panel.
    ---
    ---
    将观测图与渲染图混合为 FoundationPose 的 overlay 面板。
    '''
    overlay = observed_rgb.copy()
    render_mask = _fp_mask_to_numpy(render_mask_tensor)
    if render_mask is None:
        render_mask = np.max(rendered_rgb, axis=2) > 0
    if np.any(render_mask):
        overlay[render_mask] = np.clip(
            observed_rgb[render_mask].astype(np.float32) * 0.55 + rendered_rgb[render_mask].astype(np.float32) * 0.45,
            0.0,
            255.0,
        ).astype(np.uint8)
    overlay = _fp_draw_mask_contour(overlay, render_mask_tensor, (255, 0, 255), thickness=2)
    return overlay


def _fp_make_vis_geometry(mesh):
    '''
    ---
    ---
    Prepare sparse visualization geometry from one mesh.
    ---
    ---
    从一个 mesh 中准备稀疏可视化几何信息。
    '''
    if mesh is None or len(mesh.vertices) == 0:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    bbox_corners = np.array([
        [vmin[0], vmin[1], vmin[2]],
        [vmax[0], vmin[1], vmin[2]],
        [vmax[0], vmax[1], vmin[2]],
        [vmin[0], vmax[1], vmin[2]],
        [vmin[0], vmin[1], vmax[2]],
        [vmax[0], vmin[1], vmax[2]],
        [vmax[0], vmax[1], vmax[2]],
        [vmin[0], vmax[1], vmax[2]],
    ], dtype=np.float32)
    step = max(1, int(np.ceil(len(vertices) / 96)))
    sample_points = vertices[::step][:96]
    return {
        'bbox_corners': bbox_corners,
        'sample_points': sample_points,
    }

def _fp_refine_tile_title(stage_title, sample_id):
    match = re.search(r'(\d+)(?!.*\d)', str(stage_title))
    if match is not None:
        stage_id = int(match.group(1))
        return f'Pose {sample_id} | Refine {stage_id}'
    return f'Pose {sample_id} | {stage_title}'



def _fp_make_pose_pair_tile(pose_data, sample_id, tile_title, vis_geometry=None, pose_output=None, orange_mode='obs_box', orange_label='Obs mask box', green_label='Render box', cell_size=160):
    render_base = _fp_rgb_tensor_to_bgr(pose_data.rgbAs[sample_id], pose_data.maskAs[sample_id] if pose_data.maskAs is not None else None, size=cell_size)
    observed_base = _fp_rgb_tensor_to_bgr(pose_data.rgbBs[sample_id], pose_data.maskBs[sample_id] if pose_data.maskBs is not None else None, size=cell_size)

    input_pose = pose_data.poseA[sample_id]
    render_pose = input_pose if pose_output is None else pose_output

    render_rgb = render_base.copy()
    observed_rgb = observed_base.copy()
    overlay = _fp_make_overlay(observed_base, render_base, pose_data.maskAs[sample_id] if pose_data.maskAs is not None else None)

    obs_bbox = _fp_mask_bbox(pose_data.maskBs[sample_id] if pose_data.maskBs is not None else None)
    observed_rgb = _fp_draw_bbox(observed_rgb, obs_bbox, (0, 165, 255), thickness=2)

    if vis_geometry is not None:
        render_rgb = _fp_draw_projected_geometry(
            render_rgb,
            pose_data,
            sample_id,
            sample_points=vis_geometry.get('sample_points'),
            bbox_corners=vis_geometry.get('bbox_corners'),
            box_color=(0, 255, 0),
            point_color=(0, 220, 255),
            pose=render_pose,
        )
        overlay = _fp_draw_projected_geometry(
            overlay,
            pose_data,
            sample_id,
            sample_points=vis_geometry.get('sample_points'),
            bbox_corners=vis_geometry.get('bbox_corners'),
            box_color=(0, 255, 0),
            point_color=(0, 220, 255),
            pose=render_pose,
        )

    stage_panel = _fp_stack_cols([
        _fp_add_title_bar(observed_rgb, 'Observed'),
        _fp_add_title_bar(render_rgb, 'Render'),
        _fp_add_title_bar(overlay, 'Overlay'),
    ], pad=2)
    stage_panel = _fp_add_legend_bar(stage_panel, [
        ((0, 165, 255), orange_label),
        ((0, 255, 0), green_label),
        ((255, 0, 255), 'Render contour'),
        ((0, 220, 255), 'Vertices'),
    ])
    return _fp_add_title_bar(stage_panel, tile_title, height=34)


def _fp_build_refine_tiles_by_group(pose_data, stage_title, max_items=4, vis_geometry=None):
    sample_count = min(int(max_items), int(pose_data.rgbAs.shape[0]))
    tiles_by_group = []
    for sample_id in range(sample_count):
        tiles_by_group.append(_fp_make_pose_pair_tile(
            pose_data,
            sample_id,
            _fp_refine_tile_title(stage_title, sample_id),
            vis_geometry=vis_geometry,
            orange_mode='obs_box',
            orange_label='Obs 2D box',
            green_label='Refined box',
        ))
    return tiles_by_group


def _fp_build_score_tiles_by_group(pose_data, score_logits, best_ids, max_groups=4, topk=5, vis_geometry=None):
    if score_logits is None:
        return None
    score_logits_np = score_logits.detach().cpu().numpy()
    if score_logits_np.ndim == 1:
        score_logits_np = score_logits_np.reshape(-1, 1)
    best_ids_np = best_ids.detach().cpu().numpy().reshape(-1)
    max_groups = min(int(max_groups), int(score_logits_np.shape[0]))
    topk = min(int(topk), int(score_logits_np.shape[1]))
    tiles_by_group = []
    candidate_count = int(score_logits_np.shape[1])
    for group_id in range(max_groups):
        group_tiles = []
        order = np.argsort(-score_logits_np[group_id])
        for rank_id, cand_id in enumerate(order[:topk]):
            sample_id = group_id * candidate_count + int(cand_id)
            tile_title = f'score {rank_id} {score_logits_np[group_id, cand_id]:.2f}'
            if int(best_ids_np[group_id]) == int(cand_id):
                tile_title += ' selected'
            group_tiles.append(_fp_make_pose_pair_tile(
                pose_data,
                sample_id,
                tile_title,
                vis_geometry=vis_geometry,
                orange_mode='obs_box',
                orange_label='Obs mask box',
                green_label='Candidate box',
            ))
        tiles_by_group.append(group_tiles)
    return tiles_by_group


def _fp_build_grouped_stage_score_vis(refine_stage_tiles, score_tiles_by_group):
    row_count = 0
    for stage_tiles in refine_stage_tiles:
        if stage_tiles is not None:
            row_count = max(row_count, len(stage_tiles))
    if score_tiles_by_group is not None:
        row_count = max(row_count, len(score_tiles_by_group))
    if row_count == 0:
        return None

    row_images = []
    for group_id in range(row_count):
        row_tiles = []
        for stage_tiles in refine_stage_tiles:
            if stage_tiles is not None and group_id < len(stage_tiles):
                row_tiles.append(stage_tiles[group_id])
        if score_tiles_by_group is not None and group_id < len(score_tiles_by_group):
            row_tiles.extend(score_tiles_by_group[group_id])
        if len(row_tiles) == 0:
            continue
        row_canvas = _fp_stack_cols(row_tiles, pad=6)
        row_images.append(_fp_add_title_bar(row_canvas, f'group {group_id}', height=36))

    if len(row_images) == 0:
        return None
    return _fp_stack_rows(row_images, pad=8)



class FoundationPoseRefiner:
    '''
    ---
    ---
    Project wrapper around the FoundationPose refinement network.
    ---
    ---
    The class keeps preprocessing, optional ONNX/TensorRT acceleration and the
    iterative pose update logic used by the current project.
    ---
    ---
    当前项目中对 FoundationPose refinement 网络的封装类。
    ---
    ---
    该类维护预处理、可选 ONNX/TensorRT 加速，以及当前项目使用的迭代位姿更新逻辑。
    '''
    def __init__(self, device='cuda:0', rot_normalizer=0.35):
        '''
        ---
        ---
        Initialize the FoundationPose refiner wrapper.
        ---
        ---
        初始化 FoundationPose refiner 封装。
        '''
        self.device = device
        self.trans_normalizer = [1.0, 1.0, 1.0]
        self.trans_normalizer = torch.as_tensor(list(self.trans_normalizer), device=self.device, dtype=torch.float).reshape(1,3)
        self.rot_normalizer = rot_normalizer
        self.crop_ratio = 1.2
        self.dataset = PoseRefinePairH5Dataset()
        self.model = RefineNet(c_in=6).to(device)
        self.acceleration = 'pytorch'
        self.runtime = None

    def configure_acceleration(self, acceleration='pytorch', checkpoint_path=None, cache_dir=None):
        '''
        ---
        ---
        Configure the runtime backend for the scorer.
        ---
        ---
        为 scorer 配置运行后端。
        '''
        self.acceleration = 'pytorch' if acceleration is None else str(acceleration).lower()
        if self.acceleration not in ['pytorch', 'onnx', 'tensorrt']:
            raise ValueError('Unsupported FoundationPose refiner backend: %s' % acceleration)
        self.runtime = None
        if self.acceleration == 'pytorch':
            return
        if checkpoint_path is None:
            raise ValueError('FoundationPose refiner acceleration requires a checkpoint path.')
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(checkpoint_path), f'{self.acceleration}_cache', 'refiner')
        from Refinement.foundationpose_acceleration import FoundationPoseRefineRunner
        self.runtime = FoundationPoseRefineRunner(
            self.model,
            checkpoint_path,
            cache_dir,
            device=self.device,
            provider=self.acceleration,
        )

    @torch.inference_mode()
    def predict(self, ob_mask, rgb, depth, K, ob_in_cams, xyz_map, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5, vis_title=None, vis_max_items=4):
        '''
        ---
        ---
        Refine a batch of object poses with the FoundationPose refiner network.
        ---
        ---
        The method builds paired crops, runs the refinement network iteratively,
        applies egocentric pose deltas, and optionally produces stage-wise
        visualization tiles.

        Args:
            - ob_mask: Object masks for each pose hypothesis.
            - rgb: Input RGB image.
            - depth: Depth map.
            - K: Camera intrinsics.
            - ob_in_cams: Pose hypotheses in camera coordinates.
            - xyz_map: Precomputed xyz map.
            - get_vis: Whether to build visualization tiles.
            - mesh: Mesh object used for rendering.
            - mesh_tensors: Cached mesh tensors.
            - glctx: Shared rasterization context.
            - mesh_diameter: Object diameter.
            - iteration: Number of refine iterations.
            - vis_title: Optional visualization title.
            - vis_max_items: Maximum number of items to visualize.

        Returns:
            - B_in_cams_out: Refined pose tensor.
            - vis_tiles_by_group: Optional visualization groups.
        ---
        ---
        使用 FoundationPose refiner 网络微调一批物体位姿。
        ---
        ---
        该函数会构造成对裁剪数据，迭代执行 refinement 网络，施加自体坐标系下的
        位姿残差，并在需要时生成分阶段可视化 tile。

        参数:
            - ob_mask: 每个位姿假设对应的物体掩膜。
            - rgb: 输入 RGB 图像。
            - depth: 深度图。
            - K: 相机内参。
            - ob_in_cams: 相机坐标系下的位姿假设。
            - xyz_map: 预计算 xyz map。
            - get_vis: 是否生成可视化 tile。
            - mesh: 用于渲染的 mesh 对象。
            - mesh_tensors: 缓存后的 mesh 张量。
            - glctx: 共享光栅化上下文。
            - mesh_diameter: 物体直径。
            - iteration: 微调迭代轮数。
            - vis_title: 可选的可视化标题。
            - vis_max_items: 最多显示多少个条目。

        返回:
            - B_in_cams_out: 微调后的位姿张量。
            - vis_tiles_by_group: 可选的分组可视化结果。
        '''
        tf_to_center = np.eye(4)
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh
        crop_ratio = self.crop_ratio
        bs = 128
        B_in_cams = torch.as_tensor(ob_centered_in_cams, device=self.device, dtype=torch.float)
        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh_centered)
        if glctx is None:
            glctx = dr.RasterizeCudaContext(self.device)
        rgb_tensor = torch.as_tensor(rgb, device=self.device, dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device=self.device, dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device=self.device, dtype=torch.float)
        vis_tiles_by_group = None
        for iter_id in range(iteration):
            pose_data = make_crop_data_batch_train(ob_mask, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, xyz_map=xyz_map_tensor, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, device=self.device)
            vis_geometry = _fp_make_vis_geometry(mesh_centered) if get_vis else None
            if get_vis:
                stage_title = vis_title if vis_title is not None else 'Refine'
                if iteration > 1:
                    stage_title = f'{stage_title} iter {iter_id + 1}'
            B_in_cams = []
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat([pose_data.rgbAs[b:b+bs].to(self.device), pose_data.xyz_mapAs[b:b+bs].to(self.device)], dim=1).float()
                B = torch.cat([pose_data.rgbBs[b:b+bs].to(self.device), pose_data.xyz_mapBs[b:b+bs].to(self.device)], dim=1).float()
                if self.runtime is None:
                    output = self.model(A,B)
                else:
                    output = self.runtime.forward(A, B)
                for k in output:
                    output[k] = output[k].float()
                trans_delta = output['trans']*self.trans_normalizer
                rot_mat_delta = torch.tanh(output['rot'])*self.rot_normalizer
                rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
                trans_delta *= (mesh_diameter/2)
                B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
                identity_matrix = torch.eye(4, dtype=B_in_cam.dtype, device=B_in_cam.device)
                has_nan_inf = torch.isnan(B_in_cam).any(dim=(1, 2)) | torch.isinf(B_in_cam).any(dim=(1, 2))
                B_in_cam[has_nan_inf] = identity_matrix
                B_in_cams.append(B_in_cam)

            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams),4,4)
            if get_vis:
                pose_data_vis = make_crop_data_batch_train(
                    ob_mask,
                    B_in_cams,
                    mesh_centered,
                    rgb_tensor,
                    depth_tensor,
                    K,
                    crop_ratio=crop_ratio,
                    xyz_map=xyz_map_tensor,
                    glctx=glctx,
                    mesh_tensors=mesh_tensors,
                    dataset=self.dataset,
                    mesh_diameter=mesh_diameter,
                    device=self.device,
                )
                vis_tiles_by_group = _fp_build_refine_tiles_by_group(
                    pose_data_vis,
                    stage_title,
                    max_items=max(1, int(vis_max_items)),
                    vis_geometry=vis_geometry,
                )

        B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device=self.device, dtype=torch.float)
        return B_in_cams_out, vis_tiles_by_group



class FoundationPoseScorePredictor:
    '''
    ---
    ---
    Project wrapper around the FoundationPose scoring network.
    ---
    ---
    The scorer evaluates multiple pose candidates and selects the strongest one
    while sharing the same preprocessing and optional acceleration interface as
    the refiner wrapper.
    ---
    ---
    当前项目中对 FoundationPose 打分网络的封装类。
    ---
    ---
    scorer 会评估多个候选位姿并选出最优项，同时与 refiner 封装共享相似的
    预处理与可选加速接口。
    '''

    def __init__(self, device='cuda:0'):
        '''
        ---
        ---
        Initialize the FoundationPose scorer wrapper.
        ---
        ---
        初始化 FoundationPose scorer 封装。
        '''
        self.device = device
        self.crop_ratio = 1.1
        self.dataset = PoseScorePairH5Dataset()
        self.model = ScoreNet(c_in=6).to(device)
        self.acceleration = 'pytorch'
        self.runtime = None

    def configure_acceleration(self, acceleration='pytorch', checkpoint_path=None, cache_dir=None):
        self.acceleration = 'pytorch' if acceleration is None else str(acceleration).lower()
        if self.acceleration not in ['pytorch', 'onnx', 'tensorrt']:
            raise ValueError('Unsupported FoundationPose scorer backend: %s' % acceleration)
        self.runtime = None
        if self.acceleration == 'pytorch':
            return
        if checkpoint_path is None:
            raise ValueError('FoundationPose scorer acceleration requires a checkpoint path.')
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(checkpoint_path), f'{self.acceleration}_cache', 'scorer')
        from Refinement.foundationpose_acceleration import FoundationPoseScoreRunner
        self.runtime = FoundationPoseScoreRunner(
            self.model,
            checkpoint_path,
            cache_dir,
            device=self.device,
            provider=self.acceleration,
        )

    @torch.inference_mode()
    def predict(self, ob_mask, rgb, depth, K, ob_in_cams, xyz_map, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, L=None, vis_max_items=4, vis_score_topk=5):
        '''
        ---
        ---
        Score candidate poses and select the best ones.
        ---
        ---
        The scorer builds paired crops for all candidate poses, evaluates them
        with the scoring network, and optionally prepares grouped visualization
        tiles for the top candidates.

        Args:
            - ob_mask: Object masks for candidate poses.
            - rgb: Input RGB image.
            - depth: Depth map.
            - K: Camera intrinsics.
            - ob_in_cams: Candidate poses in camera coordinates.
            - xyz_map: Precomputed xyz map.
            - get_vis: Whether to build score visualization.
            - mesh: Mesh object used for rendering.
            - mesh_tensors: Cached mesh tensors.
            - glctx: Shared rasterization context.
            - mesh_diameter: Object diameter.
            - L: Candidate count per group.
            - vis_max_items: Maximum number of groups to visualize.
            - vis_score_topk: Maximum number of candidates shown per group.

        Returns:
            - ids: Selected candidate indices.
            - scores: Selected candidate scores.
            - vis_tiles_by_group: Optional grouped visualization tiles.
        ---
        ---
        对候选位姿进行打分并选出最佳项。
        ---
        ---
        该函数会为所有候选位姿构造成对裁剪数据，通过打分网络进行评估，
        并在需要时为各组 top 候选生成可视化 tile。

        参数:
            - ob_mask: 候选位姿对应的物体掩膜。
            - rgb: 输入 RGB 图像。
            - depth: 深度图。
            - K: 相机内参。
            - ob_in_cams: 相机坐标系下的候选位姿。
            - xyz_map: 预计算 xyz map。
            - get_vis: 是否生成打分可视化。
            - mesh: 用于渲染的 mesh 对象。
            - mesh_tensors: 缓存后的 mesh 张量。
            - glctx: 共享光栅化上下文。
            - mesh_diameter: 物体直径。
            - L: 每组候选数量。
            - vis_max_items: 最多显示多少组。
            - vis_score_topk: 每组最多显示多少个候选。

        返回:
            - ids: 选中候选的索引。
            - scores: 选中候选的得分。
            - vis_tiles_by_group: 可选的分组可视化结果。
        '''
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh
        crop_ratio = self.crop_ratio
        B_in_cams = torch.as_tensor(ob_centered_in_cams, device=self.device, dtype=torch.float)

        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh_centered)
        ob_mask = torch.as_tensor(ob_mask, device=self.device, dtype=torch.float)
        rgb_tensor = torch.as_tensor(rgb, device=self.device, dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device=self.device, dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device=self.device, dtype=torch.float)

        pose_data = make_crop_data_batch_train_score(ob_mask, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, xyz_map=xyz_map_tensor, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, device=self.device)

        def find_best_among_pairs(pose_data:BatchPoseData):
            ids = []
            scores = []
            score_logits_l = []
            if L is None:
                bs = 512
            else:
                bs = int(512 / L) * L

            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat([pose_data.rgbAs[b:b+bs].to(self.device), pose_data.xyz_mapAs[b:b+bs].to(self.device)], dim=1).float()
                B = torch.cat([pose_data.rgbBs[b:b+bs].to(self.device), pose_data.xyz_mapBs[b:b+bs].to(self.device)], dim=1).float()

                current_L = len(A) if L is None else L
                if self.runtime is None:
                    output = self.model(A, B, L=current_L)
                else:
                    output = self.runtime.forward(A, B, L=current_L)
                scores_cur = output['score_logit'].float()
                if get_vis:
                    score_logits_l.append(scores_cur.detach().cpu())

                scores_cur_v, scores_cur_id = torch.max(scores_cur, dim=1)
                ids.append(scores_cur_id)
                scores.append(scores_cur_v)
            ids = torch.cat(ids, dim=0).reshape(-1)
            scores = torch.cat(scores, dim=0).reshape(-1)
            score_logits = None
            if get_vis and len(score_logits_l) > 0:
                score_logits = torch.cat(score_logits_l, dim=0)
            return ids, scores, score_logits

        pose_data_iter = pose_data
        ids, scores, score_logits = find_best_among_pairs(pose_data_iter)
        vis_tiles_by_group = None
        if get_vis:
            vis_tiles_by_group = _fp_build_score_tiles_by_group(
                pose_data_iter,
                score_logits,
                ids,
                max_groups=max(1, int(vis_max_items)),
                topk=max(1, int(vis_score_topk)),
                vis_geometry=_fp_make_vis_geometry(mesh_centered),
            )
        scores = scores + 100
        return scores.detach().clone(), ids.detach().clone(), vis_tiles_by_group


def make_mesh_tensors(mesh, device='cuda:0', max_tex_size=None, uv = None):
    '''
    ---
    ---
    Convert a trimesh mesh into the tensor bundle used by nvdiffrast rendering.
    ---
    ---
    将 trimesh mesh 转换为 nvdiffrast 渲染所需的张量包。
    '''
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert('RGB'))
        img = img[...,:3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size>max_tex_size:
                scale = 1/max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None]/255.0
        mesh_tensors['uv_idx']  = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
        if uv is not None:
            uv = torch.as_tensor(uv, device=device, dtype=torch.float)
        else:
            uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:,1] = 1 - uv[:,1]
        mesh_tensors['uv']  = uv
    else:
        mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0
    mesh_tensors.update({
        'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
        'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
        'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
    })
    return mesh_tensors
