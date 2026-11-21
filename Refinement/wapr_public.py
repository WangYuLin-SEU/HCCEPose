"""Connect aligned instance masks to the published WAPR API.
将已对齐的实例掩码送入公开 WAPR 接口。

Source contract / 来源约定:
https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/wapr/estimator.py
"""
import numpy as np

from HccePose.image_geometry import inference_image_geometry


def estimate_from_masks(estimator, image_bgr, depth_m, cam_K, instances):
    """Return ordered object-to-camera poses (translation in meters) and scores for this input.
    按输入顺序返回本次输入的物体到相机位姿（平移单位为米）及评分。

    Each instance contains mesh (meter vertices), diameter_m and a full-frame mask.
    每个实例含米制 mesh、diameter_m 和原图大小的 mask。
    This function does not define acceptance, duplicate suppression or seed refinement.
    本接口不定义接收阈值、位姿去重或初值精修。
    """
    image_bgr = np.asarray(image_bgr)
    depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
    cam_K = np.ascontiguousarray(cam_K, dtype=np.float32)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError('image_bgr must be uint8 H x W x 3')
    if depth_m.shape != image_bgr.shape[:2] or not np.isfinite(depth_m).all() or (depth_m < 0).any():
        raise ValueError('depth_m must match the image and contain finite, nonnegative meters')
    if cam_K.shape != (3, 3) or not np.isfinite(cam_K).all() or cam_K[0, 0] <= 0 or cam_K[1, 1] <= 0:
        raise ValueError('cam_K must contain finite 3 x 3 pixel intrinsics with positive focal lengths')
    prepared = []
    for item in instances:
        mask = np.asarray(item['mask'])
        diameter_m = float(item['diameter_m'])
        if mask.shape != depth_m.shape or not np.isfinite(mask).all():
            raise ValueError('Each mask must be finite and match the full input frame')
        if not np.isfinite(diameter_m) or diameter_m <= 0:
            raise ValueError('diameter_m must be finite and positive')
        foreground = mask > 0
        if not foreground.any() or not np.any(foreground & (depth_m > 0)):
            raise ValueError('Each mask needs foreground and positive observed depth')
        prepared.append({'mesh': item['mesh'], 'diameter_m': diameter_m, 'mask': foreground})
    if not prepared:
        return []
    # HccePose uses BGR; the published WAPR contract expects RGB, meters and pixels.
    # HccePose 使用 BGR；公开 WAPR 接口使用 RGB、米和像素内参。
    image_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    results = estimator.estimate_many_categories_many_instances(image_rgb, depth_m, cam_K, prepared)
    if len(results) != len(prepared):
        raise RuntimeError('WAPR result count differs from instance count')
    for result in results:
        pose = np.asarray(result['pose_4x4'])
        score = float(result['score_6d'])
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(score):
            raise RuntimeError('WAPR returned an invalid pose or score')
    return results


def restore_full_frame_masks(prediction, image_shape, mask_threshold=0.5):
    """Undo the published Tester's crop and image transform; return boolean masks.
    还原公开 Tester 的裁剪及图像变换，返回原图大小的布尔掩码。

    mask_threshold is a foreground threshold, not a WAPR score threshold.
    mask_threshold 是掩码前景阈值，不是 WAPR 评分阈值。
    """
    import cv2
    import kornia
    import torch

    height, width = image_shape[:2]
    if height <= 0 or width <= 0 or not np.isfinite(mask_threshold) or not 0 <= mask_threshold <= 1:
        raise ValueError('Invalid image shape or foreground threshold')
    # Use the same resized frame and border offsets as the detector.
    _, scaled_h, scaled_w, padded_h, padded_w, top, left = inference_image_geometry(height, width)
    masks = prediction['pred_mask']
    transforms = prediction['Detect_Bbox_tfs_128']
    if not isinstance(masks, torch.Tensor) or masks.ndim != 3 or min(masks.shape[1:]) < 1:
        raise ValueError('pred_mask must be a tensor of shape [N,H,W] with nonempty crops')
    if not isinstance(transforms, torch.Tensor) or transforms.shape != (len(masks), 3, 3):
        raise ValueError('Expected one [3,3] crop transform per mask, in the same instance order')
    if len(masks) == 0:
        return np.zeros((0, height, width), dtype=bool)
    if not torch.isfinite(masks).all() or not torch.isfinite(transforms).all():
        raise ValueError('Crop masks and transforms must be finite')
    transforms = transforms.to(dtype=torch.float32)
    full_tensor = kornia.geometry.transform.warp_perspective(
        masks[:, None].float(), torch.linalg.inv(transforms),
        dsize=(padded_h, padded_w), mode='nearest', align_corners=False,
    )[:, 0].detach().cpu()
    full = np.array(full_tensor.numpy(), dtype=np.float32, copy=True)
    output = []
    for mask in full:
        unpadded = mask[top:top + scaled_h, left:left + scaled_w]
        restored = cv2.resize(unpadded, (width, height), interpolation=cv2.INTER_NEAREST)
        output.append(restored > mask_threshold)
    return np.asarray(output)
