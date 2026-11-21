# Refinement and depth comparison

Compare RGB, MegaPose RGB-D and WAPR with the same images, models, calibration and unit conventions. First run the [RGB](rgb-inference.md), [MegaPose](rgbd-megapose.md) and [WAPR](rgbd-wapr.md) tutorials separately, then preserve each branch’s outputs.

## Preserve results for the same frame

```python
# Prepare tester, image_bgr, cam_K and meter depth_m using the preceding tutorials.
rgb_results = tester.predict(cam_K, image_bgr, [1])
megapose_results = tester.predict(cam_K, image_bgr, [1], depth=depth_m, use_megapose=True)
```

WAPR handles this frame through the [separate mask API](refinement-api.md), without overwriting those two dictionaries. Retain object IDs and instance associations across branches; ranking or screen position alone does not establish instance identity.

Save `show_6D_vis1`, `show_6D_vis2` and MegaPose debug views to inspect projections. For numerical comparisons, check initial PnP success and valid refinement indices before converting units: HccePose / MegaPose translations use millimeters; the WAPR example JSON uses meters.

## Back-project depth

Back-project registered meter depth `z` using pixel intrinsics:

```python
import numpy as np

height, width = depth_m.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))
valid = np.isfinite(depth_m) & (depth_m > 0)
z = depth_m[valid]
x = (u[valid] - cam_K[0, 2]) * z / cam_K[0, 0]
y = (v[valid] - cam_K[1, 2]) * z / cam_K[1, 1]
points_camera_m = np.column_stack([x, y, z])
```

Use original-image intrinsics, not the processed-canvas matrix. Depth must already be registered and corrected under that camera model. Invalid depth produces no points. Transform meter mesh vertices into camera space using meter poses; do not apply the object pose again to observed camera-space points.

## Layers and views

The point cloud represents observed surfaces; a mesh represents the model placed at an estimated pose. Distinguish them using separate colors and point versus surface rendering. After orbiting to inspect occluded regions, restore camera alignment. See the [WAPR RGB-D viewer](rgbd-wapr.md#linked-rgb-and-3d-inspection) for interaction examples.

Symmetric objects may have equivalent orientations, and missing depth alone does not establish a pose error. Projection and depth checks inspect geometry. Accuracy conclusions require independent ground truth and appropriate metrics.

## Public results and evaluation

See public [MegaPose](rgbd-megapose.md) and [RGB](rgb-inference.md) examples, and historical [BOP evaluation](bop-evaluation.md). Backend scores have different meanings and cannot be directly subtracted to rank accuracy. Record first-time setup and repeated inference timings separately.
