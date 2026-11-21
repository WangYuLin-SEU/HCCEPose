# 位姿精修与 RGB-D 接口

RGB 推理只需彩色图像与内参。引入深度时，先完成像素配准、单位检查，再选择 [WAPR](rgbd-wapr.md) 或 [MegaPose](rgbd-megapose.md)。它们的入口、初值要求和评分含义不同。

## 输入与单位

| 项目 | HccePose / MegaPose 接入 | WAPR 掩码接入 |
| --- | --- | --- |
| 彩色图 | OpenCV BGR `uint8 H×W×3` | 接入函数接收 BGR，内部转为公开 API 所需 RGB。 |
| 深度 | MegaPose RGB 可省略；RGB-D 为米制 `H×W` | 必需，有限、非负、米制 `H×W`。 |
| 相机 | 原图对应的像素内参 `3×3` | 与原图、深度和完整掩码一致的像素内参。 |
| 模型 | BOP 模型按毫米注册，内部转换 | 网格顶点与 `diameter_m` 都为米。 |
| 初始位姿 | MegaPose 从 HccePose 位姿精修 | 当前掩码估计入口不接收 HccePose 位姿初值。 |
| 输出平移 | `Tester` 结果保持毫米 | `pose_4x4` / 示例 `pose_4x4_m` 为米。 |

图像尺寸相同不代表已经配准。适配器检查尺寸，无法判断传感器外参或畸变是否已正确处理；这些需在输入前完成。

## 将裁剪掩码还原至原图

```python
from Refinement.wapr_public import restore_full_frame_masks

entry = predictions[obj_id]
masks = restore_full_frame_masks(entry, image_bgr.shape)
```

返回布尔数组 `N×H×W`，顺序与该物体的裁剪掩码一致。函数先逆变换到缩放补边画布，再去掉边框，恢复原图尺寸。不要直接把 `Detect_Bbox_tfs_128` 应用到原图，也不要把 `128×128` 裁剪掩码交给整图深度估计。

可选的 `mask_threshold=0.5` 仅是掩码前景阈值，不是 WAPR 评分阈值。掩码数量与变换数量不符时明确报错。

## 调用公开 WAPR 接口

```python
import numpy as np
import trimesh
from wapr.estimator import WAPREstimator
from Refinement.wapr_public import estimate_from_masks

# 已准备同帧 image_bgr、depth_m、cam_K、masks，以及毫米模型与直径。
mesh = trimesh.load(mesh_path, force='mesh', process=False)
mesh.apply_scale(1e-3)
diameter_m = diameter_mm * 1e-3
instances = [
    {'mesh': mesh, 'diameter_m': diameter_m, 'mask': mask}
    for mask in masks
]
estimator = WAPREstimator(device='cuda:0')
results = estimate_from_masks(estimator, image_bgr, depth_m, cam_K, instances)
for result in results:
    pose_m = np.asarray(result['pose_4x4'])
    score = float(result['score_6d'])
    print(pose_m, score)
```

不同物体的实例可使用不同网格和直径。输入列表每项代表一个实例；输出与该列表一一对应。只需估计已有掩码时无需创建 HccePose `Tester`，可直接准备自己的完整图像掩码。

适配器检查深度、内参、掩码尺寸及正深度，验证输出数量、位姿形状与有限分数。空列表返回 `[]`，不调用后端；空掩码或区域内没有正深度会报错，而非伪造零分。

当前 `score_6d` 针对本次输入计算，不是概率或物体标签。估计成功、分数通过与确认目标是不同的判断，接收、去重和排序由应用定义。完整文件入口见 [WAPR 教程](rgbd-wapr.md)。

## MegaPose 注册与调用

```python
tester.register_megapose()
results_rgb = tester.predict(cam_K, image_bgr, [1], depth=None, use_megapose=True)
results_rgbd = tester.predict(cam_K, image_bgr, [1], depth=depth_m, use_megapose=True)
```

不传深度选择 RGB 模型，传入配准米制深度选择 RGB-D。`register_megapose` 为加载器中的物体注册毫米网格；推理在独立子环境中执行。评分与有效实例索引见[结果字段](results-api.md)。

## 重复处理与资源准备

多帧应用应复用已初始化的 HccePose 和 WAPR 对象。导入 `Refinement.wapr_public` 本身不安装依赖、下载资源或初始化 GPU；主动创建公开后端时可能准备其运行资源。离线前按[安装说明](installation.md)准备对应缓存。


估计之后若需要应用层分数筛选与渲染掩码 NMS，可使用[可选后处理教程](wapr-postprocessing.md)，不改变适配器的返回约定。
