# 位姿精修与 RGB-D 接口

RGB 推理只需彩色图像与内参。引入深度时，先完成像素配准、单位检查，再选择 [WAPR](rgbd-wapr.md) 或 [MegaPose](rgbd-megapose.md)。它们的入口、初值要求和评分含义不同。

## 输入与单位

| 项目 | HccePose / MegaPose 接入 | WAPR 接入 |
| --- | --- | --- |
| 彩色图 | OpenCV BGR `uint8 H×W×3` | 接入函数接收 BGR，内部转为公开 API 所需 RGB。 |
| 深度 | MegaPose RGB 可省略；RGB-D 为米制 `H×W` | 必需，有限、非负、米制 `H×W`。 |
| 相机 | 原图对应的像素内参 `3×3` | 与原图、深度和完整掩码一致的像素内参。 |
| 模型 | BOP 模型按毫米注册，内部转换 | 网格顶点与 `diameter_m` 都为米。 |
| 初始位姿 | MegaPose 从 HccePose 位姿精修 | `estimate` 不用初值；`hybrid` 逐实例可选；`refine` 必须提供。 |
| 输出平移 | `Tester` 结果保持毫米 | `Tester.Rts` 为毫米；原始 `pose_4x4` / `pose_4x4_m` 为米。 |

图像尺寸相同不代表已经配准。适配器检查尺寸，无法判断传感器外参或畸变是否已正确处理；这些需在输入前完成。

## 在 Tester 中注册并调用 WAPR

```python
# 可提前注册；predict 也会在首次使用时自动注册。
tester.register_wapr()
results = tester.predict(cam_K, image_bgr, [1], depth=depth_m,
                         use_wapr=True, wapr_mode='estimate', wbps_threshold=0.75,
                         wapr_nms_iou_threshold=0.5)
entry = results.get(1)
if entry is not None:
    print(entry['Rts'], entry['wapr_scores'], entry['wapr_instance_indices'])
raw_records = results['wapr_postprocess']['raw_records']
```

多帧复用同一个 `Tester`。三种 WAPR 模式均自动执行 WBPS 筛选及同类渲染掩码 NMS，掩码、裁剪图、置信度和位姿同步筛选。`conf` 仍为 YOLO 置信度。某物体全部被筛除时返回零行；无检测时返回空报告，不加载 WAPR 权重。同一次调用选择 WAPR 或 MegaPose。NMS 失败时预测报错，`tester.last_wapr_raw_records` 仍保存本帧原始估计；估计尚未完成时该属性为 `None`。

| `wapr_mode` | 初始化与评分 |
| --- | --- |
| `estimate`（默认） | 跳过 PnP，沿用公开的多实例掩码估计及候选组。 |
| `hybrid` | 先求解 PnP，将有效初值加入该实例的 WAPR 旋转和深度候选，一起精修，并对完整候选组统一评分。初值失败或无效时，沿用公开掩码估计路径。 |
| `refine` | 先求解 PnP，仅对有效初值执行无掩码 WAPR 和 SAPR 精修，再对原位姿与精修结果评分。不生成全局旋转候选。PnP 失败或无效的实例列入 `skipped_instances`，不伪造位姿或零分。 |

固定公开 recipe 下，候选估计使用 12 项，有 PnP 初值的混合评分组为 13 项，精修评分组为 2 项。原位姿和精修结果是同一初值的真实处理阶段，不是为了 WBPS 复制补齐的位姿。`within_group` 选择位姿；`score_6d` 沿用完整实例组的 `between_group` 最大值计算。不合并两路各自评分来选胜者。候选组组成改变可能影响评分，显式的 `0.75` 门槛应按模式核对。

有初值的模式组织[固定公开版的精修与 WBPS 组件](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/wapr/estimator.py)，逐评分组处理；默认估计模式保持公开的跨实例批量路径。recipe 的迭代数、裁剪比例、旋转限幅及后端选择不变。不支持的候选组长度或引擎配置会报错，不静默切换后端，也不补齐评分组。PnP 使用所选 `pnp_op`，组合求解器仍要求匹配的 checkpoint 先验。

## 将裁剪掩码还原至原图

```python
from Refinement.refinement_group import restore_full_frame_masks

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
from Refinement.refinement_group import estimate_from_masks

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

多帧应用应复用已初始化的 HccePose 和 WAPR 对象。`Refinement.refinement_group` 同时提供 `Refinement_MP`、`Refinement_WAPR` 及共用 RGB-D 读取和掩码还原。创建 `Refinement_WAPR` 不加载位姿权重，第一次非空推理才构造公开后端，此时可能准备运行资源。离线前按[安装说明](installation.md)准备对应缓存。


`Tester` 内部统一实现评分筛选与渲染掩码 NMS，`predict` 自动执行；静态方法 `postprocess_wapr` 与 `postprocess_saved_wapr` 可重新筛选已有记录，无需加载推理权重。底层的 `estimate_from_masks` 仍返回原始掩码估计。`Refinement_WAPR.inference_batch(..., mode='estimate')` 也支持 `hybrid` 与 `refine`，有初值的模式在实例字典中传入原始物体系下、米制的 `pose_4x4_m`。refinement group 只封装后端估计和精修，返回原始结果。详见[后处理教程](wapr-postprocessing.md)。
