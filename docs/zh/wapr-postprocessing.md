# WAPR 评分筛选与渲染掩码 NMS

[WAPR 推理示例](rgbd-wapr.md)会保存各实例的估计位姿与评分。本教程提供可选的第二步：先按分数筛选，再调用[固定公开版 WAPR 的 NMS 实现](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/wapr/suppression.py)，移除投影高度重叠的位姿。不改变原适配器，也不重新运行位姿网络。

## 准备同一帧与模型

先运行 `examples/inference/predict_wapr.py`，保留其输出 `outputs/wapr.json`。后处理时使用同一张图像、原图内参和对应物体网格。每条记录须含 `object_id`、`instance_index`、`pose_4x4_m` 与 `score_6d`。

脚本从 `<dataset_root>/models/obj_XXXXXX.ply` 读取 BOP 网格。BOP 顶点使用毫米，`mesh_scale_to_m=1e-3` 将其换算为米；如果网格已经以米表示，将其设为 `1`。网格坐标与保存位姿须使用同一个原始物体坐标系，之后不要再将网格居中。

分数筛选在 CPU 上执行。有记录通过门槛时，渲染掩码 NMS 需要[已安装 WAPR](installation.md#可选-wapr)的 CUDA 渲染器，并使用同一 Python 环境。脚本不会重新加载位姿网络权重。缺少网格或渲染器出错时会报错，不会悄悄退回未筛选结果。

脚本可在公开源码中查看：[postprocess_wapr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/examples/inference/postprocess_wapr.py)。

## 运行可选脚本

`examples/inference/postprocess_wapr.py` 底部的可编辑参数对应[资源页下载的 RGB-D 示例](assets.md#下载一个-rgb-d-示例)。核对路径后，从项目根目录运行：

```bash
python examples/inference/postprocess_wapr.py
```

原始 JSON 保持不变，结果单独写入 `outputs/wapr_filtered.json`。在 Python 应用中也可调用：

```python
from examples.inference.postprocess_wapr import main

# 路径与位姿必须对应同一帧。
report = main(
    "data/demo-bin-picking",
    "data/test_imgs_RGBD/000003_rgb.png",
    "data/test_imgs_RGBD/000003_camera_matrix.json",
    "outputs/wapr.json",
    "outputs/wapr_filtered.json",
    score_threshold=0.75,
    nms_iou_threshold=0.5,
    device="cuda:0",
    mesh_scale_to_m=1e-3,
)
retained = report["retained_records"]
```

对于内存中的记录，也可调用 `filter_records(records, meshes_by_id, cam_K, image.shape, score_threshold, nms_iou_threshold, device)`。其中 `meshes_by_id` 将每个通过评分的 BOP 物体编号映射到其米制网格；记录格式与 `predict_wapr.py` 一致。

## 两个不同的阈值

| 参数 | 示例值 | 行为 |
| --- | --- | --- |
| `score_threshold` | `0.75` | 对返回的 `score_6d` 应用 WBPS 门槛，NMS 前先保留 `score_6d >= 0.75` 的记录。这是应用参数，不是对所有物体都校准过的概率门槛；大于 1 的评分原样保留。 |
| `nms_iou_threshold` | `0.5` | 仅在同一 `object_id` 内优先考虑高分位姿；渲染掩码 IoU 严格大于 0.5 时抑制排序靠后的位姿，等于门槛时保留。 |

IoU 示例值沿用[公开 WAPR 示例](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/examples/10_robi_zigzag.py)。两个值都作为脚本中的应用参数显式保留，不改变 WAPR 的科学默认值。应根据物体和相机调整。

## NMS 比较什么

对每个通过评分的位姿，WAPR 使用对应 CAD、估计的物体到相机变换及原图内参渲染物体剪影。NMS 比较的是这些随位姿变化的掩码，不只是原检测框或输入掩码。不同物体编号使用各自的网格，分组处理。

公开贪心实现对同分项优先保留输入较早的记录；被抑制项不会继续抑制后续项。输出沿用原输入顺序，不按分数重新排序。投影掩码为空的结果也会被后端移除。

高度重叠的投影也可能来自不同真实物体，遮挡或深度不同的场景尤其如此。剪影 IoU 本身不能确认物理身份或六维位姿精度；应检查这类情况，再按应用调整策略。本示例没有另外加入平移、旋转距离判据，也不进行跨类别抑制。

## 读取输出

| 字段 | 含义 |
| --- | --- |
| `raw_records` | 所有输入记录，保留原评分与实例标识。 |
| `decisions` | 逐输入的 `input_index`、物体与实例标识、评分和筛选状态。 |
| `kept_input_indices` | 指向原列表的下标，保持输入顺序。 |
| `retained_records` | 通过分数门槛且在渲染掩码 NMS 后保留的记录。 |
| `score_threshold`、`nms_iou_threshold`、`nms_method` | 本次输出采用的策略。 |

筛选状态为 `kept`、`below_score_threshold` 或 `not_retained_by_mask_nms`。最后一种同时包含空投影和重叠抑制，不能直接解释为已确认的重复物体。跳过项不会被伪造为零分。

## 与网站演示的关系

三角块查看器展示的是已保存结果。本脚本根据传入的 JSON 和模型计算筛选结果，不改写查看器，保留数量由当前输入和阈值决定。通过评分或 NMS 也不等于真值确认；比较策略时请保留原始文件。
