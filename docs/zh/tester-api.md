# Tester 接口

`HccePose.tester.Tester` 处理单张 OpenCV 图像或视频帧。检测器为已知物体产生区域，逐物体网络预测掩码及前后表面坐标，PnP 求解位姿。一个物体类别可以出现多个实例。

## 构造器

```python
from HccePose.tester import Tester

tester = Tester(
    bop_dataset_item,
    show_op=True,
    hccepose_vis=None,
    CUDA_DEVICE='0',
    top_K=None,
    crop_size=256,
    efficientnet_key=None,
    megapose_model_name_rgb='megapose-1.0-RGB-multi-hypothesis',
    megapose_model_name_rgbd='megapose-1.0-RGBD',
    hccepose_acceleration='pytorch',
    hccepose_acceleration_cache_dir=None,
)
```

| 参数 | 含义 |
| --- | --- |
| `bop_dataset_item` | [BOP 加载器](data-api.md)，提供模型、权重根目录及物体顺序。 |
| `hccepose_vis` | 非 `None` 时覆盖 `show_op`；控制四个基础可视化字段。 |
| `CUDA_DEVICE` | CUDA 编号字符串，如 `'0'`。当前推理流程需要 CUDA。 |
| `top_K` | 预留参数，不限制实际检测或位姿数量。 |
| `crop_size` | 默认 `256`；需与网络及可视化布局匹配，不用于更改原图分辨率。 |
| `efficientnet_key` | `None` 使用 ResNet 分支，其他值选择 EfficientNet；必须匹配权重。 |
| `megapose_model_name_rgb`、`megapose_model_name_rgbd` | MegaPose 注册时采用的模型名称。 |
| `hccepose_acceleration` | `'pytorch'`、`'onnx'` 或 `'tensorrt'`，仅选择网络执行后端。 |
| `hccepose_acceleration_cache_dir` | 导出及后端缓存目录；`None` 时使用各物体的默认目录。 |

兼容参数 `acceleration` 和 `acceleration_cache_dir` 仍可使用，非 `None` 时覆盖对应的 HccePose 专用参数。加速行为见[加速说明](acceleration.md)。

初始化会加载 `models_info.json` 中的全部物体，而不只加载之后请求预测的编号。每个物体都需有匹配的位姿权重。YOLO 类别索引通过加载器的 `obj_id_list` 映射到 BOP 物体编号，因此元数据顺序必须与检测器训练时一致。

即使初始化打印 CPU 回退信息，创建物体边界张量和加载权重仍有 CUDA 路径，不能据此认为支持完整 CPU 推理。损坏或不兼容的 checkpoint 会报告对应物体、路径及原因。

## 预测

```python
results = tester.predict(
    cam_K, image_bgr, object_ids,
    confidence_threshold=0.75,
    conf=0.75,
    iou=0.50,
    max_det=200,
    pnp_op='ransac+comb',
    depth=None,
    use_megapose=False,
    megapose_vis=False,
    megapose_vis_stages=None,
)
```

| 输入 | 约定 |
| --- | --- |
| `cam_K` | NumPy `3×3` 原图内参，单位为像素，必须对应当前输入尺寸。 |
| `image_bgr` | NumPy `H×W×3`、`uint8`，保持 OpenCV BGR 顺序。 |
| `object_ids` | 请求的 BOP 物体编号，不是 YOLO 类别索引；每个编号可返回多个实例。 |
| `conf` | 传给 YOLO 的检测置信度阈值。 |
| `confidence_threshold` | 仅控制 2D 检测图的绘制，不过滤返回的位姿。 |
| `iou`、`max_det` | 传给 YOLO 的检测去重参数和数量上限。 |
| `pnp_op` | `epnp`、`ransac`、`ransac+vvs`、`ransac+comb` 或 `ransac+vvs+comb`。 |
| `depth` | 可选的配准深度，`H×W`，单位为米；提供后可用于 MegaPose RGB-D。 |
| `use_megapose` | 请求 MegaPose 精修；未注册时在首次使用时注册。 |
| `megapose_vis`、`megapose_vis_stages` | 请求 MegaPose 调试图及阶段选择。 |

组合 PnP 模式需要 checkpoint 中的 `keypoints_` 先验；缺少时会在检测前报错，不会静默替换求解方式。输出中的 `pnp_success` 用于逐实例判断初始 PnP 是否成功。

最长边超过 640 像素的输入会先缩小，再按检测器 32 像素步长补边；内参和可选深度同步调整。二维结果字段以处理后画布为准。完整图像掩码须通过[还原函数](results-api.md)回到输入分辨率。

```python
# 模型只初始化一次；在图片列表或视频循环中重复调用。
results = tester.predict(cam_K, image_bgr, [1])
entry = results.get(1)
if entry is not None:
    valid = entry['pnp_success']
    poses_mm = entry['Rts'][valid]
    print(poses_mm[:, :3, 3])
```

没有检出的物体不会有对应整数键；字符串键保存计时和图像等信息。完整字段见[结果字典](results-api.md)。WAPR 通过独立的[掩码接入函数](refinement-api.md)调用。
