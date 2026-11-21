# 位姿精修与深度对比

RGB、MegaPose RGB-D 与 WAPR 的结果应在同一套图像、模型、相机与单位约定下比较。先跑通 [RGB](rgb-inference.md)、[MegaPose](rgbd-megapose.md) 和 [WAPR](rgbd-wapr.md) 各自教程，再保存对应输出。

## 同一帧，分别保存各分支

```python
# tester、image_bgr、cam_K 和米制 depth_m 已按前面的教程准备。
rgb_results = tester.predict(cam_K, image_bgr, [1])
megapose_results = tester.predict(cam_K, image_bgr, [1], depth=depth_m, use_megapose=True)
```

WAPR 通过[独立掩码接口](refinement-api.md)处理该帧，不覆盖上面两个结果字典。对照时保留物体编号及各分支实例对应关系；不能仅凭排名或画面位置把不同检测视为同一实例。

保存 `show_6D_vis1`、`show_6D_vis2` 及 MegaPose 调试图可检查投影。需要数值对照时先检查 PnP 成功标记和精修有效索引，再统一平移单位：HccePose / MegaPose 返回毫米，WAPR 示例 JSON 返回米。

## 深度反投影

在彩色图坐标系内，以米制深度 `z` 与相机内参反投影：

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

这里的 `cam_K` 对应原图，不能换用处理后画布内参；前提是深度已经按同一相机模型配准和校正。无效深度不生成点。对齐时把米制网格顶点用米制位姿变换到相机坐标系，不要给点云再重复乘一次位姿。

## 图层与视角

点云是观测表面，网格是根据估计位姿放置的模型。观察时可用不同颜色、散点与实面区分两者；旋转检查遮挡区域后，应能复位到相机对齐视角。交互操作形式见 [WAPR RGB-D 查看器](rgbd-wapr.md#rgb-与三维联动查看)。

对称物体可能有多种等价姿态；局部点云缺失也不自动说明位姿错误。投影、深度一致性用于检查几何，精度结论需要与独立真值对比并使用相应评测指标。

## 公开结果与评测

公开示例图见 [MegaPose](rgbd-megapose.md) 与 [RGB](rgb-inference.md)，历史公开基准见 [BOP 评测](bop-evaluation.md)。不同后端的评分定义不同，不能直接把其数值相减来判断谁更准确。首次准备耗时与重复推理耗时应分别记录。
