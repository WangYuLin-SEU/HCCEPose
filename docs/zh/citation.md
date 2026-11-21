# 引用与许可

## 论文

[HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](https://arxiv.org/abs/2510.10177)，ICCV 2025。

## 来源与许可

项目许可证见 [LICENSE](../../LICENSE)。公开代码和素材来源见[来源清单（Markdown）](../../PUBLIC_SOURCES.md)；WAPR 接口来源于 [WAPR](https://github.com/WangYuLin-SEU/WAPR)。第三方组件遵循各自的许可证，并保留原作者声明。

默认 RGB 检测器使用 Ultralytics。部署前请按实际用途核对 [Ultralytics 许可](https://www.ultralytics.com/license)，并确认所用检测器和位姿权重的授权范围。WAPR 的四份第一方权重遵循单独的[权重许可](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/WEIGHTS_LICENSE.txt)。

网站三维示意使用 Three.js，按 MIT 许可证分发。

## BibTeX 引用

```bibtex
@InProceedings{HccePose_BF,
    author    = {Wang, Yulin and Hu, Mengting and Li, Hongli and Luo, Chen},
    title     = {HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {7166-7175}
}
```

## 致谢

- [BOP](https://bop.felk.cvut.cz/) 与 [BOP toolkit](https://github.com/thodan/bop_toolkit) 提供数据格式、基准与评测工具。
- [BlenderProc](https://github.com/DLR-RM/BlenderProc) 用于训练场景渲染，[KASAL](https://pypi.org/project/kasal-6d/) 用于旋转对称信息。
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 提供检测框。
- 可选 RGB-D 接入使用公开 [WAPR](https://github.com/WangYuLin-SEU/WAPR) 和 [MegaPose](https://github.com/megapose6d/megapose6d)。
