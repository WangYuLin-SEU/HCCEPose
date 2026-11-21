# Citation and licenses

## Paper

[HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](https://arxiv.org/abs/2510.10177), ICCV 2025.

## Sources and licenses

See the [project license](../../LICENSE) and the [source list (Markdown)](../../PUBLIC_SOURCES.md) for published code and media sources. The WAPR interface comes from [WAPR](https://github.com/WangYuLin-SEU/WAPR). Third-party components retain their own licenses and author notices.

The default RGB detector uses Ultralytics. Check the [Ultralytics licensing options](https://www.ultralytics.com/license) for your intended use and the terms for the checkpoints you obtain separately. The four first-party WAPR checkpoints have their own [weight license](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/WEIGHTS_LICENSE.txt).

The 3D illustration uses Three.js under MIT.

## BibTeX citation

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

## Acknowledgments

- [BOP](https://bop.felk.cvut.cz/) and [BOP toolkit](https://github.com/thodan/bop_toolkit) provide formats, benchmarks and evaluation tools.
- [BlenderProc](https://github.com/DLR-RM/BlenderProc) supports training-scene rendering; [KASAL](https://pypi.org/project/kasal-6d/) describes rotational symmetries.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) provides detections.
- Optional RGB-D integration uses public [WAPR](https://github.com/WangYuLin-SEU/WAPR) and [MegaPose](https://github.com/megapose6d/megapose6d).
