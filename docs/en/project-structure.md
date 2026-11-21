# Project structure

This project contains application inference, the public WAPR adapter and bilingual website sources. Public upstream training and rendering stage scripts are indexed below.

```text
HccePose/                 RGB inference, networks, BOP loader
s1_*.py ... s4_*.py       Public preparation, training and evaluation stages
Refinement/              WAPR adapter and optional MegaPose interface
examples/inference/      predict_image.py, predict_wapr.py, postprocess_wapr.py
docs/zh/                 Chinese documentation
docs/en/                 English documentation
docs/theme/              Shared visual theme
docs/assets/             Figures and presentation assets
pages/                   Generated static website
show_vis/                Public example figures
media/                   Public video preview
tests/                   API and documentation checks
```

## Public stage scripts

These files live in the [fixed public upstream root](installation.md#public-training-and-rendering-source). Follow the links for source; parameters are explicit in each script.

| Script | Purpose |
| --- | --- |
| [s1_p1_obj_rename_center.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s1_p1_obj_rename_center.py) | Center and name meshes |
| [s1_p2_obj_symmetry.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s1_p2_obj_symmetry.py) | Rotational symmetry |
| [s1_p3_obj_infos.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s1_p3_obj_infos.py) | BOP model metadata |
| [s2_p1_gen_pbr_data.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s2_p1_gen_pbr_data.py) | PBR rendering |
| [s3_p1_prepare_yolo_label.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s3_p1_prepare_yolo_label.py) | Detection labels |
| [s3_p2_train_yolo.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s3_p2_train_yolo.py) | Detector training |
| [s4_p1_gen_bf_labels.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p1_gen_bf_labels.py) | Front/back coordinate labels |
| [s4_p2_train_bf_pbr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_train_bf_pbr.py) | Per-object pose training |
| [s4_p2_test_bf_pbr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_test_bf_pbr.py) | Crop evaluation |
| [s4_p2_test_bf_pbr_bop_challenge.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_test_bf_pbr_bop_challenge.py) | BOP prediction export |

## Data and checkpoint layout

Meshes and metadata live in `<dataset>/models/`, the detector under `<dataset>/yolo11/`, and pose checkpoints under `<dataset>/HccePose/obj_XX/best_score/`. Training scenes and surface labels use `train_pbr/`, `train_pbr_xyz_GT_front/`, and `train_pbr_xyz_GT_back/`.

See [Assets](assets.md) for download layouts and camera files. Script organization does not change BOP object IDs, pixel coordinates or millimeter model units.

## Build the website

```bash
python -m pip install -r docs/requirements.txt
python docs/build.py
python -m http.server 8768 --directory pages
```

Building the site needs no GPU and runs no inference. See the [documentation index](index.md) for other usage guides.
