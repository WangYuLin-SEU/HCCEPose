# Data loading API

The BOP loader reads object-model metadata. For inference on your own images with [`Tester`](tester-api.md), prepare the model metadata and matching weights. Call `load_folder()` when you also need BOP scenes and annotations.

## Model metadata

```python
from HccePose.bop_loader import bop_dataset

bop_dataset(dataset_path, model_name='models', local_rank=0)
```

`dataset_path` is the dataset root, `model_name` selects its model subdirectory, and `local_rank` controls informational printing. The loader reads `<dataset_path>/<model_name>/models_info.json` and constructs model filenames as `obj_000001.ply`, using six-digit IDs.

| Attribute | Meaning |
| --- | --- |
| `dataset_path`, `model_path` | Dataset root and selected model directory. |
| `model_info` | Model metadata keyed by string object IDs. Contains dimensions, minimum coordinates, diameter and optional symmetry information. |
| `obj_id_list` | Integer object IDs in metadata iteration order. |
| `obj_model_list` | PLY paths aligned with `obj_id_list`. |
| `obj_info_list` | Metadata entries in the same order. |

YOLO class indices start at zero. `Tester` uses the class index as a position in `obj_id_list` to look up the object ID. Preserve the detector's training class order if you replace the detector or reorganize the metadata.

If the dataset root or model directory is missing, the loader prints a message and returns early. It also reports missing PLY files. Check any reported paths before creating a `Tester` or training dataset.

On Linux, importing `bop_loader` sets `PYOPENGL_PLATFORM=egl`. It imports the repository-local BOP toolkit and augmentation dependencies even when used only for inference. Run `python -m zipfile -e bop_toolkit.zip .` from the project root; the extracted library is at `bop_toolkit/bop_toolkit_lib/`.

Sources: [imports and EGL setting](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py), [model loader](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py), [detector class mapping](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/tester.py).

## Scene metadata

```python
dataset.load_folder(folder_name, scene_num=200, vis=0.0)
```

- `folder_name` is a dataset-relative directory, such as `train_pbr` or `test`.
- `scene_num` scans consecutive six-digit scene IDs, starting at `000000` and ending at ID `scene_num - 1`.
- `vis` keeps annotated instances with `visib_fract >= vis`.

Use the BOP scene layout with `scene_camera.json`, `scene_gt.json`, `scene_gt_info.json`, `rgb/` or `gray/`, `depth/`, `mask/` and `mask_visib/`. The loader handles some missing annotations but uses the depth and mask file extensions when constructing paths. Pass standalone RGB images directly to `Tester`.

```python
{
    'img_info': {...},        # Keys such as "000000_000001" (scene ID_image ID)
    'obj_info': {...},        # Keys such as "obj_000001" (object ID)
    'scene_path_list': [...], # Existing numeric scene folders
}
```

`img_info` groups paths, camera information and annotated instances by image. `obj_info` groups instance records by object. The call indexes metadata; image pixels are loaded later by dataset wrappers. If the requested folder does not exist, it returns `None`.

Source: [`load_folder`](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py).

## Training and evaluation wrappers

```python
train_bop_dataset_back_front(
    bop_dataset_item, folder_name,
    padding_ratio=1.5, crop_size_img=256, aug_op='imgaug',
)

test_bop_dataset_back_front(
    bop_dataset_item, folder_name,
    bbox_2D=None, test_targets_bop19=None,
    bbox_2D_score_threshold=0.0,
    padding_ratio=1.5, crop_size_img=256, ratio=1.0,
)
```

Before passing either dataset to a `DataLoader`, select the object with `update_obj_id(obj_id, obj_path)`. Both datasets load scenes with a visibility threshold of `0.2`. Initialization may create the `<folder_name>_xyz_GT_front/` and `<folder_name>_xyz_GT_back/` directories.

Training samples return `(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)`. Evaluation samples additionally return `(Bbox, cam_K, cam_R_m2c, cam_t_m2c)`; with external detections they append `(scene_id, image_id, score)`. Crops are BGR-normalized tensors. Existing training recipes handle batching, current-object selection and checkpoint metadata.

Missing front or back label images are filled with zeros. Check the generated labels and inspect a few samples before training.

External bbox JSON entries use `scene_id`, `image_id`, `category_id`, `bbox`, and `score`; the score filter is strictly `score > bbox_2D_score_threshold`. The BOP challenge script provides an existing integration example.

Sources: [training wrapper](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py), [evaluation wrapper](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py), [sample return values](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/bop_loader.py).

See the [refinement API](refinement-api.md) for the three files in an RGB-D frame and how to convert its depth. The [utilities guide](utilities-api.md) lists the training and evaluation scripts.
