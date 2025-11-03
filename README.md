<h2 align="center">HccePose (BF)</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2510.10177">
    <img src="https://img.shields.io/badge/arXiv-2510.10177-B31B1B.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/SEU-WYL/HccePose">
    <img src="https://img.shields.io/badge/HuggingFace-HccePose-FFD21E.svg?logo=huggingface&logoColor=white" alt="HuggingFace">
  </a>
</p>

<p align="center">
  <a href="./README.md">English</b> | <a href="./README_CN.md">中文</a>
</p>
<!-- 
<img src="/show_vis/VID_20251011_215403.gif" width=100%>
<img src="/show_vis/VID_20251011_215255.gif" width=100%> -->

## 🧩 Introduction
**HccePose(BF)** introduces a **Hierarchical Continuous Coordinate Encoding (HCCE)** mechanism that encodes the three coordinate components of object surface points into hierarchical continuous codes. Through this hierarchical encoding scheme, the neural network can effectively learn the correspondence between 2D image features and 3D surface coordinates of the object, while significantly enhancing its capability to learn accurate object masks. Unlike traditional methods that only learn the visible front surface of objects, **HccePose(BF)** additionally learns the 3D coordinates of the back surface, thereby establishing denser 2D–3D correspondences and substantially improving pose estimation accuracy.

### <img src="/show_vis/fig2.jpg" width=100%>

## ✨ Update
--- 
- ⚠️ Note: All paths must be absolute paths to avoid runtime errors.
- 2025.10.27: We’ve released cc0textures-512, a lightweight alternative to the original 44GB CC0Textures library — now only 600MB! 👉 [Download here](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/cc0textures-512.zip)
- 2025.10.28: s4_p1_gen_bf_labels.py has been updated. If the dataset does not contain a camera.json, the script will automatically create a default one.
---
## 🔧 Environment Setup

<details>
<summary>Configuration Details</summary>

Download the HccePose(BF) Project and Unzip BOP-related Toolkits
```bash
# Clone the project
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose

# Unzip toolkits
unzip bop_toolkit.zip
unzip blenderproc.zip
```
Configure Ubuntu System Environment (Python 3.10)

> ⚠️ A GPU driver with EGL support must be pre-installed.
```bash 
apt-get update && apt-get install -y wget software-properties-common gnupg2 python3-pip

apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

apt-get update && apt-get install pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl ninja-build

pip install ultralytics==8.3.70 fvcore==0.1.5.post20221221 pybind11==2.12.0 trimesh==4.2.2 ninja==1.11.1.1 kornia==0.7.2 open3d==0.19.0 transformations==2024.6.1 numpy==1.26.4 opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80

pip install scipy kiwisolver matplotlib imageio pypng Cython PyOpenGL triangle glumpy Pillow vispy imgaug mathutils pyrender pytz tqdm tensorboard kasal-6d rich h5py

pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

apt-get install libsm6 libxrender1 libxext-dev

python -c "import imageio; imageio.plugins.freeimage.download()"

pip install -U "huggingface_hub[hf_transfer]"

```

</details>

---



## 🧱 Custom Dataset and Training

#### 🎨 Object Preprocessing

<details>
<summary>Click to expand</summary>

Using the [**`demo-bin-picking`**](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking) dataset as an example, we first designed the object in **SolidWorks** and exported it as an STL mesh file.  
STL file link: 🔗 https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/raw-demo-models/multi-objs/board.STL

<img src="/show_vis/Design-3DMesh.jpg" width=100%>

Then, the STL file was imported into **MeshLab**, and surface colors were filled using the **`Vertex Color Filling`** tool.

<img src="/show_vis/color-filling.png" width=100%>
<img src="/show_vis/color-filling-2.png" width=100%>

After coloring, the object was exported as a **non-binary PLY file** containing vertex colors and normals.

<img src="/show_vis/export-3d-mesh-ply.png" width=100%>

The exported model center might not coincide with the coordinate origin, as shown below:

<img src="/show_vis/align-center.png" width=100%>

To align the model center with the origin, use the script **`s1_p1_obj_rename_center.py`**.  
This script loads the PLY file, aligns the model center, and renames it following BOP conventions.  
The **`obj_id`** must be set manually as a unique non-negative integer for each object.  
Example:

| **`input_ply`** | **`obj_id`** | **`output_ply`** |
| :---: | :---: | :---: |
| **`board.ply`** | **`1`** | **`obj_000001.ply`** |
| **`board.ply`** | **`2`** | **`obj_000002.ply`** |

After centering and renaming all objects, place them into a folder named **`models`** with the following structure:

```bash
Dataset_Name
|--- models
      |--- obj_000001.ply
      ...
      |--- obj_000015.ply
```

---

</details>

#### 🌀 Rotational Symmetry Analysis

<details>
<summary>Click to expand</summary>

In 6D pose estimation tasks, many objects exhibit various types of rotational symmetry, such as cylindrical, conical, or polyhedral symmetry. For such objects, the KASAL tool is used to analyze and export symmetry priors in BOP format.

KASAL project: 🔗 https://github.com/WangYuLin-SEU/KASAL

Installation:

```bash
pip install kasal-6d
```

Launch the **KASAL GUI** with:

```python
from kasal.app.polyscope_app import app
mesh_path = 'demo-bin-picking'
app(mesh_path)
```

KASAL automatically scans all PLY or OBJ files under **`mesh_path`** (excluding generated **`_sym.ply`** files).

<img src="/show_vis/kasal-1.png" width=100%>

In the interface:
* Use **`Symmetry Type`** to select the symmetry category
* For n-fold pyramidal or prismatic symmetry, set **`N (n-fold)`**
* Enable **`ADI-C`** for texture-symmetric objects
* If the result is inaccurate, use **`axis xyz`** for manual fitting

KASAL defines **8 symmetry types**.
Selecting the wrong one will result in visual anomalies, helping verify your choice.

<img src="/show_vis/kasal-2.png" width=100%>

Click **`Cal Current Obj`** to compute the object’s symmetry axis.
Symmetry priors will be saved as:
* Symmetry prior file: **`obj_000001_sym_type.json`**
* Visualization file: **`obj_000001_sym.ply`**

---

</details>

#### 🧾 Generating BOP-Format Model Information

<details>
<summary>Click to expand</summary>

Run **`s1_p3_obj_infos.py`** to traverse all **`ply`** files and their symmetry priors in the **`models`** folder.
This script generates a standard **`models_info.json`** file in BOP format.

Example structure:

```bash
Dataset_Name
|--- models
      |--- models_info.json
      |--- obj_000001.ply
      ...
      |--- obj_000015.ply
```

This file serves as the foundation for PBR rendering, YOLOv11 training, and HccePose(BF) model training.

---

</details>

#### 🔥 Rendering the PBR Dataset

<details>
<summary>Click to expand</summary>

Based on **BlenderProc**, we modified a rendering script — **`s2_p1_gen_pbr_data.py`** — for generating new datasets. Running this script directly in Python may cause a **memory leak**, which accumulates over time and gradually degrades rendering performance. To address this issue, we provide a **Shell script** — **`s2_p1_gen_pbr_data.sh`** — that repeatedly invokes **`s2_p1_gen_pbr_data.py`**, effectively preventing memory accumulation and improving efficiency. Additionally, several adjustments were made to BlenderProc to better support PBR dataset generation for new object sets.

---

#### Preparation Before Rendering

Before rendering, use **`s2_p0_download_cc0textures.py`** to download the **CC0Textures** material library.  
After downloading, the directory structure should look like this:
```
HCCEPose
|--- s2_p0_download_cc0textures.py
|--- cc0textures
```
---

The **cc0textures** library occupies about **44GB** of disk space, which is quite large.
To reduce storage requirements, we provide a lightweight alternative called **cc0textures-512**, with a size of approximately **600MB**.
Download link:
👉 https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/cc0textures-512.zip

When running the rendering script, simply replace the **`cc0textures`** path with **`cc0textures-512`** to use the lightweight material library.
(It is sufficient to download only **`cc0textures-512`**; the original **`cc0textures`** is not required.)

---

#### Running the Renderer

The **`s2_p1_gen_pbr_data.py`** script is responsible for PBR data generation, and it is adapted from [BlenderProc2](https://github.com/DLR-RM/BlenderProc).

Run the following commands:

```bash
cd HCCEPose
chmod +x s2_p1_gen_pbr_data.sh
nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &
```

**Folder Structure**

After executing the above process, the program will:
- Use materials from **`xxx/xxx/cc0textures`**;
- Load 3D object models from **`xxx/xxx/demo-bin-picking/models`**;
- Generate **`42 folders`**, each containing **`1000 PBR-rendered frames`**, under **`xxx/xxx/demo-bin-picking`**.

The resulting structure will be:
```
demo-bin-picking
|--- models
|--- train_pbr
      |--- 000000
      |--- 000001
      ...
      |--- 000041
```

---

</details>

#### 🚀 Training the 2D Detector

<details>
<summary>Click to expand</summary>

In 6D pose estimation tasks, a **2D detector** is typically used to locate the object’s bounding box,  from which cropped image regions are used for **6D pose estimation**.  Compared with directly regressing 6D poses from the entire image,  the **two-stage approach (2D detection → 6D pose estimation)** offers better accuracy and stability. Therefore, **HccePose(BF)** is equipped with a 2D detector based on **YOLOv11**.  

The following sections describe how to **convert BOP-format PBR training data** into YOLO-compatible data and how to **train YOLOv11**.

---

#### Converting BOP PBR Data to YOLO Format

To automate the conversion from BOP-style PBR data to YOLO training data, we provide the **`s3_p1_prepare_yolo_label.py`** script. After specifying the dataset path **`xxx/xxx/demo-bin-picking`** and running the script, the program will create a new folder named **`yolo11`** inside **`demo-bin-picking`**.

The generated directory structure is as follows:

```
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
      |--- train_obj_s
            |--- images
            |--- labels
            |--- yolo_configs
                |--- data_objs.yaml
            |--- autosplit_train.txt
            |--- autosplit_val.txt
```

Explanation:  
- **`images`** → Folder containing 2D training images  
- **`labels`** → Folder containing 2D bounding box (BBox) annotations  
- **`data_objs.yaml`** → YOLO configuration file  
- **`autosplit_train.txt`** → List of training samples  
- **`autosplit_val.txt`** → List of validation samples  

---

#### Training the YOLOv11 Detector

To train the YOLOv11 detector, use the **`s3_p2_train_yolo.py`** script. After specifying the dataset path **`xxx/xxx/demo-bin-picking`**, run the script to train YOLOv11 and save the **best model weights** as **`yolo11-detection-obj_s.pt`**.  

The final directory structure after training is shown below:

```
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
      |--- train_obj_s
            |--- detection
                |--- obj_s
                    |--- yolo11-detection-obj_s.pt
            |--- images
            |--- labels
            |--- yolo_configs
                |--- data_objs.yaml
            |--- autosplit_train.txt
            |--- autosplit_val.txt
```

---

#### ⚠️ Important Note

**`s3_p2_train_yolo.py`** continuously scans the detection folder for the file **`yolo11-detection-obj_s.pt`**.
This mechanism allows the training process to automatically resume after unexpected interruptions, which is particularly useful for cloud servers or environments where training progress cannot be easily monitored — helping to prevent idle GPU time and reduce unnecessary costs.
However, if you intend to restart training from scratch, you must delete the file **`yolo11-detection-obj_s.pt`** first;
otherwise, the program will resume from the previous checkpoint instead of reinitializing.

---

</details>

#### 🧩 Preparation of Front–Back Surface Labels

<details>
<summary>Click to expand</summary>

In **HccePose(BF)**, the network simultaneously learns the **front-surface** and **back-surface 3D coordinates** of each object. To generate these labels, separate depth maps are rendered for the front and back surfaces.

During front-surface rendering, **`gl.glDepthFunc(gl.GL_LESS)`** is applied to preserve the **smallest depth values**, corresponding to the points closest to the camera. These are defined as the **front surfaces**, following the “front-face” concept used in traditional back-face culling. Similarly, for back-surface rendering, **`gl.glDepthFunc(gl.GL_GREATER)`** is used to retain the **largest depth values**, corresponding to the farthest visible surfaces. Finally, the **3D coordinate label maps** are generated based on these depth maps and the ground-truth 6D poses.

---

#### Symmetry Handling and Pose Correction

For symmetric objects, both **discrete** and **continuous rotational symmetries** are represented as a unified set of symmetry matrices.  
Using these matrices and the ground-truth pose, a new set of valid ground-truth poses is computed.  
To ensure **pose label uniqueness**, the pose with the **minimum L2 distance** from the identity matrix is selected as the final label.

Moreover, due to the imaging principle, when an object undergoes translation without rotation, a **visual rotation** can occur from a fixed viewpoint. For symmetric objects, this apparent rotation can cause erroneous 3D label maps. To correct this effect, we reconstruct 3D coordinates from the rendered depth maps and apply **RANSAC PnP** to refine the rotation.

---

#### Batch Label Generation

Based on the above procedure, we implement **`s4_p1_gen_bf_labels.py`**, a multi-process rendering script for generating front and back 3D coordinate label maps in batches. After specifying the dataset path **`/root/xxxxxx/demo-bin-picking`** and the subfolder **`train_pbr`**, running the script produces two new folders:

- **`train_pbr_xyz_GT_front`** — Front-surface 3D label maps  
- **`train_pbr_xyz_GT_back`** — Back-surface 3D label maps  

Directory structure:

```
demo-bin-picking
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```


The following example shows three corresponding images:  
the original rendering, the front-surface label map, and the back-surface label map.
<p align="center">
  <img src="/show_vis/000000.jpg" width="32%">
  <img src="/show_vis/000000_000000-f.png" width="32%">
  <img src="/show_vis/000000_000000-b.png" width="32%">
</p>

---

</details>



#### 🚀 Training HccePose(BF)

<details>
<summary>Click to expand</summary>

When training **HccePose(BF)**, a separate weight model must be trained for each object.  
The **`s4_p2_train_bf_pbr.py`** script supports **multi-GPU batch training** across multiple objects.

Taking the `demo-tex-objs` dataset as an example, the directory structure after training is as follows:

```
demo-tex-objs
|--- HccePose
    |--- obj_01
    ...
    |--- obj_10
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```


The **`ide_debug`** flag controls whether the script runs in **single-GPU** or **multi-GPU (DDP)** mode:
- `ide_debug=True` → Single-GPU mode, ideal for debugging in IDEs.  
- `ide_debug=False` → Enables **DDP (Distributed Data Parallel)** training.

Note that directly running DDP training within IDEs such as VSCode may cause communication issues.  
Hence, we recommend launching multi-GPU training in a detached session:

```
screen -S train_ddp
nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
``` 


For single-GPU execution or debugging, use:

```
nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
```  

---

#### Setting Training Ranges

To train multiple objects, specify the range of object IDs using **`start_obj_id`** and **`end_obj_id`**. For example, setting `start_obj_id=1` and `end_obj_id=5` trains objects `obj_000001.ply` through `obj_000005.ply`. To train a single object, set both values to the same number.

You may also adjust **`total_iteration`** according to training needs (default: `50000`). For DDP training, the total number of training samples is computed as:

```
total samples = total iteration × batch size × GPU number
```


---

</details>


---



## ✏️ Quick Start

This project provides a simple **HccePose(BF)-based** application example for the **Bin-Picking** task.  
To reduce reproduction difficulty, both the objects (3D printed with standard white PLA material) and the camera (Xiaomi smartphone) are easily accessible devices.

You can:
- Print the sample object multiple times  
- Randomly place the printed objects  
- Capture photos freely using your phone  
- Directly perform **2D detection**, **2D segmentation**, and **6D pose estimation** using the pretrained weights provided in this project  

---


> Please keep the folder hierarchy unchanged.

| Type | Resource Link |
|------|----------------|
| 🎨 Object 3D Models | [demo-bin-picking/models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/models) |
| 📁 YOLOv11 Weights | [demo-bin-picking/yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/yolo11) |
| 📂 HccePose Weights | [demo-bin-picking/HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-bin-picking/HccePose) |
| 🖼️ Test Images | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs) |
| 🎥 Test Videos | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos) |

> ⚠️ Note:  
Files beginning with **`train_`** are only required for training.  
For this **Quick Start** section, only the above test files are needed.

---

#### ⏳ Model and Loader
During testing, import the following modules:
- **`HccePose.tester`** → Integrated testing module covering **2D detection**, **segmentation**, and **6D pose estimation**.  
- **`HccePose.bop_loader`** → BOP-format dataset loader for loading object models and training data.

---

#### 📸 Example Test
The following image shows the experimental setup:  

<details>
<summary>Click to expand</summary>

Several white 3D-printed objects are placed inside a bowl on a white table, then photographed with a mobile phone.  

Example input image 👇  
<div align="center">
 <img src="/test_imgs/IMG_20251007_165718.jpg" width="40%">
</div>

Source image: [Example Link](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/test_imgs/IMG_20251007_165718.jpg)

</details>

You can directly use the following script for **6D pose estimation** and visualization:

<details>
<summary>Click to expand code</summary>

```python
import cv2, os, sys
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester
if __name__ == '__main__':

    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_img_path = os.path.join(current_dir, 'test_imgs')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    
    for name in ['IMG_20251007_165718']:
        file_name = os.path.join(test_img_path, '%s.jpg'%name)
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_RGB2BGR)
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        results_dict = Tester_item.predict(cam_K, image, [obj_id],
                                                        conf = 0.85, confidence_threshold = 0.85)
        cv2.imwrite(file_name.replace('.jpg','_show_2d.jpg'), results_dict['show_2D_results'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis0.jpg'), results_dict['show_6D_vis0'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis1.jpg'), results_dict['show_6D_vis1'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis2.jpg'), results_dict['show_6D_vis2'])
    pass
```

</details>

---

#### 🎯 Visualization Results

2D Detection Result (_show_2d.jpg):

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_2d.jpg" width="40%"> </div>


Network Outputs:

- HCCE-based front and back surface coordinate encodings

- Object mask

- Decoded 3D coordinate visualizations

<div align="center"> <img src="/show_vis/IMG_20251007_165718_show_6d_vis0.jpg" width="100%"> 
<img src="/show_vis/IMG_20251007_165718_show_6d_vis1.jpg" width="100%"> </div>

---

#### 💫 If you find this tutorial helpful

Please consider giving it a ⭐️ on GitHub!
Your support motivates us to keep improving and updating the project 🙌

--- 

#### 🎥 6D Pose Estimation in Videos

<details>
<summary>Detailed Content</summary>

The single-frame pose estimation pipeline can be easily extended to video sequences, enabling continuous-frame 6D pose estimation, as shown in the following example:

<details>
<summary>Click to expand code</summary>

```python
import cv2, os, sys
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester

if __name__ == '__main__':
    
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_video_path = os.path.join(current_dir, 'test_videos')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    for name in ['VID_20251009_141247']:
        file_name = os.path.join(test_video_path, '%s.mp4'%name)
        cap = cv2.VideoCapture(file_name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_1 = None
        out_2 = None
        cam_K = np.array([
            [1.63235512e+03, 0.00000000e+00, 9.74032712e+02],
            [0.00000000e+00, 1.64159967e+03, 5.14229781e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results_dict = Tester_item.predict(cam_K, frame, [obj_id],
                                                            conf = 0.85, confidence_threshold = 0.85)
            fps_hccepose = 1 / results_dict['time']
            show_6D_vis1 = results_dict['show_6D_vis1']
            show_6D_vis1[show_6D_vis1 < 0] = 0
            show_6D_vis1[show_6D_vis1 > 255] = 255
            if out_1 is None:
                out_1 = cv2.VideoWriter(
                    file_name.replace('.mp4', '_show_1.mp4'),
                    fourcc,
                    fps,
                    (show_6D_vis1.shape[1], show_6D_vis1.shape[0])
                )
            out_1.write(show_6D_vis1.astype(np.uint8))
            show_6D_vis2 = results_dict['show_6D_vis2']
            show_6D_vis2[show_6D_vis2 < 0] = 0
            show_6D_vis2[show_6D_vis2 > 255] = 255
            if out_2 is None:
                out_2 = cv2.VideoWriter(
                    file_name.replace('.mp4', '_show_2.mp4'),
                    fourcc,
                    fps,
                    (show_6D_vis2.shape[1], show_6D_vis2.shape[0])
                )
            cv2.putText(show_6D_vis2, "FPS: {0:.2f}".format(fps_hccepose), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
            out_2.write(show_6D_vis2.astype(np.uint8))
        cap.release()
        out_1.release()
        out_2.release()
    pass
```

</details>

--- 

#### 🎯 Visualization Results

**Original Video:**
<img src="/show_vis/VID_20251009_141247.gif" width=100%>

**Detection Results:**
<img src="/show_vis/VID_20251009_141247_vis.gif" width=100%>

---

In addition, by passing a list of multiple object IDs to **`HccePose.tester`**, multi-object 6D pose estimation can also be achieved.  

> Please keep the folder hierarchy unchanged.

| Type | Resource Link |
|------|----------------|
| 🎨 Object 3D Models | [demo-tex-objs/models](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/models) |
| 📁 YOLOv11 Weights | [demo-tex-objs/yolo11](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/yolo11) |
| 📂 HccePose Weights | [demo-tex-objs/HccePose](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/demo-tex-objs/HccePose) |
| 🖼️ Test Images | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_imgs) |
| 🎥 Test Videos | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/test_videos) |

> ⚠️ Note:  
Files beginning with **`train_`** are only required for training.  
For this **Quick Start** section, only the above test files are needed.

**Original Video:**
<img src="/show_vis/VID_20251009_141731.gif" width=100%>

**Detection Results:**
<img src="/show_vis/VID_20251009_141731_vis.gif" width=100%>

</details>

---



## 🧪 BOP Challenge Testing

You can use the script [**`s4_p2_test_bf_pbr_bop_challenge.py`**](/s4_p2_test_bf_pbr_bop_challenge.py) to evaluate **HccePose(BF)** across the seven core BOP datasets.

#### Pretrained Weights

| Dataset | Weights Link |
|----------|---------------|
| **LM-O** | [Hugging Face - LM-O](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/lmo/HccePose) |
| **YCB-V** | [Hugging Face - YCB-V](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/ycbv/HccePose) |
| **T-LESS** | [Hugging Face - T-LESS](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/tless/HccePose) |
| **TUD-L** | [Hugging Face - TUD-L](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/tudl/HccePose) |
| **HB** | [Hugging Face - HB](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/hb/HccePose) |
| **ITODD** | [Hugging Face - ITODD](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/itodd/HccePose) |
| **IC-BIN** | [Hugging Face - IC-BIN](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/main/icbin/HccePose) |

---

#### Example: LM-O Dataset

As an example, we evaluated **HccePose(BF)** on the widely used **LM-O dataset** from the BOP benchmark. We adopted the [default 2D detector](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task1.zip) (GDRNPP) from the **BOP 2023 Challenge** and obtained the following output files:

- 2D segmentation results: [seg2d_lmo.json](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/lmo/seg2d_lmo.json)
- 6D pose results: [det6d_lmo.csv](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/lmo/det6d_lmo.csv)

These two files were submitted on **October 20, 2025**. The results are shown below.  
The **6D localization score** remains consistent with the 2024 submission,  
while the **2D segmentation score** improved by **0.002**, thanks to the correction of minor implementation bugs.
<details>
<summary>Click to expand</summary>
### <img src="/show_vis/BOP-website-lmo.png" width=100%>
</details>

---

#### ⚙️ Notes

- If some pretrained weights show an iteration count of **`0`**, this is **not an error**. All **HccePose(BF)** weights are fine-tuned from the standard HccePose model trained using only the front surface. In some cases, the initial weights already achieve optimal performance.

---

## 📅 Update Plan

We are currently organizing and updating the following modules:

- 📁 ~~HccePose(BF) weights for the seven core BOP datasets~~

- 🧪 ~~BOP Challenge testing pipeline~~

- 🔁 6D pose inference via inter-frame tracking

- 🏷️ Real-world 6D pose dataset preparation based on HccePose(BF)

- ⚙️ PBR + Real training workflow

- 📘 Tutorials on ~~object preprocessing~~, ~~data rendering~~, ~~YOLOv11 label preparation and training~~, as well as HccePose(BF) ~~label preparation~~ and ~~training~~

All components are expected to be completed by the end of 2025, with continuous daily updates whenever possible.

---

## 🏆 BOP LeaderBoards
### <img src="/show_vis/bop-6D-loc.png" width=100%>
### <img src="/show_vis/bop-2D-seg.png" width=100%>


***
If you find our work useful, please cite it as follows: 
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