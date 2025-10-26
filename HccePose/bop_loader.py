# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os, cv2, sys, json, copy, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
from torch.utils.data import Dataset

# Configure the EGL renderer.
# 设置 EGL 渲染器。
import platform
sys0 = platform.system()
if sys0 == "Linux":
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# Set the path to `bop_toolkit`.
# 设置 `bop_toolkit` 的路径。
sys.path.insert(0, os.getcwd())
current_directory = sys.argv[0]
pa_ = os.path.join(os.path.dirname(current_directory), 'bop_toolkit')
sys.path.append(pa_)
from bop_toolkit.bop_toolkit_lib import inout, renderer, misc, pose_error, pycoco_utils
from kasal.utils import load_json2dict

def aug_square_fp32(GT_Bbox, padding_ratio):
    '''
    ---
    ---
    Randomly augment a 2D bounding box.  
    ---
    ---
    The function randomly shifts the bounding box center along the x and y axes  
    within the range [-0.25, +0.25) times the width and height,  
    simulating deviations between the detector-predicted and ground-truth boxes.  
    The augmented bounding box is then adjusted to a square shape  
    based on the maximum side length of the original box.  

    Args:
        - GT_Bbox: The 2D bounding box, formatted as [x, y, w, h],  
        where x and y are the coordinates of the top-left corner.  
        - padding_ratio: The scaling factor for the 2D bounding box, typically set to 1.5.  

    Returns:
        - augmented_Box: The augmented bounding box.
    ---
    ---
    随机增强 2D 包围盒。  
    ---
    ---
    函数会在 x、y 方向上随机移动包围盒的中心，移动范围为 [-0.25, +0.25) 倍的宽与高，  
    以模拟检测器预测的包围盒与真实包围盒之间的偏差。  
    随后，函数会根据原始包围盒的最大边长，调整增强后的包围盒为正方形形状。  

    参数:
        - GT_Bbox: 2D 包围盒，格式为 [x, y, w, h]，其中 x、y 分别为左上角点的坐标。  
        - padding_ratio: 2D 包围盒的缩放比例，通常设置为 1.5。  

    返回:
        - augmented_Box: 增强后的包围盒。
    '''

    GT_Bbox = GT_Bbox.copy()
    center_x = GT_Bbox[0] + 0.5 * GT_Bbox[2]
    center_y = GT_Bbox[1] + 0.5 * GT_Bbox[3]
    width = GT_Bbox[2]
    height = GT_Bbox[3]
    scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)
    shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1) 
    bbox_center = np.array([center_x + width * shift_ratio[0], center_y + height * shift_ratio[1]]) 
    augmented_width = width * scale_ratio * padding_ratio
    augmented_height = height * scale_ratio * padding_ratio
    w = max(augmented_width, augmented_height) 
    augmented_Box = np.array([bbox_center[0]-w/2, bbox_center[1]-w/2, w, w])
    return augmented_Box

def pad_square_fp32(GT_Bbox, padding_ratio):
    '''
    ---
    ---
    Pad a 2D bounding box.  
    ---
    ---
    The function pads the original bounding box into a square shape  
    based on its maximum side length and the specified `padding_ratio`.  

    Args:
        - GT_Bbox: 2D bounding box in the format [x, y, w, h],  
        where x and y represent the coordinates of the top-left corner.  
        - padding_ratio: Scaling factor for padding the bounding box, typically set to 1.5.  

    Returns:
        - padded_Box: The padded bounding box.
    ---
    ---
    填充 2D 包围盒。  
    ---
    ---
    函数会根据原始包围盒的最大边长，并按 `padding_ratio` 的比例，  
    将包围盒填充为正方形形状。  

    参数:
        - GT_Bbox: 2D 包围盒，格式为 [x, y, w, h]，其中 x、y 为左上角点的坐标。  
        - padding_ratio: 2D 包围盒的缩放比例，通常设置为 1.5。  

    返回:
        - padded_Box: 填充后的包围盒。
    '''

    
    GT_Bbox = GT_Bbox.copy()
    center_x = GT_Bbox[0] + 0.5 * GT_Bbox[2]
    center_y = GT_Bbox[1] + 0.5 * GT_Bbox[3]
    width = GT_Bbox[2]
    height = GT_Bbox[3]
    bbox_center = np.array([center_x, center_y]) 
    padded_width = width * padding_ratio
    padded_height = height * padding_ratio
    w = max(padded_width, padded_height) 
    padded_Box = np.array([bbox_center[0]-w/2, bbox_center[1]-w/2, w, w])
    return padded_Box

def crop_square_resize(img, Bbox, crop_size=None, interpolation=None):
    '''
    ---
    ---
    Crop the image within a 2D bounding box.  
    ---
    ---
    The function crops the input image into a square region  
    based on the given 2D bounding box.  

    Args:
        - img: The input 2D image.  
        - Bbox: The 2D bounding box in the format [x, y, w, h],  
        where x and y are the coordinates of the top-left corner.  
        - crop_size: The side length of the cropped square image.  
        - interpolation: The interpolation algorithm used for image cropping.  

    Returns:
        - roi_img: The cropped square image within the 2D bounding box.
    ---
    ---
    裁切 2D 包围盒内的图像。  
    ---
    ---
    函数会根据给定的 2D 包围盒，将输入图像裁切为正方形区域。  

    参数:
        - img: 输入的 2D 图像。  
        - Bbox: 2D 包围盒，格式为 [x, y, w, h]，其中 x、y 为左上角点的坐标。  
        - crop_size: 裁切后正方形图像的边长。  
        - interpolation: 图像裁切时使用的插值算法。  

    返回:
        - roi_img: 2D 包围盒内的正方形图像。
    '''

    Bbox = Bbox.copy()
    center_x = Bbox[0] + 0.5 * Bbox[2]
    center_y = Bbox[1] + 0.5 * Bbox[3]
    w_2 = Bbox[2] / 2
    pts1 = np.float32([[center_x - w_2, center_y - w_2], [center_x - w_2, center_y + w_2], [center_x + w_2, center_y - w_2]])
    pts2 = np.float32([[0, 0], [0, crop_size], [crop_size, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    roi_img = cv2.warpAffine(img, M, (crop_size, crop_size), flags=interpolation)
    return roi_img

class bop_dataset():
    '''
    ---
    ---
    Instance of the BOP dataset.  
    ---
    ---
    `bop_dataset` is used to load datasets in BOP format,  
    including the paths of all 3D object models in the `models` folder and their basic information.  
    By specifying a folder path, `bop_dataset` can load all images, poses, masks, and related data within that directory.
    ---
    ---
    BOP 数据集的实例。  
    ---
    ---
    `bop_dataset` 用于加载 BOP 格式的数据集，  
    包括 `models` 文件夹中所有物体 3D 模型的路径及其基础信息。  
    通过指定文件夹路径，`bop_dataset` 可以加载文件夹下的所有图像、位姿、掩膜等数据。
    '''

    def __init__(self, dataset_path, model_name = 'models', local_rank=0):
        '''
        ---
        ---
        Initialization function.  
        ---
        ---
        Specifies the dataset path and the folder name containing 3D object models,  
        then loads information for all objects.  

        Args:
            - dataset_path: Path to the dataset.  
            - model_name: Name of the folder where 3D models are stored.  
            Most BOP datasets use `models` as the folder name,  
            but it can be changed if a different name is used.  
            - local_rank: In DDP training mode, information is printed only in process 0.  

        Returns:
            - None
        ---
        ---
        初始化函数。  
        ---
        ---
        指定数据集路径及物体 3D 模型文件夹的名称，并加载所有物体的信息。  

        参数:
            - dataset_path: 数据集路径。  
            - model_name: 模型文件夹名称。大多数 BOP 数据集的模型文件夹名称为 `models`，  
            若不同，可通过修改该参数重新设置。  
            - local_rank: 在 DDP 训练模式下，默认仅在进程 0 中打印信息。 
             
        返回:
            - 无
        '''
        
        self.local_rank = local_rank
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            if local_rank == 0:
                print()
                print('dataset_path is not existed: ', self.dataset_path)
                print()
            return
        self.dataset_name = os.path.basename(dataset_path)
        if local_rank == 0:
            print()
            print('-*-' * 30)
            print('dataset name: ', self.dataset_name)
            print()
        self.model_path = os.path.join(dataset_path, model_name)
        if not os.path.exists(self.model_path):
            if local_rank == 0:
                print()
                print('model_name is not existed: ', self.model_path)
                print()
            return
        if local_rank == 0:
            print('obj model path: ', self.model_path)
        self.model_info = load_json2dict(os.path.join(self.model_path, 'models_info.json'))
        self.obj_id_list = []
        self.obj_model_list = []
        self.obj_info_list = []
        for key_i in self.model_info:
            self.obj_id_list.append(int(key_i)) 
            ply_path = os.path.join(self.model_path, 'obj_%s.ply'%str(int(key_i)).rjust(6, '0'))
            if not os.path.exists(ply_path):
                if local_rank == 0:
                    print()
                    print('%s is not existed'%ply_path)
                    print()
            self.obj_model_list.append(ply_path)
            self.obj_info_list.append(self.model_info[key_i])
        for i in range(len(self.obj_id_list)):
            if local_rank == 0:
                print('obj id: %s        obj model: %s'%(str(self.obj_id_list[i]).rjust(4, ' '), self.obj_model_list[i]))
        if local_rank == 0:
            print('-*-' * 30)
            print()
        pass
    
    def load_folder(self, folder_name, scene_num = 200, vis = 0.0):
        '''
        ---
        ---
        Load all images, masks, poses, depth maps, and related data from the folder.  
        ---
        ---
        Args:
            - folder_name: The name of the folder in the dataset to load.  
            - scene_num: The maximum number of subfolders to scan within the folder.  
            - vis: Visibility threshold; only samples with a visible ratio greater than `vis` will be loaded.  

        Returns:
            - img_info: Information for each image, including object masks, poses, etc.  
            - obj_info: Information for each object, including all corresponding images, masks, and poses.  
            - scene_path_list: A list of subfolder paths.
        ---
        ---
        加载文件夹中的所有图像、掩膜、位姿、深度图等数据。  
        ---
        ---
        参数:
            - folder_name: 需要加载的数据集中的文件夹名称。  
            - scene_num: 文件夹中最大子文件夹的扫描数量。  
            - vis: 物体的可见比例，只有可见比例大于 `vis` 的样本才会被加载。  

        返回:
            - img_info: 每张图像对应的物体掩膜、位姿等信息。  
            - obj_info: 每个物体对应的所有图像、掩膜、位姿等信息。  
            - scene_path_list: 子文件夹路径列表。
        '''

        if self.local_rank == 0:
            print()
            print('-*-' * 30)
            print('folder name: ', folder_name)
            print()
        folder_path = os.path.join(self.dataset_path, folder_name)
        if not os.path.exists(folder_path):
            if self.local_rank == 0:
                print()
                print('folder_path is not existed: ', folder_path)
                print()
            return None
        
        scene_path_list = []
        for i in range(scene_num):
            scene_name = str(i).rjust(6, '0')
            scene_path = os.path.join(folder_path, scene_name)
            if os.path.exists(scene_path):
                scene_path_list.append(scene_path)
                if self.local_rank == 0:
                    print(scene_path)
        if self.local_rank == 0:
            print('-*-' * 30)
            print()
        
        img_info = {}
        
        obj_info = {}
        
        for scene_path_i in scene_path_list:
            if self.local_rank == 0:
                print('loading: ', scene_path_i)
            scene_camera_path = os.path.join(scene_path_i, 'scene_camera.json')
            scene_gt_info_path = os.path.join(scene_path_i, 'scene_gt_info.json')
            scene_gt_path = os.path.join(scene_path_i, 'scene_gt.json')
            
            scene_gt_info_dict = None
            if os.path.exists(scene_gt_info_path):
                scene_gt_info_dict = load_json2dict(scene_gt_info_path)
            else:
                if self.local_rank == 0:
                    print()
                    print('scene_gt_info_path is not existed: ', scene_gt_info_path)
                    print()
            
            scene_gt_dict = None
            if os.path.exists(scene_gt_path):
                scene_gt_dict = load_json2dict(scene_gt_path)
            else:
                if self.local_rank == 0:
                    print()
                    print('scene_gt_path is not existed: ', scene_gt_path)
                    print()
            if not os.path.exists(scene_camera_path):
                if self.local_rank == 0:
                    print()
                    print('scene_camera_path is not existed: ', scene_camera_path)
                    print()
                continue
                
            scene_camera_dict = load_json2dict(scene_camera_path)
            
            if os.path.exists(os.path.join(scene_path_i, 'rgb')):rgb_folder_name = 'rgb'
            if os.path.exists(os.path.join(scene_path_i, 'gray')):rgb_folder_name = 'gray'
            dep_folder_name = 'depth'
            mask_folder_name = 'mask'
            mask_vis_folder_name = 'mask_visib'
            
            rgb_suffix = None
            dep_suffix = None
            mask_suffix = None
            mask_vis_suffix = None
            
            for camera_key in tqdm(scene_camera_dict):
                camera_i = scene_camera_dict[camera_key]
                
                scene_gt_info_i = None
                scene_gt_i = None
                if scene_gt_info_dict is not None:
                    scene_gt_info_i = scene_gt_info_dict[camera_key]
                if scene_gt_dict is not None:
                    scene_gt_i = scene_gt_dict[camera_key]
                camera_key_pad = camera_key.rjust(6, '0')
                if rgb_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, rgb_folder_name, camera_key_pad + suffix_i)):
                            rgb_suffix = suffix_i
                if dep_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, dep_folder_name, camera_key_pad + suffix_i)):
                            dep_suffix = suffix_i
                if mask_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, mask_folder_name, camera_key_pad + '_000000' + suffix_i)):
                            mask_suffix = suffix_i
                if mask_vis_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, mask_vis_folder_name, camera_key_pad + '_000000' + suffix_i)):
                            mask_vis_suffix = suffix_i
                rgb_suffix
                img_info_i = {}
                if rgb_suffix is not None:
                    img_info_i['rgb'] = os.path.join(scene_path_i, rgb_folder_name, camera_key_pad + rgb_suffix)
                    img_info_i['depth'] = os.path.join(scene_path_i, dep_folder_name, camera_key_pad + dep_suffix)
                    img_info_i.update(camera_i)
                    if scene_gt_info_i is not None:
                        for j in range(len(scene_gt_info_i)):
                            scene_gt_i_j_for_obj = {
                                'scene' : os.path.basename(scene_path_i),
                                'image' : camera_key_pad,
                                'rgb' : os.path.join(scene_path_i, rgb_folder_name, camera_key_pad + rgb_suffix),
                                'depth' : os.path.join(scene_path_i, dep_folder_name, camera_key_pad + dep_suffix),
                            }
                            scene_gt_info_i_j = scene_gt_info_i[j]
                            scene_gt_i_j = copy.deepcopy(scene_gt_i[j])
                            scene_gt_i_j.update(scene_gt_info_i_j)
                            scene_gt_i_j.update(camera_i)
                            
                            obj_id = scene_gt_i_j['obj_id']
                            if scene_gt_i_j['visib_fract'] >= vis:
                                scene_gt_i_j['mask_path'] = os.path.join(scene_path_i, mask_folder_name, camera_key_pad + '_' + str(j).rjust(6, '0') + mask_suffix)
                                scene_gt_i_j['mask_visib_path'] = os.path.join(scene_path_i, mask_vis_folder_name, camera_key_pad + '_' + str(j).rjust(6, '0') + mask_vis_suffix)
                                obj_id_key = 'obj_' + str(obj_id).rjust(6, '0')
                                if obj_id_key in img_info_i:
                                    img_info_i[obj_id_key].append(scene_gt_i_j)
                                else:
                                    img_info_i[obj_id_key] = [scene_gt_i_j]
                                scene_gt_i_j_for_obj.update(scene_gt_i_j)
                                scene_gt_i_j_for_obj.update(camera_i)
                                if obj_id_key in obj_info:
                                    obj_info[obj_id_key].append(scene_gt_i_j_for_obj)
                                else:
                                    obj_info[obj_id_key] = [scene_gt_i_j_for_obj]
                img_info['%s_%s'%(os.path.basename(scene_path_i), camera_key_pad)] = img_info_i
        if self.local_rank == 0:
            print('-*-' * 30)
            print()
        return {
            'img_info' : img_info,
            'obj_info' : obj_info,
            'scene_path_list' : scene_path_list,
        }
    
class rendering_bop_dataset_back_front(Dataset):
    '''
    ---
    ---
    Preparation of front and back 3D coordinate label maps.  
    ---
    ---
    `rendering_bop_dataset_back_front` is based on PyTorch’s Dataset and implements multiprocessing
    for generating 3D coordinate label maps of both front and back surfaces.  
    To enable front–back rendering, we modified the VisPy instance in [`bop_toolkit`](https://github.com/thodan/bop_toolkit)
    by adding an option to switch between front and back rendering modes.  

    - Front rendering:
    The depth test is set via `gl.glDepthFunc(gl.GL_LESS)` to retain the smallest depth values,
    corresponding to the surfaces closest to the camera.  
    These nearest surfaces are defined as the object’s “front side”, consistent with the “front–back culling” concept in rendering pipelines.  

    - Back rendering:
    The depth test is set via `gl.glDepthFunc(gl.GL_GREATER)` to retain the largest depth values,
    corresponding to the surfaces farthest from the camera.  

    - Handling rotationally symmetric objects: 
    Both discrete and continuous rotational symmetries are converted into a set of rotational symmetry matrices.  
    Using this matrix set and the ground-truth object pose, a new set of valid pose labels is computed.  
    To maintain the uniqueness of the 6D pose label, the pose with the smallest L2 distance to the identity matrix
    is selected as the final ground-truth pose.  

    - Correcting visually induced rotation:
    According to imaging geometry, when translation changes but rotation remains constant,
    an object may appear visually rotated under a fixed camera viewpoint.  
    To correct this translation-induced apparent rotation, the object’s 3D coordinates are computed from the rendered depth map,
    and rotation refinement is performed using RANSAC PnP.
    ---
    ---
    正背面 3D 坐标标签图制备。  
    ---
    ---
    `rendering_bop_dataset_back_front` 基于 PyTorch 的 Dataset 实现，支持多进程生成物体正背面 3D 坐标的标签图。  
    为实现正背面渲染，我们修改了 [`bop_toolkit`](https://github.com/thodan/bop_toolkit) 中的 VisPy 实例，增加了正背面渲染模式切换的功能。  
    
    - 渲染正面：
    通过 `gl.glDepthFunc(gl.GL_LESS)` 设置深度测试以保留最小的深度值，
    即距离相机最近的物体表面。  
    这些最近的表面被定义为物体的 **正面**，该定义参考了渲染流程中“正背面剔除”的“正面”概念。  

    - 渲染背面：  
    通过 `gl.glDepthFunc(gl.GL_GREATER)` 设置深度测试以保留最大的深度值，
    即距离相机最远的物体表面。  

    - 旋转对称物体处理： 
    将离散与连续旋转对称统一转换为旋转对称矩阵集合，
    并基于该矩阵集合与物体的真值位姿计算出新的真值位姿集合。  
    为保持 6D 位姿标签的唯一性，从中选取与单位矩阵 L2 距离最小的真值位姿作为最终标签。  

    - 修正视觉旋转误差：  
    依据相机成像原理，当物体旋转不变而发生位移时，
    在固定视角下物体会出现“视觉上的旋转”。  
    为修正这种由位移引起的视觉旋转，我们根据渲染得到的深度图计算物体的 3D 坐标，
    并使用 RANSAC PnP 对旋转进行校正。
    '''

    def __init__(self, bop_dataset_item : bop_dataset, folder_name):
        '''
        ---
        ---
        Initialize the preparation of label maps.  
        ---
        ---
        Args:
            - bop_dataset_item: Instance for loading the BOP dataset.  
            - folder_name: Name of the folder within the BOP dataset.  

        Returns:
            - None
        ---
        ---
        初始化标签图制备过程。  
        ---
        ---
        参数:
            - bop_dataset_item: 用于加载 BOP 数据集的实例。  
            - folder_name: BOP 数据集中的文件夹名称。  

        返回:
            - 无
        '''

        # Load the data from the folder.
        # 加载文件夹中的数据。
        self.bop_dataset_item = bop_dataset_item
        self.dataset_info = bop_dataset_item.load_folder(folder_name)
        self.folder_name = folder_name
        self.nSamples = 0
        if self.dataset_info is None:
            return

        # Create a folder for storing label maps.
        # 创建用于存储标签图的文件夹。
        target_dir_front = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_front')
        try:
            os.mkdir(target_dir_front)
        except:
            1
        self.target_dir_front = target_dir_front
        target_dir_back = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_back')
        try:
            os.mkdir(target_dir_back)
        except:
            1
        self.target_dir_back = target_dir_back
        
        # Create subfolders within the label directory according to the folder structure of the BOP dataset.
        # 按照 BOP 数据集的文件夹结构，在标签文件夹中创建对应的子文件夹。
        for scene_path_i in self.dataset_info['scene_path_list']:
            try:
                os.mkdir(os.path.join(self.target_dir_front, os.path.basename(scene_path_i)))
            except:
                1
            try:
                os.mkdir(os.path.join(self.target_dir_back, os.path.basename(scene_path_i)))
            except:
                1
            
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        # Retrieve the information of the sample.
        # 获取样本的信息。
        info_ = self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][index]
        scene_id = info_['scene']
        label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
        
        # Set the save paths for the front and back label maps corresponding to the sample.
        # 设置该样本对应的正面和背面标签图的保存路径。
        front_label_image_path = os.path.join(self.target_dir_front, scene_id, label_image_name + '.png')
        back_label_image_path = os.path.join(self.target_dir_back, scene_id, label_image_name + '.png')
        
        # Retrieve the pose of the sample.
        # 获取样本的位姿。
        R = np.array(info_['cam_R_m2c']).reshape((3, 3))
        t = np.array(info_['cam_t_m2c']).reshape((3, 1))
        RT = [R, t]
        cam_K = np.array(info_['cam_K']).reshape((3, 3))
        fx, fy, cx, cy = cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]
        
        # Render the front depth map.
        # 渲染正面的深度图。
        self.renderer_vispy.render_object(0, R, t, fx, fy, cx, cy, draw_back = False)
        depth = self.renderer_vispy.depth
        pose_4 = np.eye(4)
        pose_4[:3,:3] = R
        pose_4[:3,3] = t[:,0]
        model_info_obj = self.model_info_obj
        
        # Correct the rotation deviation caused by translation.
        # 修正由位移引起的旋转偏差。
        if 'symmetries_discrete' in model_info_obj:
            pose_4_re = self.pnp_solve_re_4p(pose_4, depth, fx, fy, cx, cy)
            RT[0], RT[1] = self.modified_sym_Rt(R, t, model_info_obj, e_rot = pose_4_re[:3,:3])
            
        # Compute the front label map based on the depth map, mask, and corrected 6D pose.
        # 根据深度图、掩膜图以及修正后的 6D 位姿计算正面标签图。
        mask_n = depth.copy()
        mask_n[mask_n>0] = 255
        mask_n = mask_n.astype(np.uint8)
        grid_row, grid_column = np.nonzero(mask_n.astype(np.int64)>0)
        p2dxy = np.empty((len(grid_row), 2))
        p2dxy[:, 1] = grid_row
        p2dxy[:, 0] = grid_column
        p2dxy[:, 0] -= cx
        p2dxy[:, 1] -= cy
        p2dxy[:, 0] /= fx
        p2dxy[:, 1] /= fy
        p2z = depth[mask_n>0]
        p2z = p2z.reshape((-1,1)).repeat(2, axis=1)
        T = RT[1]
        p2dxy *= p2z
        p2dxy[:, 0] -= T[0]
        p2dxy[:, 1] -= T[1]
        p2z -= T[2]
        p3xyz = np.empty((len(grid_row), 3))
        p3xyz[:, :2] = p2dxy
        p3xyz[:, 2] = p2z[:,0]
        R = RT[0]
        p3xyz = np.dot(p3xyz, R)
        p3xyz[:,0] = (p3xyz[:,0]-self.min_xyz[0]) / self.div_v[0]
        p3xyz[:,1] = (p3xyz[:,1]-self.min_xyz[1]) / self.div_v[1]
        p3xyz[:,2] = (p3xyz[:,2]-self.min_xyz[2]) / self.div_v[2]
        p3xyz[p3xyz>1] = 1
        p3xyz[p3xyz<0] = 0
        p3xyz = np.round(p3xyz * 255).astype(np.uint8)
        rgb_xyz = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgb_xyz[:,:, 0][mask_n>0] = p3xyz[:,0]
        rgb_xyz[:,:, 1][mask_n>0] = p3xyz[:,1]
        rgb_xyz[:,:, 2][mask_n>0] = p3xyz[:,2]
        cv2.imwrite(front_label_image_path, rgb_xyz)
        
        # Render the back depth map and generate the back label map.
        # 渲染背面的深度图并生成背面标签图。
        self.renderer_vispy.render_object(0, R, t, fx, fy, cx, cy, draw_back = True)
        depth = self.renderer_vispy.depth
        grid_row, grid_column = np.nonzero(mask_n.astype(np.int64)>0)
        p2dxy = np.empty((len(grid_row), 2))
        p2dxy[:, 1] = grid_row
        p2dxy[:, 0] = grid_column
        p2dxy[:, 0] -= cx
        p2dxy[:, 1] -= cy
        p2dxy[:, 0] /= fx
        p2dxy[:, 1] /= fy
        p2z = depth[mask_n>0]
        p2z = p2z.reshape((-1,1)).repeat(2, axis=1)
        T = RT[1]
        p2dxy *= p2z
        p2dxy[:, 0] -= T[0]
        p2dxy[:, 1] -= T[1]
        p2z -= T[2]
        p3xyz = np.empty((len(grid_row), 3))
        p3xyz[:, :2] = p2dxy
        p3xyz[:, 2] = p2z[:,0]
        R = RT[0]
        p3xyz = np.dot(p3xyz, R)
        p3xyz[:,0] = (p3xyz[:,0]-self.min_xyz[0]) / self.div_v[0]
        p3xyz[:,1] = (p3xyz[:,1]-self.min_xyz[1]) / self.div_v[1]
        p3xyz[:,2] = (p3xyz[:,2]-self.min_xyz[2]) / self.div_v[2]
        p3xyz[p3xyz>1] = 1
        p3xyz[p3xyz<0] = 0
        p3xyz = np.round(p3xyz * 255).astype(np.uint8)
        rgb_xyz = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgb_xyz[:,:, 0][mask_n>0] = p3xyz[:,0]
        rgb_xyz[:,:, 1][mask_n>0] = p3xyz[:,1]
        rgb_xyz[:,:, 2][mask_n>0] = p3xyz[:,2]
        cv2.imwrite(back_label_image_path, rgb_xyz)
        
        return 1

    def pnp_solve_re_4p(self, pose_, depth, fx, fy, cx, cy):
        '''
        ---
        ---
        Correct the rotation deviation caused by translation.  
        ---
        ---
        According to imaging geometry, when translation changes but rotation remains constant,
        an object may appear visually rotated under a fixed camera viewpoint.  
        To correct this translation-induced apparent rotation, the object’s 3D coordinates are computed from the rendered depth map,
        and rotation refinement is performed using RANSAC PnP.
    
        Args:
            - pose_: The original 6D pose (4×4) before correction.  
            - depth: The rendered depth map corresponding to the 6D pose.  
            - fx: Focal length along the x-axis.  
            - fy: Focal length along the y-axis.  
            - cx: Principal point offset along the x-axis.  
            - cy: Principal point offset along the y-axis.  

        Returns:
            - pnp_pose: The corrected 6D pose.  

        This idea originates from our observation in keypoint-based pose estimation:  
        for some objects, the 2D projections of keypoints change significantly with translation  
        even when the rotation remains constant.  
        Such changes have little impact on non-symmetric objects  
        but can strongly affect pose estimation for rotationally symmetric objects.  

        Reference:  
        Yulin Wang, Hongli Li, and Chen Luo.  
        _Object Pose Estimation Based on Multi-precision Vectors and Seg-Driven PnP_,  
        International Journal of Computer Vision (IJCV), 2025, 133: 2620–2634.
        ---
        ---
        修正由位移引起的旋转偏差。  
        ---
        ---
        依据相机成像原理，当物体旋转不变而发生位移时，
        在固定视角下物体会出现“视觉上的旋转”。  
        为修正这种由位移引起的视觉旋转，我们根据渲染得到的深度图计算物体的 3D 坐标，
        并使用 RANSAC PnP 对旋转进行校正。
    
        参数:
            - pose_: 修正前的 6D 位姿（4×4）。  
            - depth: 与该位姿对应的渲染深度图。  
            - fx: 相机在 x 轴方向的焦距。  
            - fy: 相机在 y 轴方向的焦距。  
            - cx: 相机主点在 x 轴方向的偏移量。  
            - cy: 相机主点在 y 轴方向的偏移量。  

        返回:
            - pnp_pose: 修正后的 6D 位姿。  

        这一思路源自我们在研究基于关键点检测的位姿估计算法时的观察：  
        对于某些物体，当旋转保持不变而发生位移时，关键点的 2D 投影会发生明显变化。  
        这种变化对非旋转对称物体影响较小，但会显著影响旋转对称物体的位姿估计结果。  

        参考文献：  
        Yulin Wang, Hongli Li, and Chen Luo.  
        _Object Pose Estimation Based on Multi-precision Vectors and Seg-Driven PnP_,  
        International Journal of Computer Vision (IJCV), 2025, 133: 2620–2634.
        '''
        
        def perspective_unknown_kp_2D(kp, RT ):
            keypoints = kp
            Rot_m, T_m = RT
            keypoints_dot_mat = np.array(keypoints)
            mat = Rot_m
            for i in range(keypoints_dot_mat.shape[0]):
                keypoints_dot_mat[i]=np.dot(mat, keypoints_dot_mat[i])
            for i in range(keypoints_dot_mat.shape[0]):
                keypoints_dot_mat[i][2] = keypoints_dot_mat[i][2] + T_m[2]
                keypoints_dot_mat[i][0] = (keypoints_dot_mat[i][0] + T_m[0]) / keypoints_dot_mat[i][2] * fx + cx #- c_x
                keypoints_dot_mat[i][1] = (keypoints_dot_mat[i][1] + T_m[1]) / keypoints_dot_mat[i][2] * fy + cy #- c_y
            keypoints_xy = np.array(keypoints_dot_mat[:,:2])
            keypoints_xyz = np.array(keypoints_dot_mat[:,:3])
            return keypoints_xy, keypoints_xyz, keypoints

        def depth2kp(depth, RT):
            mask = depth.copy()
            mask[mask>0] = 255
            mask = mask.astype(np.uint8)
            grid_row, grid_column = np.nonzero(mask.astype(np.int64)>0)
            p2dxy = np.empty((len(grid_row), 2))
            p2dxy[:, 1] = grid_row
            p2dxy[:, 0] = grid_column
            p2dxy_r = p2dxy.copy()
            p2dxy[:, 0] -= cx
            p2dxy[:, 1] -= cy
            p2dxy[:, 0] /= fx
            p2dxy[:, 1] /= fy
            p2z = depth[mask>0]
            p2z = p2z.reshape((-1,1)).repeat(2, axis=1)
            T = RT[1]
            p2dxy *= p2z
            p2dxy[:, 0] -= T[0]
            p2dxy[:, 1] -= T[1]
            p2z -= T[2]
            p3xyz = np.empty((len(grid_row), 3))
            p3xyz[:, :2] = p2dxy
            p3xyz[:, 2] = p2z[:,0]
            R = RT[0]
            p3xyz = np.dot(p3xyz, R)
            return p2dxy_r, p3xyz

        def solvePnP(cam, image_points, object_points, return_inliers=False, ransac_iter=5000):
            dist_coeffs = np.zeros((5, 1)) 
            inliers = None
            if image_points.shape[0] < 4:
                pose = np.eye(4)
                inliers = []
            else:
                object_points = np.expand_dims(object_points, 1)
                image_points = np.expand_dims(image_points, 1)
                success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points.astype(float), cam.astype(float),
                                                                                dist_coeffs, iterationsCount=ransac_iter,
                                                                                reprojectionError=1.5,
                                                                                confidence = 0.9995,)[:4]
                pose = np.eye(4)
                if success:
                    pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
                    pose[:3, 3] = np.squeeze(translation_vector)
            if return_inliers:
                return pose, len(inliers)
            else:
                return pose
        
        p2dxy_r, p3xyz = depth2kp(depth, [pose_[:3,:3], pose_[:3,3].reshape((3))])
        kp_ = np.array([[0,0,0]])
        kp_xy, _, _ = perspective_unknown_kp_2D(kp_, [pose_[:3, :3],pose_[:3, 3]], )
        T_0 = pose_[:3, 3].copy()
        T_0[:2] = 0
        kp_xy0, _, _ = perspective_unknown_kp_2D(kp_, [pose_[:3, :3],T_0], )
        dT_ = kp_xy0[0] - kp_xy[0]
        p2dxy_r[:, 0] += dT_[0]
        p2dxy_r[:, 1] += dT_[1]
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        pnp_pose = solvePnP(K, p2dxy_r, p3xyz, False)
        return pnp_pose

    def modified_sym_Rt(self, rot_pose, tra_pose, model_info, e_rot=None):
        '''
        ---
        ---
        Eliminate the ambiguity of 6D poses for rotationally symmetric objects.  
        ---
        ---
        Based on rotational symmetry priors, this function computes all possible ground-truth 6D pose labels  
        and selects the one whose rotation matrix has the smallest L2 distance from the identity matrix.  
        During the L2 distance computation, the rotation corrected by RANSAC PnP (`e_rot`) is used  
        to remove the apparent rotation deviation caused by translation.  

        Args:
            - rot_pose: Ground-truth rotation matrix.  
            - tra_pose: Ground-truth translation vector.  
            - model_info: Object information containing rotational symmetry priors.  
            - e_rot: Rotation matrix corrected by RANSAC PnP.  

        Returns:
            - rot_pose: Rotation matrix with rotational symmetry ambiguity removed.  
            - tra_pose: Translation vector with rotational symmetry ambiguity removed.  
        This idea is inspired by **ZebraPose**: https://github.com/suyz526/ZebraPose
        
        ---
        ---
        消除旋转对称物体在 6D 位姿中的歧义性。  
        ---
        ---
        基于旋转对称先验，函数会计算所有潜在的真值 6D 位姿标签，  
        并从中筛选出与单位旋转矩阵的 L2 距离最小的 6D 位姿。  
        在计算 L2 距离时，使用经 RANSAC PnP 校正后的旋转 (`e_rot`)，  
        以消除由位移引起的视觉旋转偏差。  

        参数:
            - rot_pose: 真值标签的旋转矩阵。  
            - tra_pose: 真值标签的位移向量。  
            - model_info: 物体信息，包含旋转对称先验。  
            - e_rot: 通过 RANSAC PnP 校正得到的旋转矩阵。  

        返回:
            - rot_pose: 消除了旋转对称歧义性的旋转矩阵。  
            - tra_pose: 消除了旋转对称歧义性的位移向量。  

        这一思路来源于 **ZebraPose**：https://github.com/suyz526/ZebraPose
        '''


        trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({'R': R, 't': t})
        best_R = None
        best_t = None
        froebenius_norm = 1e8
        for sym in trans_disc:
            R = sym['R']
            t = sym['t']
            if e_rot is None:
                tmp_froebenius_norm = np.linalg.norm(rot_pose.dot(R)-np.eye(3))
            else:
                tmp_froebenius_norm = np.linalg.norm(e_rot.dot(R)-np.eye(3))
            if tmp_froebenius_norm < froebenius_norm:
                froebenius_norm = tmp_froebenius_norm
                best_R = R
                best_t = t
        tra_pose = rot_pose.dot(best_t) + tra_pose
        rot_pose = rot_pose.dot(best_R)
        return rot_pose, tra_pose
    
    def update_obj_id(self, obj_id, obj_path):
        '''
        Update the currently loaded object.  
        `obj_id` is the object's ID, and `obj_path` is the path to its 3D model.

        更新当前加载的物体。  
        `obj_id` 为物体的 ID，`obj_path` 为该物体的 3D 模型路径。
        '''

        self.current_obj_id = obj_id
        self.current_obj_path = obj_path
        self.nSamples = len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])
        self.model_info_obj = copy.deepcopy(self.bop_dataset_item.model_info[str(self.current_obj_id)])
        
        if 'symmetries_continuous' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_continuous']):
                if "axis" in self.model_info_obj['symmetries_continuous'][0]:
                    self.model_info_obj['symmetries_discrete'] = misc.get_symmetry_transformations(self.model_info_obj, np.pi / 180)
                   
            self.model_info_obj.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_discrete']) == 0:
                self.model_info_obj.pop("symmetries_discrete")
        return
    
    def worker_init_fn(self, worker_id):
        '''
        Create a VisPy renderer for each process.
        
        为每个进程创建一个 VisPy 渲染器。
        '''
        
        print(worker_id)
        self.worker_id = worker_id
        self.img_shape = cv2.imread(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][0]['rgb']).shape[:2]
        self.renderer_vispy = renderer.create_renderer(self.img_shape[1], self.img_shape[0], 'vispy', mode='depth', shading='flat')
        self.renderer_vispy.add_object(0, self.current_obj_path)
        vertices = inout.load_ply(self.current_obj_path)["pts"]
        div_v = np.max(vertices,axis=0) - np.min(vertices,axis=0)
        self.div_v = div_v
        self.min_xyz = np.min(vertices,axis=0)

class train_bop_dataset_back_front(Dataset):
    '''
    BOP data loader for training.

    用于训练的 BOP 数据加载器。
    '''

    def __init__(self, bop_dataset_item : bop_dataset, folder_name, padding_ratio=1.5, crop_size_img=256, aug_op = 'imgaug', ):
        '''
        ---
        ---
        Initialization function.  
        ---
        ---
        Args:
            - bop_dataset_item: An instance of the `bop_dataset` type.  
            - folder_name: The name of the folder in the dataset from which data will be loaded.  
            - padding_ratio: The scaling factor for the bounding box, default is 1.5.  
            - crop_size_img: The side length of the square image cropped based on the bounding box, default is 256.  
            - aug_op: The type of data augmenter, default is `imgaug`.  

        Returns:
            - None
        ---
        ---
        初始化函数。  
        ---
        ---
        参数:
            - bop_dataset_item: `bop_dataset` 类型的数据集实例。  
            - folder_name: 数据集中需要加载数据的文件夹名称。  
            - padding_ratio: 包围盒的缩放比例，默认值为 1.5。  
            - crop_size_img: 基于包围盒裁切出的正方形图像边长，默认值为 256。  
            - aug_op: 增强器类型，默认使用 `imgaug`。  

        返回:
            - 无
        '''

        self.bop_dataset_item = bop_dataset_item
        self.dataset_info = bop_dataset_item.load_folder(folder_name, vis = 0.2)
        self.folder_name = folder_name
        self.nSamples = 0
        self.aug_op = aug_op
        self.padding_ratio = padding_ratio
        self.crop_size_img = crop_size_img
        self.crop_size_gt = int(crop_size_img / 2)
        if self.dataset_info is None:return
        target_dir_front = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_front')
        try:os.mkdir(target_dir_front)
        except:1
        self.target_dir_front = target_dir_front
        target_dir_back = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_back')
        try:os.mkdir(target_dir_back)
        except:1
        self.target_dir_back = target_dir_back
        for scene_path_i in self.dataset_info['scene_path_list']:
            try:os.mkdir(os.path.join(self.target_dir_front, os.path.basename(scene_path_i)))
            except:1
            try:os.mkdir(os.path.join(self.target_dir_back, os.path.basename(scene_path_i)))
            except:1
        self.composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        pass
    
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        '''
        ---
        ---
        Get data.  
        ---
        ---
        Args:
            - index: The index of the sample.  

        Returns:
            - rgb_c: The RGB image data.  
            - mask_vis_c: The visible mask of the object.  
            - GT_Front_hcce: Hierarchical code obtained by encoding the object's front 3D coordinates using HCCE.  
            - GT_Back_hcce: Hierarchical code obtained by encoding the object's back 3D coordinates using HCCE.  
        ---
        ---
        获取数据。  
        ---
        ---
        参数:
            - index: 样本的序号。  

        返回:
            - rgb_c: RGB 图像数据。  
            - mask_vis_c: 物体的可见掩膜。  
            - GT_Front_hcce: 物体正面 3D 坐标经 HCCE 编码后的层次化代码。  
            - GT_Back_hcce: 物体背面 3D 坐标经 HCCE 编码后的层次化代码。
        '''

        info_ = self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][index]
        rgb = cv2.imread(info_['rgb'])
        mask_vis = cv2.imread(info_['mask_visib_path'], 0)
        label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
        front_label_image_path = os.path.join(self.target_dir_front, info_['scene'], label_image_name + '.png')
        back_label_image_path = os.path.join(self.target_dir_back, info_['scene'], label_image_name + '.png')
        GT_Front = cv2.imread(front_label_image_path)
        GT_Back = cv2.imread(back_label_image_path)
        if GT_Front is None: GT_Front = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if GT_Back is None: GT_Back = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if self.aug_op == 'imgaug': rgb = self.apply_augmentation(rgb)
        Bbox = aug_square_fp32(np.array(info_['bbox_visib']), padding_ratio=self.padding_ratio)
        rgb_c = crop_square_resize(rgb, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR)
        mask_vis_c = crop_square_resize(mask_vis, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_c = crop_square_resize(GT_Front, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_hcce = self.hcce_encode(GT_Front_c)
        GT_Back_c = crop_square_resize(GT_Back, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Back_hcce = self.hcce_encode(GT_Back_c)
        rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce = self.preprocess(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)
        return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce

    def hcce_encode(self, code_img, iteration=8):
        '''
        ---
        ---
        HCCE Encoding.  
        ---
        ---
        Args:
            - code_img: The 3D coordinate map.  

        Returns:
            - check_hcce_images: The hierarchical codes obtained after HCCE encoding.  
        ---
        ---
        HCCE 编码。  
        ---
        ---
        参数:
            - code_img: 3D 坐标图。  

        返回:
            - check_hcce_images: 经 HCCE 编码得到的层次化代码。
        '''

        code_img = code_img.copy()
        
        code_img = [code_img[:, :, 0], code_img[:, :, 1], code_img[:, :, 2]]
        hcce_images = np.zeros((code_img[0].shape[0], code_img[0].shape[1], iteration * 3))
        for i in range(iteration):
            temp1 = np.array(code_img[0] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i] = temp1
            temp1 = np.array(code_img[1] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration] = temp1
            temp1 = np.array(code_img[2] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration*2] = temp1
        check_hcce_images = hcce_images.copy()
        k_ = iteration
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1].copy()
            temp[hcce_images[:,:,i] >= 0.5] = -temp[hcce_images[:,:,i] >= 0.5] + 1
            check_hcce_images[:,:,i+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_].copy()
            temp[hcce_images[:,:,i+k_] >= 0.5] = -temp[hcce_images[:,:,i+k_] >= 0.5] + 1
            check_hcce_images[:,:,i+k_+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_*2].copy()
            temp[hcce_images[:,:,i+k_*2] >= 0.5] = -temp[hcce_images[:,:,i+k_*2] >= 0.5] + 1
            check_hcce_images[:,:,i+k_*2+1]=temp
        
        
        return check_hcce_images

    def update_obj_id(self, obj_id, obj_path):
        '''
        Update the currently loaded object.  
        `obj_id` is the object's ID, and `obj_path` is the path to its 3D model.

        更新当前加载的物体。  
        `obj_id` 为物体的 ID，`obj_path` 为该物体的 3D 模型路径。
        '''
        
        self.current_obj_id = obj_id
        self.current_obj_path = obj_path
        self.nSamples = len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])
        self.model_info_obj = copy.deepcopy(self.bop_dataset_item.model_info[str(self.current_obj_id)])
        
        if 'symmetries_continuous' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_continuous']):
                if "axis" in self.model_info_obj['symmetries_continuous'][0]:
                    self.model_info_obj['symmetries_discrete'] = misc.get_symmetry_transformations(self.model_info_obj, np.pi / 180)
                   
            self.model_info_obj.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_discrete']) == 0:
                self.model_info_obj.pop("symmetries_discrete")
        return

    def apply_augmentation(self, x):
        '''
        Random image augmentation strategy proposed by GDR-Net.  
        Project link: https://github.com/THU-DA-6D-Pose-Group/GDR-Net  

        GDR-Net 提出的随机图像增强策略。  
        项目链接：https://github.com/THU-DA-6D-Pose-Group/GDR-Net
        '''

        def build_augmentations_depth():
            augmentations = []
            augmentations.append(iaa.Sometimes(0.3, iaa.SaltAndPepper(0.05)))
            augmentations.append(iaa.Sometimes(0.2, iaa.MotionBlur(k=5)))
            augmentations = augmentations + [iaa.Sometimes(0.4, iaa.CoarseDropout( p=0.1, size_percent=0.05) ),
                                            iaa.Sometimes(0.5, iaa.GaussianBlur(np.random.rand())),
                                            iaa.Sometimes(0.5, iaa.Add((-20, 20), per_channel=0.3)),
                                            iaa.Sometimes(0.4, iaa.Invert(0.20, per_channel=True)),
                                            iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4), per_channel=0.8)),
                                            iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4))),
                                            iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.0), per_channel=0.3))
                                            ]
            image_augmentations=iaa.Sequential(augmentations, random_order = False)
            return image_augmentations
        self.augmentations = build_augmentations_depth()
        color_aug_prob = 0.8
        if np.random.rand() < color_aug_prob:
            x = self.augmentations.augment_image(x)
        return x
    
    def preprocess(self, rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce):

        rgb_c_pil = Image.fromarray(np.uint8(rgb_c)).convert('RGB')
        mask_vis_c = mask_vis_c / 255.
        mask_vis_c = torch.from_numpy(mask_vis_c).type(torch.float)
        GT_Front_hcce = torch.from_numpy(GT_Front_hcce).permute(2, 0, 1)
        GT_Back_hcce = torch.from_numpy(GT_Back_hcce).permute(2, 0, 1)
        return self.composed_transforms_img(rgb_c_pil), mask_vis_c, GT_Front_hcce, GT_Back_hcce

class test_bop_dataset_back_front(Dataset):
    
    '''
    BOP data loader for testing.

    用于测试的 BOP 数据加载器。
    '''
    
    def __init__(self, bop_dataset_item : bop_dataset, folder_name, bbox_2D = None, test_targets_bop19=None, bbox_2D_score_threshold = 0.0, padding_ratio=1.5, crop_size_img=256, ratio = 1.0 ):
        
        self.ratio = ratio
        self.bbox_2D = bbox_2D
        self.bop_dataset_item = bop_dataset_item
        self.dataset_info = bop_dataset_item.load_folder(folder_name, vis = 0.2)
        
        
        if test_targets_bop19 is not None:
            test_targets_bop19_dict = load_json2dict(test_targets_bop19)
            test_targets_bop19_dict_new = []
            for test_targets_bop19_dict_i in test_targets_bop19_dict:
                test_targets_bop19_dict_new.append({
                    'im_id' : test_targets_bop19_dict_i['im_id'],
                    'obj_id' : test_targets_bop19_dict_i['obj_id'],
                    'scene_id' : test_targets_bop19_dict_i['scene_id'],
                    })
        self.obj_info_w_bbox_2D = {}
        if bbox_2D is not None:
            bbox_2D_dict = load_json2dict(bbox_2D)
            for bbox_2D_i in bbox_2D_dict:
                if test_targets_bop19 is not None:
                    if {'im_id' : bbox_2D_i['image_id'], 'scene_id' : bbox_2D_i['scene_id'], 'obj_id' : bbox_2D_i['category_id']} not in test_targets_bop19_dict_new:
                        continue
                scene_id = str(bbox_2D_i['scene_id']).rjust(6, '0')
                image_id = str(bbox_2D_i['image_id']).rjust(6, '0')
                category_id = str(bbox_2D_i['category_id']).rjust(6, '0')
                img_info_i = self.dataset_info['img_info'][scene_id + '_' + image_id]
                if 'obj_'+category_id not in self.obj_info_w_bbox_2D:
                    self.obj_info_w_bbox_2D['obj_'+category_id] = []
                if bbox_2D_score_threshold < bbox_2D_i['score']:
                    obj_info_i = {
                        'scene' : scene_id,
                        'image' : image_id,
                        'rgb' : img_info_i['rgb'],
                        'depth' : img_info_i['depth'],
                        'bbox' : bbox_2D_i['bbox'],
                        'score' : bbox_2D_i['score'],
                        'cam_K' : img_info_i['cam_K'],
                        'depth_scale' : img_info_i['depth_scale'],
                        
                    }
                    self.obj_info_w_bbox_2D['obj_'+category_id].append(obj_info_i)
            self.dataset_info['obj_info_origin'] = self.dataset_info['obj_info']
            self.dataset_info['obj_info'] = self.obj_info_w_bbox_2D
        
        
        self.folder_name = folder_name
        self.nSamples = 0
        self.padding_ratio = padding_ratio
        self.crop_size_img = crop_size_img
        self.crop_size_gt = int(crop_size_img / 2)
        if self.dataset_info is None:return
        target_dir_front = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_front')
        try:os.mkdir(target_dir_front)
        except:1
        self.target_dir_front = target_dir_front
        target_dir_back = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_back')
        try:os.mkdir(target_dir_back)
        except:1
        self.target_dir_back = target_dir_back
        for scene_path_i in self.dataset_info['scene_path_list']:
            try:os.mkdir(os.path.join(self.target_dir_front, os.path.basename(scene_path_i)))
            except:1
            try:os.mkdir(os.path.join(self.target_dir_back, os.path.basename(scene_path_i)))
            except:1
        self.composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        pass
    
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        info_ = self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][index]
        cam_K = np.array(info_['cam_K']).reshape((3,3))
        if 'cam_R_m2c' in info_ and 'cam_t_m2c' in info_:
            cam_R_m2c = np.array(info_['cam_R_m2c']).reshape((3,3))
            cam_t_m2c = np.array(info_['cam_t_m2c']).reshape((3,1))
        else:
            cam_R_m2c = np.eye(3)
            cam_t_m2c = np.zeros((3, 1))
        rgb = cv2.imread(info_['rgb'])
        if 'mask_path' in info_: mask_vis = cv2.imread(info_['mask_visib_path'], 0)
        else: mask_vis = None
        if mask_vis is None: mask_vis = np.zeros((self.crop_size_gt, self.crop_size_gt))
        if 'mask_path' in info_:
            label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
            front_label_image_path = os.path.join(self.target_dir_front, info_['scene'], label_image_name + '.png')
            back_label_image_path = os.path.join(self.target_dir_back, info_['scene'], label_image_name + '.png')
            GT_Front = cv2.imread(front_label_image_path)
            GT_Back = cv2.imread(back_label_image_path)
        else:
            GT_Front, GT_Back = None, None
        if GT_Front is None: GT_Front = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if GT_Back is None: GT_Back = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if self.bbox_2D is not None:
            Bbox = pad_square_fp32(np.array(info_['bbox']), padding_ratio=self.padding_ratio)
        else:
            Bbox = pad_square_fp32(np.array(info_['bbox_visib']), padding_ratio=self.padding_ratio)
        # mask_vis_b = cv2.boundingRect(mask_vis)
        rgb_c = crop_square_resize(rgb, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR)
        mask_vis_c = crop_square_resize(mask_vis, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_c = crop_square_resize(GT_Front, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_hcce = self.hcce_encode(GT_Front_c)
        GT_Back_c = crop_square_resize(GT_Back, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Back_hcce = self.hcce_encode(GT_Back_c)
        rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce = self.preprocess(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)
        if self.bbox_2D is None: return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c, 
        else: return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c, int(info_['scene']), int(info_['image']), info_['score']

    def hcce_encode(self, code_img, iteration=8):
        code_img = [code_img[:, :, 0], code_img[:, :, 1], code_img[:, :, 2]]
        hcce_images = np.zeros((code_img[0].shape[0], code_img[0].shape[1], iteration * 3))
        for i in range(iteration):
            temp1 = np.array(code_img[0] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i] = temp1
            temp1 = np.array(code_img[1] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration] = temp1
            temp1 = np.array(code_img[2] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration*2] = temp1
        check_hcce_images = hcce_images.copy()
        k_ = iteration
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1].copy()
            temp[hcce_images[:,:,i] >= 0.5] = -temp[hcce_images[:,:,i] >= 0.5] + 1
            check_hcce_images[:,:,i+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_].copy()
            temp[hcce_images[:,:,i+k_] >= 0.5] = -temp[hcce_images[:,:,i+k_] >= 0.5] + 1
            check_hcce_images[:,:,i+k_+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_*2].copy()
            temp[hcce_images[:,:,i+k_*2] >= 0.5] = -temp[hcce_images[:,:,i+k_*2] >= 0.5] + 1
            check_hcce_images[:,:,i+k_*2+1]=temp
        return check_hcce_images

    def update_obj_id(self, obj_id, obj_path):
        
        
        self.current_obj_id = obj_id
        self.current_obj_path = obj_path
        
        self.nSamples = len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])
        
        if self.ratio != 1.0:
            len_ = int(self.ratio * len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])) + 0
            
            self.nSamples = len_
        
        self.model_info_obj = copy.deepcopy(self.bop_dataset_item.model_info[str(self.current_obj_id)])
        
        if 'symmetries_continuous' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_continuous']):
                if "axis" in self.model_info_obj['symmetries_continuous'][0]:
                    self.model_info_obj['symmetries_discrete'] = misc.get_symmetry_transformations(self.model_info_obj, np.pi / 180)
                   
            self.model_info_obj.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_discrete']) == 0:
                self.model_info_obj.pop("symmetries_discrete")
        return

    def preprocess(self, rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce):
        rgb_c_pil = Image.fromarray(np.uint8(rgb_c)).convert('RGB')
        mask_vis_c = mask_vis_c / 255.
        mask_vis_c = torch.from_numpy(mask_vis_c).type(torch.float)
        GT_Front_hcce = torch.from_numpy(GT_Front_hcce).permute(2, 0, 1)
        GT_Back_hcce = torch.from_numpy(GT_Back_hcce).permute(2, 0, 1)
        return self.composed_transforms_img(rgb_c_pil), mask_vis_c, GT_Front_hcce, GT_Back_hcce


