# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
s2_p1_gen_pbr_data.py is used to generate PBR data.  
The original script is adapted from BlenderProc2.  
Project link: https://github.com/DLR-RM/BlenderProc

Usage:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    
Arguments (example: s2_p1_gen_pbr_data.sh 0 42 ... ):
    Arg 1 (`GPU_ID`): GPU index. Set to 0 for the first GPU.
    Arg 2 (`SCENE_NUM`): Number of scenes; total images generated = 1000 * 42.
    Arg 3 (`cc0textures`): Path to the cc0textures material library.
    Arg 4 (`dataset_path`): Path to the dataset.
    Arg 4 (`s2_p1_gen_pbr_data`): Path to the s2_p1_gen_pbr_data.py script.
    
------------------------------------------------------    

s2_p1_gen_pbr_data.py 用于生成 PBR 数据，原始脚本改编自 BlenderProc2。  
项目链接: https://github.com/DLR-RM/BlenderProc

运行方法:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

参数说明 (以 s2_p1_gen_pbr_data.sh 0 42 ... 为例):
    参数 1 (`GPU_ID`): GPU 的编号。设置为 0 表示使用第一块显卡。
    参数 2 (`SCENE_NUM`): 场景数量，对应生成的图像数 = 1000 * 42
    参数 3 (`cc0textures`): cc0textures 材质库的路径。
    参数 3 (`dataset_path`): 数据集的路径。
    参数 3 (`s2_p1_gen_pbr_data`): s2_p1_gen_pbr_data.py的路径。
'''

import os
import bpy
import argparse
import blenderproc as bproc
import numpy as np
from tqdm import tqdm
from kasal.utils.io_json import load_json2dict, write_dict2json


if __name__ == '__main__':
    
    # Retrieve the GPU ID.
    # 获取 GPU 的编号。
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('gpu_id', type=int, help='')
    parser.add_argument('cc0textures', type=str, help='')
    args = parser.parse_args()
    gpu_id = int(args.gpu_id)

    # Retrieve the folder path of the current dataset.
    # 获取当前数据集的文件夹路径。
    current_dir = os.path.abspath(os.getcwd())

    # Retrieve the name of the dataset.
    # 获取数据集的名称。
    dataset_name = os.path.basename(current_dir) 

    # BlenderProc requires the path to the parent directory of the dataset.
    # BlenderProc 需要传入数据集的父级目录路径。
    bop_parent_path = os.path.dirname(current_dir)

    # Load the 3D model information of the dataset.
    # 加载数据集的 3D 模型信息。
    models_info = load_json2dict(os.path.join(current_dir, 'models', 'models_info.json'))

    if not os.path.exists(os.path.join(current_dir, 'camera.json')):
        write_dict2json(os.path.join(current_dir, 'camera.json'), 
                            {
                            "cx": 325.2611083984375,
                            "cy": 242.04899588216654,
                            "depth_scale": 0.1,
                            "fx": 572.411363389757,
                            "fy": 573.5704328585578,
                            "height": 480,
                            "width": 640
                            }
                        )
    
    # Retrieve the list of 3D model IDs from the dataset.
    # 获取数据集中 3D 模型的 ID 列表。
    models_ids = []
    for key in models_info:
        models_ids.append(int(key))
    models_ids = np.array(models_ids)

    # Print the parent path and name of the dataset.
    # 打印数据集的父级路径和名称。
    print('-*' * 10)
    print('-*' * 10)
    print('bop_parent_path', bop_parent_path)
    print('dataset_name', dataset_name)
    print('-*' * 10)
    print('-*' * 10)

    # Retrieve the path to the cc0textures assets.
    # 获取 cc0textures 的路径。
    cc_textures_path = args.cc0textures

    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    num_scenes = (50 * 1) 

    # Create the rendering scene.
    # 创建渲染场景。
    bproc.init()
    bproc.loader.load_bop_intrinsics(bop_dataset_path = bop_dataset_path)
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    # Load all texture images from the cc_textures directory.
    # 加载 cc_textures 目录中的所有纹理图。
    if os.path.basename(cc_textures_path) == 'cc0textures-512':
        cc_textures = bproc.loader.load_512_ccmaterials(cc_textures_path, use_all_materials=True)
    else:
        cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)

    
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.15, -0.15, 0.0], [-0.1, -0.1, 0.0])
        max = np.random.uniform([0.1, 0.1, 0.4], [0.15, 0.15, 0.6])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    # Set the GPU ID.
    # 设置 GPU 的编号。
    bproc.renderer.set_render_devices(desired_gpu_device_type='CUDA', desired_gpu_ids = [gpu_id])

    for i in tqdm(range(num_scenes)):
        
        rand_s = np.random.rand()
        
        # Bin-picking selection mode.
        # bin-picking 的挑选模式。
        
        # idx_l = np.random.choice(models_ids, size=2, replace=True)
        # obj_ids = []
        # for _ in range(15):
        #     obj_ids.append(int(idx_l[0]))
        #     obj_ids.append(int(idx_l[1]))
        
        # Multi-class object picking mode.
        # 多类别物体的挑选模式。
        
        if rand_s > 0.5:
            idx_l = np.random.choice(models_ids, size=30, replace=True)
        else:
            idx_l = np.random.choice(models_ids, size=min(models_ids.shape[0],30), replace=False)
        obj_ids = []
        for idx_i in idx_l:
            obj_ids.append(int(idx_i))
        
        # Load objects into BlenderProc.
        # 将物体加载到 BlenderProc 中。
        target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = bop_dataset_path, 
                                                    mm2m = True,
                                                    obj_ids = obj_ids,
                                                    )
        
        # Set object materials and poses, then render 20 frames.
        # 设置物体的材质和位姿，并渲染 20 帧图像。
        
        for obj in (target_bop_objs):
            obj.set_shading_mode('auto')
            obj.hide(True)
        sampled_target_bop_objs = target_bop_objs
        for obj in (sampled_target_bop_objs):      
            mat = obj.get_materials()[0]     
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.hide(False)
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                elevation_min = 5, elevation_max = 89)
        light_point.set_location(location)
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                                sample_pose_func = sample_pose_func, 
                                max_tries = 1000)
                
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                        max_simulation_time=10,
                                                        check_object_interval=1,
                                                        substeps_per_frame = 20,
                                                        solver_iters=25)

        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

        cam_poses = 0
        while cam_poses < 20:
            location = bproc.sampler.shell(center = [0, 0, 0],
                                    radius_min = 0.3,
                                    radius_max = 1.2,
                                    elevation_min = 5,
                                    elevation_max = 89)
            poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=int(round(0.6 * len(obj_ids))), replace=False))
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1
        data = bproc.renderer.render()
        bproc.writer.write_bop(bop_parent_path,
                            target_objects = sampled_target_bop_objs,
                            dataset = dataset_name,
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"], 
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10)
        
        for obj in (sampled_target_bop_objs):    
            obj.disable_rigidbody()  
            obj.hide(True)
    pass