import blenderproc as bproc
import argparse
import os
import numpy as np
from tqdm import tqdm
import bpy

'''
chmod +x s2_p1_gen_pbr_data.sh
nohup ./s2_p1_gen_pbr_data.sh 0 0 42 > output_0_0.log 2>&1 &

'''

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('gpu_id', type=int, help='输入的参数')
args = parser.parse_args()
gpu_id = int(args.gpu_id)
current_dir = os.path.abspath(os.getcwd())
dataset_name = os.path.basename(current_dir) 
bop_parent_path = os.path.dirname(current_dir)

print('-*' * 10)
print('-*' * 10)
print('bop_parent_path', bop_parent_path)
print('dataset_name', dataset_name)
print('-*' * 10)
print('-*' * 10)

cc_textures_path = 'cc0textures'

bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
num_scenes = (50 * 1) 

bproc.init()

bproc.loader.load_bop_intrinsics(bop_dataset_path = bop_dataset_path)

# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
''''''
cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)
''''''
# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.15, -0.15, 0.0], [-0.1, -0.1, 0.0])
    max = np.random.uniform([0.1, 0.1, 0.4], [0.15, 0.15, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.set_render_devices(desired_gpu_device_type='CUDA', desired_gpu_ids = [gpu_id])
for i in tqdm(range(num_scenes)):
    
    
    rand_s = np.random.rand()
    
    # if rand_s > 0.8:
    #     idx_l = np.random.randint(0, 50, 2)
    #     obj_ids = []
    #     for _ in range(25):
    #         obj_ids.append(int(idx_l[0]))
    #         obj_ids.append(int(idx_l[1]))
    #         # obj_ids.append(int(idx_l[2]))
    # else:
        
    ''''''
    if rand_s > 0.5:
        idx_l = np.random.choice(np.arange(10), size=10, replace=False)
    else:
        idx_l = np.random.choice(np.arange(10), size=10, replace=True)
    obj_ids = []
    for idx_i in idx_l:
        obj_ids.append(int(idx_i) + 1)
    ''''''
    
    target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = bop_dataset_path, 
                                                mm2m = True,
                                                obj_ids = obj_ids,
                                                )
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
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    ''''''
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)
    ''''''
    
    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs, # + sampled_distractor_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)
    # bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < 20:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.3,
                                radius_max = 1.2,
                                elevation_min = 5,
                                elevation_max = 89)
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=int(0.6 * len(obj_ids)), replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()
            

    # Write data in bop format
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