# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os, shutil
import pymeshlab as ml # pymeshlab==2023.12.post1
import numpy as np

def modify_ply_texture_filename(input_file, output_file, new_texture_name):
    ''' 
    ---
    ---
    Modify the texture file name in a PLY file.
    ---
    ---
    Args:
        - input_file: Path to the input PLY file.
        - output_file: Path to the output PLY file.
        - new_texture_name: New file name for the texture image.

    Returns:
        None
    ---
    ---

    修改 PLY 文件中纹理图的文件名。
    ---
    ---
    参数:
        - input_file: 输入 PLY 文件的路径。
        - output_file: 输出 PLY 文件的路径。
        - new_texture_name: 新的纹理图文件名。

    返回:
        无
    '''
    try:

        # Open the PLY file
        # 打开 PLY 文件
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Locate the TextureFile and replace the old texture file name with the new one
        # 查找 TextureFile，并将旧的纹理图名称替换为新的纹理图名称
        for i, line in enumerate(lines):
            if line.strip().startswith('comment TextureFile'):
                lines[i] = f'comment TextureFile {new_texture_name}\n'
                break
        with open(output_file, 'w') as f:
            f.writelines(lines)
    except FileNotFoundError:
        1

if __name__ == '__main__':

    # Provide the path to the input PLY file. 
    # The script generates a corresponding BOP-format PLY file based on the given obj_id. 
    # If the PLY file contains a texture image, a BOP-format texture file will also be generated.
    # 输入 PLY 文件路径，脚本会根据 obj_id 生成符合 BOP 格式的对应 PLY 文件。
    # 若 PLY 文件包含纹理图，脚本同时会生成符合 BOP 格式的纹理图文件。

    input_ply = 'raw-demo-models/multi-objs/board.ply'
    obj_id = 1
    output_ply = os.path.join(os.path.dirname(input_ply), 'obj_%s.ply'%str(obj_id).rjust(6, '0'))

    # Open the PLY file
    # 打开 PLY 文件
    mesh = ml.MeshSet()
    mesh.load_new_mesh(input_ply)

    # Compute the normal vectors of the 3D model vertices
    # 计算 3D 模型顶点的法向量
    mesh.compute_normal_per_vertex()
    mesh_c = mesh.current_mesh()

    # Get the vertex matrix of the 3D model
    # 获取 3D 模型的顶点矩阵
    mesh_vertex_matrix = mesh_c.vertex_matrix().copy()

    # Get the maximum and minimum values of the vertex matrix
    # 获取顶点矩阵的最大值和最小值
    vertex_min = np.min(mesh_vertex_matrix, axis = 0)
    vertex_max = np.max(mesh_vertex_matrix, axis = 0)

    # Compute the center of the 3D object model
    # 计算物体模型的中心
    vertex_center = (vertex_min + vertex_max) / 2

    # Align the object model center with the coordinate system origin based on the computed center
    # 根据计算得到的中心，将物体模型的中心对齐到坐标系原点
    mesh.compute_matrix_from_translation_rotation_scale(
        translationx = -vertex_center[0],
        translationy = -vertex_center[1],
        translationz = -vertex_center[2],
    )

    # Check whether the object model contains a texture map
    # 判断物体模型是否包含纹理图
    if mesh_c.texture_number() > 0:

        # If a texture map exists, copy and rename the texture file
        # 若存在纹理图，则拷贝纹理图并重命名
        if not os.path.exists(input_ply.replace('.ply', '.png')):
            shutil.copy2(input_ply.replace('.ply', '.png'), output_ply.replace('.ply', '.png'))
        
        # Convert wedge UVs to vertex UVs
        # 将 wedge UV 转换为 vertex UV
        if mesh_c.has_wedge_tex_coord():
            mesh.compute_texcoord_transfer_wedge_to_vertex()
        
        # Save the PLY file with the associated texture map
        # 保存带有纹理图的 PLY 文件
        mesh.save_current_mesh(output_ply,
                                binary = False, 
                                save_vertex_normal = True,
                                save_vertex_coord  = True,
                                save_wedge_texcoord = False
                                )
    else:
        # Save the PLY file without a texture map
        # 保存无纹理图的 PLY 文件
        mesh.save_current_mesh(output_ply,
                                binary = False, 
                                save_vertex_normal = True,
                                )
    # MeshLab cannot set the texture filename in PLY files
    # Use `modify_ply_texture_filename` to correct the texture filename in the PLY file
    # MeshLab 无法设置 PLY 文件中的纹理图名称
    # 需要使用 `modify_ply_texture_filename` 来修正 PLY 文件中的纹理图名称
    if mesh_c.texture_number() > 0:
        modify_ply_texture_filename(output_ply, output_ply, os.path.basename(output_ply.replace('.ply', '.png')))

    pass
