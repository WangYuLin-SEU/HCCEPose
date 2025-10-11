
# pymeshlab==2023.12.post1

import os, shutil
import pymeshlab as ml
import numpy as np

def modify_ply_texture_filename(input_file, output_file, new_texture_name):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('comment TextureFile'):
                lines[i] = f'comment TextureFile {new_texture_name}\n'
                break
        with open(output_file, 'w') as f:
            f.writelines(lines)
    except FileNotFoundError:
        1

if __name__ == '__main__':
    input_ply = 'raw-demo-models/obj_000001.ply'
    output_ply = os.path.join(os.path.dirname(input_ply), 'obj_000001.ply')


    mesh = ml.MeshSet()
    mesh.load_new_mesh(input_ply)

    mesh.compute_normal_per_vertex()

    mesh_c = mesh.current_mesh()
    mesh_vertex_matrix = mesh_c.vertex_matrix().copy()

    vertex_min = np.min(mesh_vertex_matrix, axis = 0)
    vertex_max = np.max(mesh_vertex_matrix, axis = 0)
    vertex_center = (vertex_min + vertex_max) / 2

    mesh.compute_matrix_from_translation_rotation_scale(
        translationx = -vertex_center[0],
        translationy = -vertex_center[1],
        translationz = -vertex_center[2],
    )

    if mesh_c.texture_number() > 0:
        if not os.path.exists(input_ply.replace('.ply', '.png')):
            shutil.copy2(input_ply.replace('.ply', '.png'), output_ply.replace('.ply', '.png'))
        if mesh_c.has_wedge_tex_coord():
            mesh.compute_texcoord_transfer_wedge_to_vertex()

        mesh.save_current_mesh(output_ply,
                                binary = False, 
                                save_vertex_normal = True,
                                save_vertex_coord  = True,
                                save_wedge_texcoord = False
                                )
    else:
        mesh.save_current_mesh(output_ply,
                                binary = False, 
                                save_vertex_normal = True,
                                )
        
    if mesh_c.texture_number() > 0:
        modify_ply_texture_filename(output_ply, output_ply, os.path.basename(output_ply.replace('.ply', '.png')))

    mesh.delete_current_mesh()
    
    pass
