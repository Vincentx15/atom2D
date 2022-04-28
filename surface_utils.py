import open3d as o3d
import os
import numpy as np
import subprocess

"""
In this file, we define functions to make the following transformations : 
PDB -> surfaces in .vert+.faces -> .ply file

We also define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to 
"""


def pdb_to_surf(pdb, out_name, density=1.):
    """
    Runs msms on the input PDB file and dumps the output in out_name
    :param pdb:
    :param out_name:
    :return:
    """
    # First get the xyzr file
    temp_xyzr_name = f"{out_name}_temp.xyzr"
    temp_log_name = f"{out_name}_msms.log"
    with open(temp_xyzr_name, "w") as f:
        cline = f"pdb_to_xyzr {pdb}"
        subprocess.run(cline.split(), stdout=f)
    cline = f"msms -if {temp_xyzr_name} -of {out_name} -density {density}"
    with open(temp_log_name, "w") as f:
        subprocess.run(cline.split(), stdout=f)
    os.remove(temp_xyzr_name)
    os.remove(temp_log_name)
    pass


def parse_verts(vert_file, face_file, keep_normals=False):
    """
    Generate the vertices and faces (and optionally the normals) from .vert and .face files generated by MSMS
    :param vert_file:
    :param face_file:
    :param keep_normals:
    :return:
    """
    with open(vert_file, 'r') as f:
        # Parse the file and ensure it looks sound
        lines = f.readlines()
        n_vert = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_vert

        # Parse the info to retrieve vertices and normals
        lines = [line.split() for line in no_header]
        lines = np.array(lines).astype(np.float)
        verts = lines[:, :3]
        if keep_normals:
            normals = lines[:, 3:6]

    with open(face_file, 'r') as f:
        # Parse the file and ensure it looks sound
        lines = f.readlines()
        n_faces = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_faces

        # Parse the lines and remove 1 to get zero based indexing
        lines = [line.split() for line in no_header]
        lines = np.array(lines).astype(np.int)
        faces = lines[:, :3]
        faces -= 1

    if keep_normals:
        return verts, faces, keep_normals
    else:
        return verts, faces


def pdb_to_surf_with_min(pdb, out_name, min_number=128):
    """
    This function is useful to retrieve at least min_number vertices, which is useful for later use in DiffNets
    :param pdb:
    :param out_name:
    :param min_number:
    :return:
    """
    vert_file = out_name + '.vert'
    face_file = out_name + '.face'
    number_of_vertices = 0
    density = 1.
    while number_of_vertices < min_number:
        pdb_to_surf(pdb=pdb, out_name=out_name, density=density)
        verts, faces = parse_verts(vert_file=vert_file, face_file=face_file)
        number_of_vertices = len(verts)
        # print(f'After {density} try, number is {number_of_vertices}')
        density += 1


def mesh_simplification(vert_file, face_file, out_name, vert_number=1000, maximum_error=np.inf):
    """
    Generate a .ply of a simplified mesh from .vert and .face files
    :param vert_file:
    :param face_file:
    :param out_name:
    :param vert_number:
    :param maximum_error:
    :return:
    """

    verts, faces = parse_verts(vert_file, face_file)

    # create the Mesh
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    faces_num = int(vert_number * len(faces) / len(verts))
    mesh_reduced = mesh.simplify_quadric_decimation(target_number_of_triangles=faces_num,
                                                    maximum_error=maximum_error)
    # Because we control the faces and not the vertices, it could be that we go below the number
    #  For instance, if we start with 129 vertices, asking to remove a few faces could lead to vertices going below 128
    if len(mesh_reduced.vertices) < vert_number:
        missing = max(vert_number - len(mesh_reduced.vertices), 100)
        mesh_reduced = mesh.simplify_quadric_decimation(target_number_of_triangles=faces_num + missing,
                                                        maximum_error=maximum_error)
        # print(f'We start with {len(faces)} for a target of {faces_num}. '
        #       f'That is an initial {len(verts)}, and we are missing {missing} vertices. '
        #       f'Try now with a target {int(faces_num) + missing} instead and get only '
        #       f'{vert_number - len(mesh_reduced.vertices)} missing')
    assert len(mesh_reduced.vertices) >= vert_number
    # print(f'Simplified from {len(verts)} to {len(mesh_reduced.vertices)}')
    # visualization, you need to compute normals for rendering
    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh,mesh_reduced])

    # save to ply
    o3d.io.write_triangle_mesh(f"{out_name}_mesh.ply", mesh_reduced, write_vertex_normals=True)

    # print(f'vertices: {len(verts)} -> {len(mesh_reduced.vertices)} ')
    # print(f'triangles: {len(faces)} -> {len(mesh_reduced.triangles)} ')


def read_face_and_triangles(ply_file):
    mesh = o3d.io.read_triangle_mesh(filename=ply_file)
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.triangles, np.int64)
    return vertices, faces


if __name__ == "__main__":
    vert_file = "data/example_files/test.vert"
    faces_file = "data/example_files/test.face"

    mesh_simplification(vert_file, faces_file, "data/example_files/example", vert_number=1000, maximum_error=5)

    verts, faces = parse_verts(vert_file, faces_file)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    # compute normal for rendering
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    mesh_reduced = o3d.io.read_triangle_mesh("data/example_files/example_mesh.ply")
    mesh_reduced.compute_triangle_normals()
    # mesh_reduced.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_reduced])
