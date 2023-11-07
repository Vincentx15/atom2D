import open3d as o3d
import os
import numpy as np
import subprocess
from pathlib import Path
import pymesh
import pandas as pd

"""
In this file, we define functions to make the following transformations :
PDB -> surfaces in .vert+.faces -> .ply file

We also define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to
"""


def pdb_to_surf_with_min(pdb, out_name, min_number=128, clean_temp=True):
    """
    This function is useful to retrieve at least min_number vertices, which is useful for later use in DiffNets
    :param pdb:
    :param out_name:
    :param min_number:
    :return:
    """

    number_of_vertices = 0
    density = 1.
    while number_of_vertices < min_number:
        verts, faces = pdb_to_surf(pdb=pdb, out_name=out_name, density=density, clean_temp=clean_temp)
        number_of_vertices = len(verts)
        density += 1

    return verts, faces


def pdb_to_surf(pdb, out_name, density=1., clean_temp=True):
    """
    Runs msms on the input PDB file and dumps the output in out_name
    :param pdb:
    :param out_name:
    :return:
    """
    vert_file = out_name + '.vert'
    face_file = out_name + '.face'

    # First get the xyzr file
    temp_xyzr_name = f"{out_name}_temp.xyzr"
    temp_log_name = f"{out_name}_msms.log"
    with open(temp_xyzr_name, "w") as f:
        cline = f"{Path.cwd()}/../executables/pdb_to_xyzr {pdb}"
        subprocess.run(cline.split(), stdout=f)

    # Then run msms on this file
    cline = f"{Path.cwd()}/../executables/msms -if {temp_xyzr_name} -of {out_name} -density {density}"
    with open(temp_log_name, "w") as f:
        result = subprocess.run(cline.split(), stdout=f, stderr=f, timeout=10)
    if result.returncode != 0:
        print(f"*** An error occurred while executing the command: {cline}, see log file for details. *** ")
        raise RuntimeError(f"MSMS failed with return code {result.returncode}")

    if clean_temp:
        os.remove(temp_xyzr_name)
    os.remove(temp_log_name)

    verts, faces = parse_verts(vert_file=vert_file, face_file=face_file)
    os.remove(vert_file)
    os.remove(face_file)

    return verts, faces


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
        lines = [line.split()[:6] for line in no_header]
        lines = np.array(lines).astype(np.float32)
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
        lines = np.array(lines).astype(np.int32)
        faces = lines[:, :3]
        faces -= 1

    if keep_normals:
        return verts, faces, normals
    else:
        return verts, faces


def mesh_simplification2(vert_file, face_file, out_ply, vert_number=2000):
    """
    Generate a .ply of a simplified mesh from .vert and .face files
    :param vert_file:
    :param face_file:
    :param out_ply:
    :param vert_number:
    :param maximum_error:
    :return:
    """

    verts, faces = parse_verts(vert_file, face_file)

    # create the Mesh
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    faces_num = int(vert_number * len(faces) / len(verts))
    mesh_reduced = mesh.simplify_quadric_decimation(target_number_of_triangles=faces_num)
    # Because we control the faces and not the vertices, it could be that we go below the number
    #  For instance, if we start with 129 vertices, asking to remove a few faces could lead to vertices going below 128
    if len(mesh_reduced.vertices) < vert_number:
        missing = max(vert_number - len(mesh_reduced.vertices), 100)
        mesh_reduced = mesh.simplify_quadric_decimation(target_number_of_triangles=faces_num + missing)
    assert len(mesh_reduced.vertices) >= vert_number

    # save to ply
    o3d.io.write_triangle_mesh(out_ply, mesh_reduced, write_vertex_normals=True)
    return mesh_reduced

    # visualization, you need to compute normals for rendering
    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh,mesh_reduced])
    # print(f'vertices: {len(verts)} -> {len(mesh_reduced.vertices)} ')
    # print(f'triangles: {len(faces)} -> {len(mesh_reduced.triangles)} ')


def mesh_simplification(verts, faces, out_ply, vert_number=2000):
    # remeshing to have a target number of vertices
    faces_num = int(vert_number * len(faces) / len(verts))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=faces_num)
    verts_out = np.asarray(mesh.vertices)
    faces_out = np.asarray(mesh.triangles)

    # cleaning the mesh
    mesh = pymesh.form_mesh(verts_out, faces_out)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 1E-6)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    num_verts = mesh.num_vertices
    iteration = 0
    while iteration < 10:
        mesh, _ = pymesh.collapse_short_edges(mesh, rel_threshold=0.1)
        mesh, _ = pymesh.remove_obtuse_triangles(mesh, 170.0, 100)
        if abs(mesh.num_vertices - num_verts) < 20:
            break
        num_verts = mesh.num_vertices
        iteration += 1

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    # mesh = pymesh.compute_outer_hull(mesh) (Is this needed?)
    mesh, _ = pymesh.remove_obtuse_triangles(mesh, 179.0, 100)
    mesh = remove_abnormal_triangles(mesh)
    mesh_py, _ = pymesh.remove_isolated_vertices(mesh)

    # save to ply
    verts, faces = np.array(mesh_py.vertices, dtype=np.float32), np.array(mesh_py.faces, dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    o3d.io.write_triangle_mesh(out_ply, mesh, write_vertex_normals=True)

    disconnected, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles = check_mesh_validity(mesh_py,
                                                                                                        check_triangles=True)
    is_valid_mesh = not (disconnected or has_isolated_verts or has_duplicate_verts or has_abnormal_triangles)

    if verts.shape[0] > 15000:
        raise ValueError(f'Too many vertices in the mesh: {verts.shape[0]}')

    return verts, faces, is_valid_mesh


def remove_abnormal_triangles(mesh):
    """Remove abnormal triangles (angles ~180 or ~0) in the mesh

    Returns:
        pymesh.Mesh, a new mesh with abnormal faces removed
    """
    verts = mesh.vertices
    faces = mesh.faces
    v1 = verts[faces[:, 0]]
    v2 = verts[faces[:, 1]]
    v3 = verts[faces[:, 2]]
    e1 = v3 - v2
    e2 = v1 - v3
    e3 = v2 - v1
    L1 = np.linalg.norm(e1, axis=1)
    L2 = np.linalg.norm(e2, axis=1)
    L3 = np.linalg.norm(e3, axis=1)
    cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
    cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
    cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)
    cos123 = np.concatenate((cos1.reshape(-1, 1),
                             cos2.reshape(-1, 1),
                             cos3.reshape(-1, 1)), axis=-1)
    valid_faces = np.where(np.all(1 - cos123 ** 2 > 1E-5, axis=-1))[0]
    faces_new = faces[valid_faces]

    return pymesh.form_mesh(verts, faces_new)


def check_mesh_validity(mesh, check_triangles=False):
    """Check if a mesh is valid by following criteria

    1) disconnected
    2) has isolated vertex
    3) face has duplicated vertices (same vertex on a face)
    4) has triangles with angle ~0 or ~180

    Returns
        four-tuple of bool: above criteria

    """
    mesh.enable_connectivity()
    verts, faces = mesh.vertices, mesh.faces

    # check if a manifold is all-connected using BFS
    visited = np.zeros(len(verts)).astype(bool)
    groups = []
    for ivert in range(len(verts)):
        if visited[ivert]:
            continue
        old_visited = visited.copy()
        queue = [ivert]
        visited[ivert] = True
        while queue:
            curr = queue.pop(0)
            for nbr in mesh.get_vertex_adjacent_vertices(curr):
                if not visited[nbr]:
                    queue.append(nbr)
                    visited[nbr] = True
        groups.append(np.where(np.logical_xor(old_visited, visited))[0])
    groups = sorted(groups, key=lambda x: len(x), reverse=True)
    assert sum(len(ig) for ig in groups) == sum(visited) == len(verts)
    disconnected = len(groups) > 1

    # check for isolated vertices
    valid_verts = np.unique(faces)
    has_isolated_verts = verts.shape[0] != len(valid_verts)

    # check for faces with duplicate vertices
    df = pd.DataFrame(faces)
    df = df[df.nunique(axis=1) == 3]
    has_duplicate_verts = df.shape[0] != mesh.num_faces

    # check for abnormal triangles
    if check_triangles:
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        L3 = np.linalg.norm(e3, axis=1)
        cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
        cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
        cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)
        cos123 = np.concatenate((cos1.reshape(-1, 1),
                                 cos2.reshape(-1, 1),
                                 cos3.reshape(-1, 1)), axis=-1)
        valid_faces = np.where(np.all(1 - cos123 ** 2 >= 1E-5, axis=-1))[0]
        has_abnormal_triangles = faces.shape[0] != len(valid_faces)
    else:
        has_abnormal_triangles = False

    return disconnected, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles


def get_vertices_and_triangles(mesh):
    """
    Just a small wrapper to retrieve directly the vertices and faces as np arrays with the right dtypes
    :param mesh:
    :return:
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


def read_vertices_and_triangles(ply_file):
    """
    Just a small wrapper to retrieve directly the vertices and faces as np arrays with the right dtypes
    :param ply_file:
    :return:
    """
    mesh = o3d.io.read_triangle_mesh(filename=ply_file)
    return get_vertices_and_triangles(mesh)


if __name__ == "__main__":
    # Check that msms gives the right output
    pdb = "../data/example_files/4kt3.pdb"
    outname = "../data/example_files/test"
    vert_file = f"{outname}.vert"
    faces_file = f"{outname}.face"
    pdb_to_surf(pdb, out_name=outname, density=1.)
    verts, faces = parse_verts(vert_file, faces_file)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    # compute normal for rendering
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    # Now build a surface with at least min vertex (lower bound)
    verts, faces = pdb_to_surf_with_min(pdb, out_name=outname, min_number=128)

    # Now simplify this into a coarser mesh (upper bound), and turn it into a corrected ply file
    ply_file = "../data/example_files/example_mesh.ply"
    mesh_simplification(verts=verts,
                        faces=faces,
                        out_ply=ply_file,
                        vert_number=1000)
    mesh_reduced = o3d.io.read_triangle_mesh(ply_file)
    mesh_reduced.compute_triangle_normals()
    o3d.visualization.draw_geometries([mesh_reduced])