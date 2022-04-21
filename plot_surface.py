import open3d as o3d
import numpy as np

def parse_verts(vert_file, face_file, keep_normals=False):
    """
    Generate the vertices and faces (and optionally the normals) from .vert and .face files
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

def mesh_simplification(vert_file, face_file, out_name, vert_number=1e3, maximum_error=np.inf):
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

    ## create the Mesh
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),o3d.utility.Vector3iVector(faces))
    # compute normal for rendering
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    ## normal check
    # print(computed_normals.shape)
    # print(pre_normals.shape)
    # print(pre_normals[0])
    # print(computed_normals[0])
    # print(torch.dot(pre_normals[0], computed_normals[0]))
    ## visualization
    # o3d.visualization.draw_geometries([mesh])

    ## mesh simplification
    faces_num = len(faces) * (int(vert_number) / len(verts))
    mesh_reduced = mesh.simplify_quadric_decimation(int(faces_num),maximum_error=maximum_error)
    ## visualization
    # o3d.visualization.draw_geometries([mesh,mesh_reduced])
    ## save to ply
    o3d.io.write_triangle_mesh(f"{out_name}_mesh.ply", mesh_reduced, write_vertex_normals=True)

    print(f'vertices: {len(verts)} -> {len(mesh_reduced.vertices)} ')
    print(f'triangles: {len(faces)} -> {len(mesh_reduced.triangles)} ')

if __name__=="__main__":
    vert_path = "data/example_files/test.vert"
    faces_path = "data/example_files/test.face"

    mesh_simplification(vert_path, faces_path, "data/example_files/example", vert_number=1e3, maximum_error=5)

    verts, faces = parse_verts(vert_path, faces_path)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    # compute normal for rendering
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    mesh_reduced = o3d.io.read_triangle_mesh("data/example_files/example_mesh.ply")
    mesh_reduced.compute_triangle_normals()
    # mesh_reduced.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_reduced])