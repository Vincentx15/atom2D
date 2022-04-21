import open3d as o3d
import numpy as np

def plot_surf(vert_file, face_file):
    """
    Takes the output of msms and dump the diffusion nets operators in dump dir
    :param vert_file:
    :param face_file:
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

    verts = np.ascontiguousarray(verts)
    faces = np.ascontiguousarray(faces)
    # pre_normals = torch.from_numpy(np.ascontiguousarray(normals))
    # normals = None

    print(f'found {len(verts)} vertices')
    print(f'found {len(faces)} triangles')


    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),o3d.utility.Vector3iVector(faces))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("mesh.ply", mesh)

    mesh_reduced = mesh.simplify_quadric_decimation(len(faces)//60,maximum_error=5)
    o3d.visualization.draw_geometries([mesh,mesh_reduced])
    o3d.io.write_triangle_mesh("mesh_low.ply", mesh_reduced)
    print(f'found {len(mesh_reduced.vertices)} vertices')
    print(f'found {len(mesh_reduced.triangles)} triangles')

if __name__=="__main__":
    vert_path = "data/example_files/test.vert"
    faces_path = "data/example_files/test.face"
    plot_surf(vert_path,faces_path)