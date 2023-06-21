import numpy as np
import os
import scipy
import torch
from base_nets import geometry, utils

"""
In this file, we define functions to make the following transformations :
.ply -> DiffNets operators in .npz format
"""


def get_operators(npz_path, verts, faces, k_eig=128, normals=None, recompute=False):
    """
    See documentation for get_operators() in diffnet.

    We remove the hashing util and add a filename for the npz instead.

    This is a wrapper that allows for even further caching checks (just file names) and
     legible file names (a protein chain is expected)
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = utils.toNP(verts)
    faces_np = utils.toNP(faces)

    if (np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    try:
        npzfile = np.load(npz_path, allow_pickle=True)
        cache_verts = npzfile["verts"]
        cache_faces = npzfile["faces"]
        cache_k_eig = npzfile["k_eig"].item()

        # If the cache doesn't match,  we're overwriting, or there aren't enough eigenvalues, or no L operator
        # just delete it; we'll create a new entry below more eigenvalues
        wrong_surface = (not np.allclose(verts, cache_verts)) or (not np.allclose(faces, cache_faces))

        redo_computation = wrong_surface or recompute
        if cache_k_eig < k_eig:
            # print("  overwriting cache --- not enough eigenvalues")
            redo_computation = True
        if "L_data" not in npzfile:
            # print("  overwriting cache --- entries are absent")
            redo_computation = True

        if redo_computation:
            os.remove(npz_path)

        if not redo_computation:
            def read_sp_mat(prefix):
                data = npzfile[prefix + "_data"]
                indices = npzfile[prefix + "_indices"]
                indptr = npzfile[prefix + "_indptr"]
                shape = npzfile[prefix + "_shape"]
                mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                return mat

            # This entry matches! Return it.
            frames = npzfile["frames"]
            mass = npzfile["mass"]
            L = read_sp_mat("L")
            evals = npzfile["evals"][:k_eig]
            evecs = npzfile["evecs"][:, :k_eig]
            gradX = read_sp_mat("gradX")
            gradY = read_sp_mat("gradY")

            frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
            mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
            L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
            evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
            evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
            gradX = utils.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
            gradY = utils.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)
            return frames, mass, L, evals, evecs, gradX, gradY

    except FileNotFoundError:
        pass
        # print("  cache miss -- constructing operators")

    except Exception:
        pass
        # print("unexpected error loading file: " + str(E))
        # print("-- constructing operators")

    # Recompute and cache it
    frames, mass, L, evals, evecs, gradX, gradY = geometry.compute_operators(verts, faces, k_eig, normals=normals)

    dtype_np = np.float32

    L_np = utils.sparse_torch_to_np(L).astype(dtype_np)
    gradX_np = utils.sparse_torch_to_np(gradX).astype(dtype_np)
    gradY_np = utils.sparse_torch_to_np(gradY).astype(dtype_np)
    np.savez(npz_path,
             verts=verts_np.astype(dtype_np),
             frames=utils.toNP(frames).astype(dtype_np),
             faces=faces_np,
             k_eig=k_eig,
             mass=utils.toNP(mass).astype(dtype_np),
             L_data=L_np.data.astype(dtype_np),
             L_indices=L_np.indices,
             L_indptr=L_np.indptr,
             L_shape=L_np.shape,
             evals=utils.toNP(evals).astype(dtype_np),
             evecs=utils.toNP(evecs).astype(dtype_np),
             gradX_data=gradX_np.data.astype(dtype_np),
             gradX_indices=gradX_np.indices,
             gradX_indptr=gradX_np.indptr,
             gradX_shape=gradX_np.shape,
             gradY_data=gradY_np.data.astype(dtype_np),
             gradY_indices=gradY_np.indices,
             gradY_indptr=gradY_np.indptr,
             gradY_shape=gradY_np.shape,
             )

    return frames, mass, L, evals, evecs, gradX, gradY


def surf_to_operators(vertices, faces, npz_path, recompute=False):
    """
    Takes the output of msms and dump the diffusion nets operators in dump dir
    :param vert_file: Vx3 tensor of coordinates
    :param face_file: Fx3 tensor of indexes, zero based
    :param dump_dir: Where to dump the precomputed operators
    :return:
    """

    verts = torch.from_numpy(np.ascontiguousarray(vertices))
    faces = torch.from_numpy(np.ascontiguousarray(faces))
    frames, mass, L, evals, evecs, gradX, gradY = get_operators(verts=verts,
                                                                faces=faces,
                                                                npz_path=npz_path,
                                                                recompute=recompute)

    # pre_normals = torch.from_numpy(np.ascontiguousarray(normals))
    # computed_normals = frames[:, 2, :]
    # print(computed_normals.shape)
    # print(pre_normals.shape)
    # print(pre_normals[0])
    # print(computed_normals[0])
    # print(torch.dot(pre_normals[0], computed_normals[0]))
    # print(torch.allclose(computed_normals, pre_normals))
    return frames, mass, L, evals, evecs, gradX, gradY
