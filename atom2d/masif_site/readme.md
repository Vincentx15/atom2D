## Getting the data:

The data is originally built from the MaSIF repo.
Actually, the ground truth is obtained as a by-product of the masif-search:
it computes the GT using the difference between the surface of the whole complex and of just one.
This is detailed [here](https://github.com/LPDI-EPFL/masif/blob/master/source/data_preparation/01-pdb_extract_and_triangulate.py#L105).
In dMasif, they propose a [small function](https://github.com/FreyrS/dMaSIF/blob/master/model.py#L270) to send this GT onto their own data.


Then, we have pdbs that need to be turned into a graph and ply files that incorporate the GT in the form of: 
P["mesh_xyz"] and P["mesh_labels"].
The code to compute loss and metrics is [here](https://github.com/FreyrS/dMaSIF/blob/master/data_iteration.py#L157).

## Diff with HMR
- There are other info on the surface (following masif)
- We don't use Gaussian Distance Function encoding for curvatures
- We use the ply files from masif and our own evecs
