# Atom2D 
This repository takes the ATOM3D benchmark and applies 
a DiffusionNet onto a protein surface representation.

## Installation
To transform the data from the dataframe format into the one
used by DiffNets, we need three dependencies
`atom3d, diffusion_nets and msms`.

- For atom3d and diffusionnets, you just need to clone 
them and add them to your PYTHONPATH.
- For msms, one needs to download it and add the executable 
in your PATH.

Then one need to download and extract the appropriate data from atom3D zenodo repo.
You should get the DIPS_split dataset, extract it and save it at the following path :
`data/DIPS-split/data/train`

Finally run :
`python build_surfaces.py`
and the surfaces should start being processed along with the operators used by DiffNets.

To learn a PPI residue predictor, run 
`python train.py`

### Structure of the project
The data processing is organized as pipeline. 
The object we start from is an ATOM3D Dataframe, which is the item
contained in LMDB datasets. 
Those Dataframes are then processed : 
- by Atom3D to produce the coordinates of positive and negative pairs of CA.
- by our pipeline to produce the geometry objects needed for DiffNets

The steps of our pipeline are the following :
- (DF -> PDB) in df_utils.py
- (PDB -> surface mesh .ply file) in surf_utils.py
- (surface -> precomputed operators) in build_surfaces.py
- (.ply mesh and PDB -> npz features) in point_cloud_utils.py

The features are obtained through RBF interpolation from the 
protein residues to the vertices of the mesh. We can
then use the mesh and operators to embed our protein surface
with DiffusionNet. 

Then we use RBF interpolation from the vertices onto selected
CA of each protein to get a feature vector for these atoms.
Finally we feed these feature vectors to an MLP to discriminate
CA that interact from the one that do not.