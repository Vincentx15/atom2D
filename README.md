# Atom2D
This repository takes the ATOM3D benchmark and applies
a DiffusionNet onto a protein surface representation.

## Installation

We have a dependency on pymesh, which can be installed following their homepage.
```bash
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
export PYMESH_PATH=`pwd`

# Install Pymesh dependencies with apt-get
apt-get install libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-dev \
    libboost-thread-dev libtbb-dev python3-dev
# Or in jean zay by loading modules 
module load gmp/6.1.2 eigen/3.3.7-mpi cmake/3.21.3 mpfr/4.0.2 boost/1.70.0



./setup.py build
./setup.py install
# Check everything works ok :
python -c "import pymesh; pymesh.test()"
```

Then you can install other packages in the usual way.

```bash
conda create -n atom2d -y
conda activate atom2d
conda install python=3.8
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

## Data

### PIP
In this task, we are given two residues from two proteins and the task is to predict if they interact.
To do so, we learn separate embeddings of each protein and use these embeddings in a pairwise manner.

Then one need to download and extract the appropriate data from atom3D zenodo repo.
You should get the DIPS_split dataset, extract it and save it at the following path :
`data/DIPS-split/data/train`

Then, run :
```
cd pip_task
python preprocess_data.py
```
and the surfaces should start being processed along with the operators used by DiffNets.

To learn a PPI residue predictor, run:
`python train.py`

### MSP
In this task, the goal is to predict whether a mutation is stabilizing a protein interaction.
based on its modified structure. The task is framed as a binary task, where 1 indicates that the modified
structure is more stable.

### PSR
The goal is to predict each protein model TM-score from the ground truth released at CASP. The task is framed
as a regression task on the TM score.

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