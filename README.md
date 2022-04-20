This repo takes the ATOM3D benchmark and applies 
a DiffusionNet onto a protein surface representation.

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