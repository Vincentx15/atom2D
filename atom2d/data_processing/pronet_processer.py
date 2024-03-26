from Bio.PDB import PDBParser
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

res_type_dict = {
    'ALA': 0, 'GLY': 1, 'SER': 2, 'THR': 3, 'LEU': 4, 'ILE': 5, 'VAL': 6, 'ASN': 7, 'GLN': 8, 'ARG': 9, 'HIS': 10,
    'TRP': 11, 'PHE': 12, 'TYR': 13, 'GLU': 14, 'ASP': 15, 'LYS': 16, 'PRO': 17, 'CYS': 18, 'MET': 19, 'UNK': 20, }


class PdbEmbedder:
    """
    adapted from https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/dataset/ECdataset.py
    """

    def __init__(self):
        pass

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, 'N')
        mask_ca = np.char.equal(atom_names, 'CA')
        mask_c = np.char.equal(atom_names, 'C')
        mask_cb = np.char.equal(atom_names, 'CB')  # This was wrong
        mask_g = np.char.equal(atom_names, 'CG') | np.char.equal(atom_names, 'SG') | np.char.equal(atom_names,
                                                                                                   'OG') | np.char.equal \
                     (atom_names, 'CG1') | np.char.equal(atom_names, 'OG1')
        mask_d = np.char.equal(atom_names, 'CD') | np.char.equal(atom_names, 'SD') | np.char.equal(atom_names,
                                                                                                   'CD1') | np.char.equal \
                     (atom_names, 'OD1') | np.char.equal(atom_names, 'ND1')
        mask_e = np.char.equal(atom_names, 'CE') | np.char.equal(atom_names, 'NE') | np.char.equal(atom_names, 'OE1')
        mask_z = np.char.equal(atom_names, 'CZ') | np.char.equal(atom_names, 'NZ')
        mask_h = np.char.equal(atom_names, 'NH1')

        pos_n = np.full((len(amino_types), 3), np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types), 3), np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types), 3), np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types), 3), np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types), 3), np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types), 3), np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types), 3), np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types), 3), np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types), 3), np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h

    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_dihedrals(v1, v2, v3), 1)
        angle2 = torch.unsqueeze(self.compute_dihedrals(v2, v3, v4), 1)
        angle3 = torch.unsqueeze(self.compute_dihedrals(v3, v4, v5), 1)
        angle4 = torch.unsqueeze(self.compute_dihedrals(v4, v5, v6), 1)
        angle5 = torch.unsqueeze(self.compute_dihedrals(v5, v6, v7), 1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4), 1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)), 1)

        return side_chain_embs

    def bb_embs(self, X):
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_dihedrals(u0, u1, u2)

        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2])
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    def compute_dihedrals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion

    def protein_to_graph(self, pdb_path):
        parser = PDBParser()
        structure = parser.get_structure("toto", pdb_path)

        amino_types = []  # size: (n_amino,)
        atom_amino_id = []  # size: (n_atom,)
        atom_names = []  # size: (n_atom,)
        atom_pos = []  # size: (n_atom,3)
        res_id = 0
        # Iterate over all residues in a model
        for residue in structure.get_residues():
            resname = residue.get_resname()
            # resname = protein_letters_3to1[resname.title()]
            if resname.upper() not in res_type_dict:
                resname = 'UNK'
            resname = res_type_dict[resname.upper()]
            amino_types.append(resname)
            for atom in residue.get_atoms():
                # skip h
                if atom.get_name().startswith("H"):
                    continue
                atom_amino_id.append(res_id)
                atom_names.append(atom.get_name())
                atom_pos.append(atom.get_coord())
            res_id += 1

        amino_types = np.asarray(amino_types)
        atom_amino_id = np.asarray(atom_amino_id)
        atom_names = np.asarray(atom_names)
        atom_pos = np.asarray(atom_pos)
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(amino_types, atom_names,
                                                                                            atom_amino_id, atom_pos)

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(
            torch.cat((torch.unsqueeze(pos_n, 1), torch.unsqueeze(pos_ca, 1), torch.unsqueeze(pos_c, 1)), 1))
        bb_embs[torch.isnan(bb_embs)] = 0

        # Save those results now
        data = Data()
        data.side_chain_embs = side_chain_embs
        data.bb_embs = bb_embs
        data.x = torch.unsqueeze(torch.tensor(amino_types), 1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c
        assert len(data.x) == len(data.coords_ca) == len(data.coords_n) == len(data.coords_c) == len \
            (data.side_chain_embs) == len(data.bb_embs)

        return data


if __name__ == "__main__":
    processer = PdbEmbedder()
    # process_one = processer.protein_to_graph(
    #     "/home/vmallet/projects/atom2d/data/MasifLigand/raw_data_MasifLigand/pdb/1A27_AB.pdb")
    input_dir = '/home/vmallet/projects/atom2d/data/MasifLigand/raw_data_MasifLigand/pdb/'
    output_dir = '/home/vmallet/projects/atom2d/data/MasifLigand/pronet/'
    import os

    for pdb in os.listdir(input_dir):
        input_file = os.path.join(input_dir, pdb)
        output_file = os.path.join(output_dir, pdb.replace(".pdb", "pronetgraph.pt"))
        if os.path.exists(output_file):
            continue
        graph = processer.protein_to_graph(input_file)
        torch.save(graph, output_file)
