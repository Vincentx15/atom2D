import os
import numpy as np
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.sequence.embeddings import esm_residue_embedding


def compute_esm_embs(pdb,
                     pdb_dir="../../data/MasifLigand/raw_data_MasifLigand/pdb/",
                     out_emb_dir="../../data/MasifLigand/computed_embs/",
                     recompute=False):
    pdb_path = os.path.join(pdb_dir, pdb)
    out_embs_path = os.path.join(out_emb_dir, pdb.split(".")[0] + ".npy")

    if os.path.exists(out_embs_path) and not recompute:
        return True
    try:
        os.makedirs(out_emb_dir, exist_ok=True)
        config = ProteinGraphConfig()
        g = construct_graph(config=config, path=pdb_path)
        g = esm_residue_embedding(g,
                                  model_name='esm1b_t33_650M_UR50S',
                                  output_layer=33)
        all_embs = []
        for n, d in g.nodes(data=True):
            all_embs.append(d["esm_embedding"])
        all_embs = np.stack(all_embs)
        np.save(open(out_embs_path, "wb"), all_embs)
    except Exception as e:
        print(e)
        return False
    return True


def get_esm_embs(pdb, out_emb_dir):
    out_embs_path = os.path.join(out_emb_dir, pdb.split(".")[0] + ".npy")
    if not os.path.exists(out_embs_path):
        return None
    esm_embs = np.load(open(out_embs_path, "rb"))
    return esm_embs


if __name__ == '__main__':
    # get_esm("1A27_AB.pdb")
    input_dir = "../../data/MasifLigand/raw_data_MasifLigand/pdb"
    for i, pdb in enumerate(os.listdir(input_dir)):
        if not i % 20:
            print(f"Doing {i}/{len(os.listdir(input_dir))}")
        compute_esm_embs(pdb)
