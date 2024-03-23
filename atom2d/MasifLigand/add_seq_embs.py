import os
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
# from graphein.protein.features.sequence.embeddings import esm_residue_embedding
from graphein.protein.features.sequence.utils import subset_by_node_feature_value
import numpy as np
import torch


def esm_residue_embedding(G, model_name, output_layer, device='cpu', loaded_model=None):
    """
    Similar to graphein.protein.features.sequence.embeddings.esm_residue_embedding but with model


    :param G:
    :param model_name:
    :param output_layer:
    :param device:
    :return:
    """
    if loaded_model is None:
        model, alphabet = torch.hub.load("facebookresearch/esm", model_name)
    else:
        model, alphabet = loaded_model

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    for chain in G.graph["chain_ids"]:
        sequence = G.graph[f"sequence_{chain}"]
        data = [("protein1", sequence), ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # Extract per-residue representations
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[output_layer], return_contacts=True)
        token_representations = results["representations"][output_layer]
        # remove start and end tokens from per-token residue embeddings
        embedding = token_representations.cpu().numpy()[0, 1:-1]
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)
        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["esm_embedding"] = embedding[i]
    return G


def compute_esm_embs(pdb,
                     pdb_dir="../../data/MasifLigand/raw_data_MasifLigand/pdb/",
                     out_emb_dir="../../data/MasifLigand/computed_embs/",
                     recompute=False,
                     device='cpu',
                     loaded_model=None):
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
                                  output_layer=33,
                                  device=device,
                                  loaded_model=loaded_model)
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
    device = 'cpu'
    model, alphabet = torch.hub.load("facebookresearch/esm", 'esm1b_t33_650M_UR50S')
    model = model.to(device)

    # compute_esm_embs("1A27_AB.pdb", recompute=True, loaded_model=(model, alphabet))
    input_dir = "../../data/MasifLigand/raw_data_MasifLigand/pdb"
    for i, pdb in enumerate(os.listdir(input_dir)):
        if not i % 20:
            print(f"Doing {i}/{len(os.listdir(input_dir))}")
        compute_esm_embs(pdb, loaded_model=(model, alphabet), recompute=True)
