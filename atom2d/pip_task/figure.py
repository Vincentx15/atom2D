import os
import sys
import pickle
from tqdm import tqdm

import hydra
from pathlib import Path
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PIPModule
from data_loader import NewPIP
from data_processing.data_module import AtomBatch
from omegaconf import OmegaConf
from base_nets.architectures import AddAggregate, compute_bipartite_graphs

# model_name = "version_17_parallel_final_real"
# config_dir = f"../../outputs/pip/lightning_logs/{model_name}"
# config_path = f"{config_dir}/hparams.yaml"
config_path = "toy.yaml"


def get_prediction_scores(model, positive_embs, processed):
    # Now that we have the positive left embeddings, let's see if processed right embeddings find a high score
    # We output one score per residue of the right
    # We do this one by one, but we could probably batch
    all_results = []
    for positive_emb in positive_embs:
        in_batch_left = positive_emb.expand(len(processed), -1)
        in_batch = torch.cat((in_batch_left, processed), dim=1)
        out_one = model.model.top_net(in_batch)
        all_results.append(out_one)
    all_results = torch.concatenate(all_results, dim=1)
    all_results_max = torch.max(all_results, dim=1).values
    return all_results_max


def push_to_surf(vertices, graph, graph_features):
    # First we build the graph and message passing
    bipartite_graphsurf_right, _ = compute_bipartite_graphs(vertices,
                                                            graph,
                                                            neigh_th=10)
    mp = AddAggregate(aggr='max')

    # Then we need to have a full input (bipartite have vertices as nodes) and then select back.
    num_vertices = len(vertices[0])
    input_bipartite = torch.hstack((torch.zeros(num_vertices), graph_features))[:, None]
    output_bipartite = mp(input_bipartite,
                          bipartite_graphsurf_right[0].edge_index,
                          bipartite_graphsurf_right[0].edge_weight)
    output_vertices = output_bipartite[:num_vertices]
    return output_vertices


@hydra.main(config_path="./", config_name="config")
def main(cfg=None):
    # cfg = OmegaConf.load(config_path)
    # cfg = cfg.hparams
    model = PIPModule(cfg)
    version = "version_150_bipartite_4_skip_gat"  # todo change
    name = "epoch=18-auroc_val=0.855"  # todo change
    save_name = "/mnt/disk2/souhaib/data4.pkl"  # todo change

    config_dir = Path(__file__).resolve().parent / f"../../outputs/pip/lightning_logs/{version}/"
    saved_model_path = config_dir / f"checkpoints/{name}.ckpt"
    model.load_state_dict(torch.load(saved_model_path, map_location="cpu")["state_dict"])
    data_dir = Path(cfg.dataset.data_dir) / "test"
    dataset = NewPIP(data_dir, return_graph=True, return_surface=True, big_graphs=True, neg_to_pos_ratio=-1)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=False,
                            shuffle=True, collate_fn=lambda x: AtomBatch.from_data_list(x))
    counter = 0
    save_dict = {}
    for item in tqdm(dataloader):
        with torch.no_grad():
            # Forward with return_embs is new, it returns the embeddings instead of prediction
            embeddings = model.forward(item, return_embs=True)

            # The loader was created with neg_to_pos=-1 => all positive and negative are returned
            locs_left = item.locs_left[0]
            locs_right = item.locs_right[0]

            # Now let's select the coordinates of all positive pairs.
            labels_pip = item.labels_pip
            positive_samples_left = locs_left[torch.where(labels_pip == 1)]
            positive_samples_right = locs_right[torch.where(labels_pip == 1)]
            # Then we take the left and right unique positives ie two tensors N,3 with coordinates of the positive
            unique_positive_samples_left = positive_samples_left.unique(dim=0)
            unique_positive_samples_right = positive_samples_right.unique(dim=0)

            # Now we follow the code in PIP models and get the indices of positive residues
            processed_left, processed_right, graph_left, graph_right = embeddings
            dists_left = torch.cdist(unique_positive_samples_left, graph_left.pos)
            min_indices_left = torch.argmin(dists_left, dim=1)
            dists_right = torch.cdist(unique_positive_samples_right, graph_right.pos)
            min_indices_right = torch.argmin(dists_right, dim=1)

            # Use the residue ids to get GT vectors over graph nodes
            gt_nodes_left = torch.zeros(graph_left.num_nodes)
            gt_nodes_left[min_indices_left] = 1
            gt_nodes_right = torch.zeros(graph_right.num_nodes)
            gt_nodes_right[min_indices_right] = 1

            # Use the positive residue ids to get positive vectors, and then prediction for all graph nodes
            positive_left_embs = processed_left[min_indices_left]
            predictions_right = get_prediction_scores(model, positive_left_embs, processed_right)
            positive_right_embs = processed_right[min_indices_right]
            predictions_left = get_prediction_scores(model, positive_right_embs, processed_left)

            # Finally let's push these residue scores to the surface.
            vertices_right = item.surface_2.vertices
            graph_right = item.graph_2
            surf_gt_right = push_to_surf(vertices_right, graph_right, gt_nodes_right)
            surf_preds_right = push_to_surf(vertices_right, graph_right, predictions_right)

            vertices_left = item.surface_1.vertices
            graph_left = item.graph_1
            surf_gt_left = push_to_surf(vertices_left, graph_left, gt_nodes_left)
            surf_preds_left = push_to_surf(vertices_left, graph_left, predictions_left)
            a = 1

            save_dict[f"{item.name1[0]}_{item.name2[0]}"] = [vertices_left[0].numpy(), item.surface_1.faces[0].numpy(),
                                                             surf_preds_left.numpy().flatten(), surf_gt_left.numpy().flatten(),
                                                             vertices_right[0].numpy(), item.surface_2.faces[0].numpy(),
                                                             surf_preds_right.numpy().flatten(), surf_gt_right.numpy().flatten()]

        counter += 1
        if counter > 50:  # todo change
            break

    with open(save_name, 'wb') as file:
        pickle.dump(save_dict, file)


if __name__ == "__main__":
    main()
