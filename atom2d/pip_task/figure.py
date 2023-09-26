import os
import sys

# import hydra
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


def main():
    cfg = OmegaConf.load(config_path)
    cfg = cfg.hparams
    model = PIPModule(cfg)
    # saved_model_path = os.path.join(config_dir, "checkpoints/last.ckpt")
    # model.load_state_dict(torch.load(saved_model_path, map_location="cpu")["state_dict"])
    data_dir = Path(cfg.dataset.data_dir) / "test"
    dataset = NewPIP(data_dir, return_graph=True, return_surface=True, big_graphs=True, neg_to_pos_ratio=-1)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=False,
                            shuffle=False, collate_fn=lambda x: AtomBatch.from_data_list(x))
    for item in dataloader:
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

            # Now we follow the code in PIP models and get the embeddings of the posive left (right was not done here)
            processed_left, processed_right, graph_left, graph_right = embeddings
            dists = torch.cdist(unique_positive_samples_left, graph_left.pos)
            min_indices = torch.argmin(dists, dim=1)
            positive_left_embs = processed_left[min_indices]

            # Now that we have the positive left embeddings, let's see if processed right embeddings find a high score
            # We output one score per residue of the right
            # We do this one by one, but we could probably batch
            all_res_right = []
            for positive_left_emb in positive_left_embs:
                in_batch_left = positive_left_emb.expand(len(processed_right), -1)
                in_batch = torch.cat((in_batch_left, processed_right), dim=1)
                out_one = model.model.top_net(in_batch)
                all_res_right.append(out_one)
            all_res_right = torch.concatenate(all_res_right, dim=1)
            all_res_max_right = torch.max(all_res_right, dim=1).values

            # Finally let's push these residue scores to the surface. First we build the graph and message passing
            vertices_right = item.surface_2.vertices
            graph_right = item.graph_2
            bipartite_graphsurf_right, _ = compute_bipartite_graphs(vertices_right,
                                                                    graph_right,
                                                                    neigh_th=8)
            mp = AddAggregate(aggr='max')

            # Then we need to have a full input (bipartite have vertices as nodes) and then select back.
            num_vertices = len(vertices_right[0])
            input_bipartite = torch.hstack((torch.zeros(num_vertices), all_res_max_right))[:, None]
            output_bipartite = mp(input_bipartite,
                                  bipartite_graphsurf_right[0].edge_index,
                                  bipartite_graphsurf_right[0].edge_weight)
            output_vertices = output_bipartite[:num_vertices]
            print(output_vertices)

            # TODO : make the same for the other side.

            # dists = torch.cdist(unique_positive_samples_right, graph_right.pos)
            # min_indices = torch.argmin(dists, dim=1)
            # positive_right_embs = processed_right[min_indices]

        break


if __name__ == "__main__":
    main()
