import os
import sys

from functools import partialmethod
import hydra
import logging
import torch
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.hmr_min import set_logger, set_seed
from MasifLigand.data import DataLoaderMasifLigand
from MasifLigand.trainer import Trainer

from base_nets.pronet_updated import ProNet


class ProNetMasifLigand(torch.nn.Module):
    def __init__(self,
                 level='allatom',
                 num_blocks=4,
                 hidden_channels=128,
                 mid_emb=64,
                 num_radial=6,
                 num_spherical=2,
                 cutoff=10.0,
                 max_num_neighbors=32,
                 int_emb_layers=3,
                 # out_layers=2,
                 num_pos_emb=16,
                 add_seq_emb=False):
        super(ProNetMasifLigand, self).__init__()
        self.pronet = ProNet(level=level,
                             num_blocks=num_blocks,
                             hidden_channels=hidden_channels,
                             mid_emb=mid_emb,
                             num_radial=num_radial,
                             num_spherical=num_spherical,
                             cutoff=cutoff,
                             max_num_neighbors=max_num_neighbors,
                             int_emb_layers=int_emb_layers,
                             num_pos_emb=num_pos_emb,
                             add_seq_emb=add_seq_emb)

        self.top_net = nn.Sequential(*[
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_channels, 7)
        ])

    def project_processed_graph(self, locs, processed, graph):
        # find nearest neighbors between doing last layers
        dists = torch.cdist(locs, graph.pos)
        min_indices = torch.argmin(dists, dim=1)
        processed = processed[min_indices]
        x = self.top_net(processed)
        return x

    def select_close(self, verts, processed, graph_pos):
        # find nearest neighbors between doing last layers
        with torch.no_grad():
            dists = torch.cdist(verts, graph_pos)
            min_indices = torch.argmin(dists, dim=1)
        selected = processed[min_indices.unique()]
        return selected

    def forward(self, batch):
        pronet_graph = batch.pronet_graph
        embed_graph = self.pronet(pronet_graph)
        verts = batch.verts
        processed_feats = embed_graph.split(pronet_graph.batch.bincount().tolist())
        selected_feats = []
        for vert, feats, graph in zip(verts, processed_feats, pronet_graph.to_data_list()):
            feats_close = self.select_close(vert, feats, graph.coords_ca)
            selected_feats.append(feats_close)
        # all_mean_embs = torch.stack([torch.mean(x, dim=-2) for x in selected_feats])
        all_mean_embs = torch.stack([torch.sum(x, dim=-2) for x in selected_feats])
        out = self.top_net(all_mean_embs)
        return out


torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="./", config_name="config")
def main(config=None):
    if torch.cuda.is_available() and int(config.device) >= 0:
        config.device = f'cuda:{config.device}'
    else:
        config.device = 'cpu'
    print("Training on ", config.device)
    config.out_dir = os.path.join(config.out_dir, config.run_name)
    set_logger(os.path.join(config.out_dir, 'train.log'))
    logging.info('==> Configurations')
    for key, val in sorted(config.items(), key=lambda x: x[0]):
        if isinstance(val, dict):
            for k, v in val.items():
                logging.info(f'\t[{key}] {k}: {v}')
        else:
            logging.info(f'\t{key}: {val}')

    # set random seed
    set_seed(config.seed)

    # mute tqdm
    if config.mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    train(config)


def train(config):
    # get dataloader
    # config.batch_size = 2
    # config.num_data_workers = 0
    # config.add_seq_emb = False
    # config.c_width = 10
    # config.lr = 0.0005

    config.data_dir = "../../data/MasifLigand/dataset_MasifLigand/"
    config.processed_dir = "../../data/MasifLigand/cache_npz/"

    data = DataLoaderMasifLigand(config, use_pronet=True)
    model = ProNetMasifLigand(hidden_channels=config.c_width,
                              add_seq_emb=config.add_seq_emb)

    trainer = Trainer(config, data, model)
    trainer.train()


if __name__ == '__main__':
    main()
