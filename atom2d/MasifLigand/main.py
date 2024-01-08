import os
import sys

from functools import partialmethod
import hydra
import logging
import torch
import torch.multiprocessing
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from trainer import Trainer
from data import DataLoaderMasifLigand
from hmr_min import set_logger, set_seed
from psr_task.models import PSRSurfNet

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
    data = DataLoaderMasifLigand(config)

    # initialize model
    # from models import load_model
    # Model = load_model(config.model)
    # model = Model(config)


    model = PSRSurfNet(C_width=config.c_width,
                       N_block=config.n_blocks,
                       use_mean=True,
                       batch_norm=True,
                       output_graph=False,
                       use_skip=True,
                       in_channels=37,
                       in_channels_surf=54,
                       out_channel=128,
                       out_features=7,
                       use_graph=True,
                       # use_graph_only=True,
                       use_gat=True,
                       neigh_th=config.neigh_th,
                       use_v2=config.use_v2,
                       dropout=config.dropout,
                       graph_model='bipartite',
                       use_distance=config.use_distance)



    # w_path = "/home/vmallet/projects/atom2d/data/MasifLigand/out_dir/new_init_6/model_last.pt"
    # state_dict = torch.load(w_path, map_location="cpu")['model']
    # model.load_state_dict(state_dict)
    # initialize trainer
    # config.auto_resume = True
    trainer = Trainer(config, data, model)
    trainer.train()


if __name__ == '__main__':
    main()
