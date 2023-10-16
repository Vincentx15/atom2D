import os
import sys

# import argparse
# from easydict import EasyDict as edict
from functools import partialmethod
import json
import logging
import torch
from tqdm import tqdm
import hydra

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from trainer import Trainer
from data import DataLoaderMasifLigand
from hmr_min import set_logger, set_seed
from psr_task.models import PSRSurfNet

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="./", config_name="config")
def main(config=None):
    if torch.cuda.is_available():
        if int(config.device) > 0:
            config.device = f'cuda:{config.device}'
        else:
            config.device = f'cuda'
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
                       use_gat=True,
                       graph_model='bipartite',
                       **config)
    # initialize trainer
    trainer = Trainer(config, data, model)
    trainer.train()


if __name__ == '__main__':
    main()
