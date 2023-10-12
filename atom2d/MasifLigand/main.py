import os
import sys

import argparse
from easydict import EasyDict as edict
from functools import partialmethod
import json
import logging
import torch
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from trainer import Trainer
from data import DataLoaderMasifLigand
from hmr_min import set_logger, set_seed
from psr_task.models import PSRSurfNet


def train(config):
    # get dataloader
    data = DataLoaderMasifLigand(config)

    # initialize model
    # from models import load_model
    # Model = load_model(config.model)
    # model = Model(config)
    model = PSRSurfNet(C_width=184,
                       N_block=4,
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


def get_config():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')

    # logging arguments
    parser.add_argument('--run_name', type=str, default="default")
    parser.add_argument('--out_dir', type=str, default='../../data/MasifLigand/outdir')
    parser.add_argument('--test_freq', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--auto_resume', type=lambda x: eval(x))
    parser.add_argument('--mute_tqdm', type=lambda x: eval(x))
    # data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/MasifLigand/dataset_MasifLigand/')
    parser.add_argument('--processed_dir', type=str, default='../../data/MasifLigand/cache_npz')
    parser.add_argument('--operator_dir', type=str, default='../../data/MasifLigand/cache_operator')
    parser.add_argument('--train_split_file', type=str, default='splits/train-list.txt')
    parser.add_argument('--valid_split_file', type=str, default='splits/val-list.txt')
    parser.add_argument('--test_split_file', type=str, default='splits/test-list.txt')
    parser.add_argument('--use_chem_feat', type=lambda x: eval(x), default=True)
    parser.add_argument('--use_geom_feat', type=lambda x: eval(x), default=True)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_data_workers', type=int)
    parser.add_argument('--num_gdf', type=int)
    # optimizer arguments
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, choices=['PolynomialLRWithWarmup', 'CosineAnnealingLRWithWarmup'])
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--fp16', type=lambda x: eval(x))
    # model-specific arguments
    model_names = ['HMR']
    parser.add_argument('--model', type=str, choices=model_names)
    parser.add_argument('--device', type=int, default=-1)
    args = parser.parse_args()

    # load default config
    with open(args.config) as f:
        config = json.load(f)

    # update config with user-defined args
    for arg in vars(args):
        if getattr(args, arg) is not None:
            model_name = arg[:arg.find('_')]
            if model_name in model_names:
                model_arg = arg[arg.find('_') + 1:]
                config[model_name][model_arg] = getattr(args, arg)
            else:
                config[arg] = getattr(args, arg)

    return edict(config)


if __name__ == '__main__':
    # init config
    config = get_config()
    config.is_master = True
    if torch.cuda.is_available():
        if int(config.device) > 0:
            config.device = f'cuda:{config.device}'
        else:
            config.device = f'cuda'

    else:
        config.device = 'cpu'

    print("Training on ", config.device)

    # init horovod for distributed training
    config.use_hvd = torch.cuda.device_count() > 1
    # if config.use_hvd:
    #     hvd.init()
    #     torch.cuda.set_device(hvd.local_rank())
    #     config.is_master = hvd.local_rank() == 0
    #     config.num_GPUs = hvd.size() # for logging purposes
    #     assert hvd.size() == torch.cuda.device_count()

    # logging, attach the hook after automatic download from HDFS
    set_logger(os.path.join(config.out_dir, 'train.log'))
    if config.is_master:
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
    if config.mute_tqdm or not config.is_master:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    train(config)
