import os
import sys

import random
from pathlib import Path
import time
import shutil
import logging
import numpy as np
from tqdm import tqdm

import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# atom type label for one-hot-encoding
atom_type_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'P': 6, 'Cl': 7, 'Se': 8,
                  'Br': 9, 'I': 10, 'UNK': 11}

# residue type label for one-hot-encoding
res_type_dict = {
    'ALA': 0, 'GLY': 1, 'SER': 2, 'THR': 3, 'LEU': 4, 'ILE': 5, 'VAL': 6, 'ASN': 7, 'GLN': 8, 'ARG': 9, 'HIS': 10,
    'TRP': 11, 'PHE': 12, 'TYR': 13, 'GLU': 14, 'ASP': 15, 'LYS': 16, 'PRO': 17, 'CYS': 18, 'MET': 19, 'UNK': 20, }

# Kyte Doolittle scale for hydrophobicity
hydrophob_dict = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7,
    'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5,
    'LYS': -3.9, 'ARG': -4.5, 'UNK': 0.0,
}

res_type_to_hphob = {
    idx: hydrophob_dict[res_type] for res_type, idx in res_type_dict.items()
}

def compute_HKS(eigen_vecs, eigen_vals, num_t, t_min=0.1, t_max=1000, scale=1000):
    eigen_vals = eigen_vals.flatten()
    assert eigen_vals[1] > 0
    assert np.min(eigen_vals) > -1E-6
    assert np.array_equal(eigen_vals, sorted(eigen_vals))

    t_list = np.geomspace(t_min, t_max, num_t)
    phase = np.exp(-np.outer(t_list, eigen_vals[1:]))
    wphi = phase[:, None, :] * eigen_vecs[None, :, 1:]
    HKS = np.einsum('tnk,nk->nt', wphi, eigen_vecs[:, 1:]) * scale
    heat_trace = np.sum(phase, axis=1)
    HKS /= heat_trace

    return HKS


class CSVWriter(object):
    def __init__(self, csv_fpath, columns, overwrite):
        self.csv_fpath = csv_fpath
        self.columns = columns

        if os.path.isfile(self.csv_fpath) and overwrite:
            os.remove(csv_fpath)

        if not os.path.isfile(self.csv_fpath):
            # write columns
            with open(self.csv_fpath, 'w') as handles:
                handles.write(','.join(self.columns) + '\n')

        self.values = {key: '' for key in self.columns}

    def add_scalar(self, name, value):
        assert name in self.columns
        self.values[name] = value

    def write(self):
        with open(self.csv_fpath, 'a') as handles:
            handles.write(','.join([str(self.values[key]) for key in self.columns]) + '\n')
        self.values = {key: '' for key in self.columns}


# base class for data loaders
class DataLoaderBase(ABC):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_data_workers = config.num_data_workers

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    @abstractmethod
    def _init_datasets(self):
        pass

    def _init_samplers(self):
        self.train_sampler = RandomSampler(self.train_set)
        self.valid_sampler = RandomSampler(self.valid_set)
        self.test_sampler = RandomSampler(self.test_set)

    def _init_loaders(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.train_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available())
        self.valid_loader = DataLoader(self.valid_set,
                                       batch_size=self.batch_size,
                                       sampler=self.valid_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.valid_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available())
        self.test_loader = DataLoader(self.test_set,
                                      batch_size=self.batch_size,
                                      sampler=self.test_sampler,
                                      num_workers=self.num_data_workers,
                                      collate_fn=self.test_set.collate_wrapper,
                                      pin_memory=torch.cuda.is_available())


def set_logger(log_fpath):
    """Set file logger at log_fpath"""
    Path(log_fpath).parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_fpath), 'a'),
        ]
    )


def set_seed(seed):
    """Set all random seeds"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operations have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

