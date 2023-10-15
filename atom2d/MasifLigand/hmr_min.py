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
from torch.optim.lr_scheduler import _LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR


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


def get_lr_scheduler(scheduler, optimizer, warmup_epochs, total_epochs):
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=1E-3,
                                total_iters=warmup_epochs)

    if scheduler == 'PolynomialLRWithWarmup':
        decay_scheduler = PolynomialLR(optimizer,
                                       total_iters=total_epochs - warmup_epochs,
                                       power=1)
    elif scheduler == 'CosineAnnealingLRWithWarmup':
        decay_scheduler = CosineAnnealingLR(optimizer,
                                            T_max=total_epochs - warmup_epochs,
                                            eta_min=1E-8)
    elif scheduler == 'constant':
        lambda1 = lambda epoch: 1.0
        decay_scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise NotImplementedError

    return SequentialLR(optimizer,
                        schedulers=[warmup_scheduler, decay_scheduler],
                        milestones=[warmup_epochs])


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (
                1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]


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

"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


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
        self.use_hvd = config.use_hvd
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


# base trainer class
class TrainerBase(ABC):
    def __init__(self, config, data, model):
        self.config = config
        self.device = config.device
        self.use_hvd = config.use_hvd
        self.is_master = config.is_master
        self.run_name = config.run_name
        self.train_loader, self.valid_loader, self.test_loader = \
            data.train_loader, data.valid_loader, data.test_loader
        self.train_sampler = data.train_sampler
        self.best_perf = float('-inf')
        self.epochs = config.epochs
        self.start_epoch = 1
        self.warmup_epochs = config.warmup_epochs
        self.test_freq = config.test_freq
        self.clip_grad_norm = config.clip_grad_norm
        self.fp16 = config.fp16
        self.scaler = torch.cuda.amp.GradScaler()
        self.out_dir = config.out_dir
        self.auto_resume = config.auto_resume

        # model
        self.model = model
        if self.device != 'cpu':
            self.model = self.model.to(self.device)
        if self.is_master:
            # logging.info(self.model)
            learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f'Number of learnable model parameters: {learnable_params}')
            msg = [f'Batch size: {config.batch_size}, number of batches in data loaders - train:',
                   f'{len(self.train_loader)}, valid: {len(self.valid_loader)}, test: {len(self.test_loader)}']
            logging.info(' '.join(msg))

        # optimizer
        if not config.optimizer in ['Adam', 'AdamW']:
            raise NotImplementedError
        optim = getattr(torch.optim, config.optimizer)
        self.optimizer = optim(self.model.parameters(),
                               lr=config.lr,
                               weight_decay=config.weight_decay)

        # # distributed training
        # if self.use_hvd:
        #     compression = hvd.Compression.fp16 if self.fp16 else hvd.Compression.none
        #     self.optimizer = hvd.DistributedOptimizer(self.optimizer, compression=compression)
        #     hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        #     hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

        # LR scheduler
        self.scheduler = get_lr_scheduler(scheduler=config.lr_scheduler,
                                          optimizer=self.optimizer,
                                          warmup_epochs=config.warmup_epochs,
                                          total_epochs=config.epochs)

    def train(self):
        # automatically resume training
        try:
            self._auto_resume()
        except:
            if self.is_master:
                logging.info(f'Failed to load checkpoint from {self.out_dir}, start training from scratch..')

        train_t0 = time.time()
        epoch_times = []
        with tqdm(range(self.start_epoch, self.epochs + 1)) as tq:
            for epoch in tq:
                tq.set_description(f'Epoch {epoch}')
                epoch_t0 = time.time()

                train_loss, train_perf = self._train_epoch(epoch=epoch,
                                                           data_loader=self.train_loader,
                                                           data_sampler=self.train_sampler,
                                                           partition='train')
                valid_loss, valid_perf = self._train_epoch(epoch=epoch,
                                                           data_loader=self.valid_loader,
                                                           data_sampler=None,
                                                           partition='valid')
                # train_loss, train_perf = self._train_epoch(epoch=epoch,
                #                                            data_loader=self.test_loader,
                #                                            data_sampler=None,
                #                                            partition='test')

                self.scheduler.step()

                tq.set_postfix(train_loss=train_loss, valid_loss=valid_loss,
                               train_perf=abs(train_perf), valid_perf=abs(valid_perf))

                epoch_times.append(time.time() - epoch_t0)

                # save checkpoint
                is_best = valid_perf > self.best_perf
                self.best_perf = max(valid_perf, self.best_perf)
                if self.is_master:
                    self._save_checkpoint(epoch=epoch,
                                          is_best=is_best,
                                          best_perf=self.best_perf)

                # predict on test set using the latest model
                if epoch % self.test_freq == 0:
                    if self.is_master:
                        logging.info('Evaluating the latest model on test set')
                    self._train_epoch(epoch=epoch,
                                      data_loader=self.test_loader,
                                      data_sampler=None,
                                      partition='test')

        # evaluate best model on test set
        if self.is_master:
            log_msg = [f'Total training time: {time.time() - train_t0:.1f} sec,',
                       f'total number of epochs: {epoch:d},',
                       f'average epoch time: {np.mean(epoch_times):.1f} sec']
            logging.info(' '.join(log_msg))
            self.tb_writer = None  # do not write to tensorboard
            logging.info('---------Evaluate Best Model on Test Set---------------')
        with open(os.path.join(self.out_dir, 'model_best.pt'), 'rb') as fin:
            best_model = torch.load(fin, map_location='cpu')['model']
        self.model.load_state_dict(best_model)
        self._train_epoch(epoch=-1,
                          data_loader=self.test_loader,
                          data_sampler=None,
                          partition='test')

    def _auto_resume(self):
        assert self.auto_resume
        # load from local output directory
        with open(os.path.join(self.out_dir, 'model_last.pt'), 'rb') as fin:
            checkpoint = torch.load(fin, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_perf = checkpoint['best_perf']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.is_master:
            logging.info(f'Loaded checkpoint from {self.out_dir}, resume training at epoch {self.start_epoch}..')

    def _save_checkpoint(self, epoch, is_best, best_perf):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_perf': best_perf,
        }
        filename = os.path.join(self.out_dir, 'model_last.pt')
        torch.save(state_dict, filename)
        if is_best:
            logging.info(f'Saving current model as the best')
            shutil.copyfile(filename, os.path.join(self.out_dir, 'model_best.pt'))

    @abstractmethod
    def _train_epoch(self):
        pass

    @staticmethod
    def all_reduce(val):
        if torch.cuda.device_count() < 2:
            return val

        # if not isinstance(val, torch.tensor):
        #     val = torch.tensor(val)
        # avg_tensor = hvd.allreduce(val)
        # return avg_tensor.item()
