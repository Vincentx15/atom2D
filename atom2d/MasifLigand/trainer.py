import os
import torch
from tqdm import tqdm
import logging
import contextlib
import numpy as np
import shutil
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
# from ema_pytorch import EMA
from data_processing.hmr_min import CSVWriter

from metrics import multi_class_eval
from abc import ABC

from torch.optim.lr_scheduler import _LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR


def get_lr_scheduler(scheduler, optimizer, warmup_epochs, total_epochs, eta_min=1E-8):
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
                                            eta_min=eta_min)
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

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) /
                        (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]


# base trainer class
class Trainer(ABC):
    def __init__(self, config, data, model):
        self.config = config
        self.device = config.device
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
        self.test_best = False
        self.shrink_outputs = config.shrink_outputs
        self.shrink_epochs = config.shrink_epochs
        self.use_ema = config.use_ema

        # model
        self.model = model
        if self.device != 'cpu':
            self.model = self.model.to(self.device)

        # ema
        if self.use_ema:
            self.ema = EMA(self.model, beta=config.ema_decay, update_every=config.ema_update_every)

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
        # LR scheduler
        self.scheduler = get_lr_scheduler(scheduler=config.lr_scheduler,
                                          optimizer=self.optimizer,
                                          warmup_epochs=config.warmup_epochs,
                                          total_epochs=config.epochs,
                                          eta_min=config.lr_eta_min)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # tensorboard and HDFS
        self.tb_writer = SummaryWriter(log_dir=self.out_dir, filename_suffix=f'.{self.run_name}')
        columns = [
            "Epoch", "Partition", "CrossEntropy_avg", "Accuracy_micro", "Accuracy_macro", "Accuracy_balanced",
            "Precision_micro", "Precision_macro", "Recall_micro", "Recall_macro", "F1_micro", "F1_macro",
            "AUROC_macro",
        ]
        self.csv_writer = CSVWriter(os.path.join(self.out_dir, 'metrics.csv'), columns, overwrite=False)

    def train(self):
        # automatically resume training
        if self.auto_resume:
            try:
                self._auto_resume()
            except:
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
                self._save_checkpoint(epoch=epoch,
                                      is_best=is_best,
                                      best_perf=self.best_perf)

                # predict on test set using the latest model
                if epoch % self.test_freq == 0:
                    logging.info('Evaluating the latest model on test set')
                    self._train_epoch(epoch=epoch,
                                      data_loader=self.test_loader,
                                      data_sampler=None,
                                      partition='test')

        # evaluate best model on test set
        log_msg = [f'Total training time: {time.time() - train_t0:.1f} sec,',
                   f'total number of epochs: {epoch:d},',
                   f'average epoch time: {np.mean(epoch_times):.1f} sec']
        logging.info(' '.join(log_msg))
        self.test_best = True
        logging.info('---------Evaluate Best Model on Test Set---------------')
        with open(os.path.join(self.out_dir, 'model_best.pt'), 'rb') as fin:
            best_model = torch.load(fin, map_location='cpu')['model']
        self.model.load_state_dict(best_model)
        self._train_epoch(epoch=-1,
                          data_loader=self.test_loader,
                          data_sampler=None,
                          partition='test')

    def _auto_resume(self):
        # load from local output directory
        with open(os.path.join(self.out_dir, 'model_last.pt'), 'rb') as fin:
            checkpoint = torch.load(fin, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_perf = checkpoint['best_perf']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
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

    # override pure virtual function
    def _train_epoch(self, epoch, data_loader, data_sampler, partition):
        # init average meters
        pred_scores = []
        labels = []
        cross_entropy_avg_list = []

        # reshuffle data across GPU workers
        if isinstance(data_sampler, DistributedSampler):
            data_sampler.set_epoch(epoch)

        if partition == 'train':
            self.model.train()
        else:
            self.model.eval()
            self.model.train()
            if self.use_ema:
                self.ema.ema_model.eval()

        exploding_grad = []
        context = contextlib.nullcontext() if partition == 'train' else torch.no_grad()
        with context:
            for i, batch in enumerate(tqdm(data_loader)):
                # send data to device and compute model output
                batch.to(self.device)
                if partition == 'train':
                    if self.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            output = self.model.forward(batch)
                            cross_entropy_loss = self.criterion(output, batch.labels)
                    else:
                        output = self.model.forward(batch)
                        cross_entropy_loss = self.criterion(output, batch.labels)
                        if self.shrink_outputs > 0:
                            if epoch < self.shrink_epochs:
                                shrink_loss = self.shrink_outputs * torch.mean(output) ** 2
                                cross_entropy_loss += shrink_loss
                else:
                    if self.use_ema:
                        output = self.ema.ema_model.forward(batch).squeeze(-1)
                    else:
                        output = self.model.forward(batch).squeeze(-1)
                    cross_entropy_loss = self.criterion(output, batch.labels)

                # print(self.model.encoder_model.diffnet_block_0.diffusion.diffusion_time[:3])
                # print(output) # TODO: shrink outputs for stability :
                #  current outputs look like :
                #  -1078.4442, -1106.6555, 370.8554, -570.8733, -265.3181, -603.5452, 745.7878

                if partition == 'train':
                    # compute gradient and optimize
                    self.optimizer.zero_grad()
                    if self.fp16:  # mixed precision
                        self.scaler.scale(cross_entropy_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        if grad_norm > self.clip_grad_norm:
                            exploding_grad.append(grad_norm.item())
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:  # torch.float32 default precision
                        cross_entropy_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        if grad_norm > self.clip_grad_norm:
                            exploding_grad.append(grad_norm.item())
                        self.optimizer.step()

                    # update ema model
                    if self.use_ema:
                        self.ema.update()

                # add ouputs
                pred_scores.append(output)
                labels.append(batch['labels'])
                cross_entropy_avg_list += [cross_entropy_loss.item()] * batch['labels'].size(dim=0)

        # synchronize metrics
        cross_entropy = np.mean(cross_entropy_avg_list)
        accuracy_macro, accuracy_micro, accuracy_balanced, \
            precision_macro, precision_micro, \
            recall_macro, recall_micro, \
            f1_macro, f1_micro, \
            auroc_macro = multi_class_eval(torch.cat(pred_scores, dim=0), torch.cat(labels, dim=0), K=7)

        current_lr = self.optimizer.param_groups[0]['lr']
        lr = f'{current_lr:.8f}' if partition == 'train' else '--'

        print_info = [
            f'===> Epoch {epoch} {partition.upper()}, LR: {lr}\n',
            f'CrossEntropyAvg: {cross_entropy:.3f}\n',
            f'AccuracyAvg: {accuracy_macro:.3f} (macro), {accuracy_micro:.3f} (micro), {accuracy_balanced:.3f} (balanced)\n',
            f'PrecisionAvg: {precision_macro:.3f} (macro), {precision_micro:.3f} (micro)\n',
            f'RecallAvg: {recall_macro:.3f} (macro), {recall_micro:.3f} (micro)\n',
            f'F1Avg: {f1_macro:.3f} (macro), {f1_micro:.3f} (micro)\n',
            f'AUROCAvg: {auroc_macro:.3f} (macro)\n',
        ]
        logging.info(''.join(print_info))

        self.csv_writer.add_scalar("Epoch", epoch)
        self.csv_writer.add_scalar("Partition", partition)
        self.csv_writer.add_scalar("CrossEntropy_avg", cross_entropy)

        self.csv_writer.add_scalar("Accuracy_macro", accuracy_macro)
        self.csv_writer.add_scalar("Accuracy_micro", accuracy_micro)
        self.csv_writer.add_scalar("Accuracy_balanced", accuracy_balanced)

        self.csv_writer.add_scalar("Precision_macro", precision_macro)
        self.csv_writer.add_scalar("Precision_micro", precision_micro)

        self.csv_writer.add_scalar("Recall_macro", recall_macro)
        self.csv_writer.add_scalar("Recall_micro", recall_micro)

        self.csv_writer.add_scalar("F1_macro", f1_macro)
        self.csv_writer.add_scalar("F1_micro", f1_micro)

        self.csv_writer.add_scalar("AUROC_macro", auroc_macro)

        self.csv_writer.write()

        if self.tb_writer is not None:
            if self.test_best:
                self.tb_writer.add_scalar(f"best_acc_balanced", accuracy_balanced, 10)
            else:
                self.tb_writer.add_scalar(f"CrossEntropy_avg/{partition}", cross_entropy, epoch)

                self.tb_writer.add_scalar(f"Accuracy_macro/{partition}", accuracy_macro, epoch)
                self.tb_writer.add_scalar(f"Accuracy_micro/{partition}", accuracy_micro, epoch)
                self.tb_writer.add_scalar(f"Accuracy_balanced/{partition}", accuracy_balanced, epoch)

                self.tb_writer.add_scalar(f"Precision_macro/{partition}", precision_macro, epoch)
                self.tb_writer.add_scalar(f"Precision_micro/{partition}", precision_micro, epoch)

                self.tb_writer.add_scalar(f"Recall_macro/{partition}", recall_macro, epoch)
                self.tb_writer.add_scalar(f"Recall_micro/{partition}", recall_micro, epoch)

                self.tb_writer.add_scalar(f"F1_macro/{partition}", f1_macro, epoch)
                self.tb_writer.add_scalar(f"F1_micro/{partition}", f1_micro, epoch)

                self.tb_writer.add_scalar(f"AUROC_macro/{partition}", auroc_macro, epoch)
                if partition == 'train':
                    self.tb_writer.add_scalar('LR', current_lr, epoch)

        # check exploding gradient
        explode_ratio = len(exploding_grad) / len(data_loader)
        if explode_ratio > 0.01:
            log_msg = [f'Exploding gradient ratio: {100 * explode_ratio:.1f}%,',
                       f'exploded gradient mean: {np.mean(exploding_grad):.2f}']
            logging.info(' '.join(log_msg))

        performance = accuracy_balanced  # we always maximize model performance
        return cross_entropy, performance
