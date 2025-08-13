# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, mask=None, weights=None) -> Tensor:
        """
          scores: [N, ..., C], unnormalized scores
          target: [N, ...]
          mask: [N, ...], where elements with `True` are allowed and `False` are masked-out
        """
        n_tokens = target.numel()
        n_nonpad_tokens = target.ne(2).long().sum() - target.size(0)

        bsz, num_classes = scores.shape[0], scores.shape[-1]        

        if mask is not None:
            scores = scores[mask]  # [N * len, C]
            target = target[mask]  # [N]
            sample_size = torch.tensor(target.numel(), device=target.device)
        scores = scores.reshape(-1, num_classes)
        target = target.reshape(-1)

        # if self.ignore_index is not None:
        #     sample_size = target.ne(self.ignore_index).long().sum()            

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        
        if weights is not None:
            weighted_loss, weighted_nll_loss = loss * weights, nll_loss * weights

        ppl = torch.exp(loss.sum() / sample_size)

        weighted_loss = weighted_loss.sum() / sample_size

        logging_output = {
            'nll_loss_sum': weighted_nll_loss.data,
            'loss_sum': weighted_loss.data,
            'ppl': ppl.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens
        }
        return weighted_loss, logging_output


class RDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, weights=None,
                cal_constant_loss=False,
                watch_t1_t2_loss=False,
                ) -> Tensor:
        """
          scores: [N, L, C], unnormalized scores
          target: [N, L]
          coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        bsz, num_classes = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        # [N, L]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        t1_loss, t2_loss = None, None
        if watch_t1_t2_loss:
            t1_loss, t2_loss = loss.chunk(2)
            t1_mask, t2_mask = label_mask.chunk(2)
            t1_loss = (t1_loss * t1_mask).sum() / (t1_mask.sum())
            t2_loss = (t2_loss * t2_mask).sum() / (t2_mask.sum())
            
        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss

        ppl = torch.exp(nll_loss)
        
        logging_output = {
            'nll_loss': nll_loss.data,
            'ppl': ppl.data,
            'fullseq_loss': fullseq_loss.data,
            'fullseq_nll_loss': fullseq_nll_loss.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens,
            'weight_diff_loss': loss.data
        }
        
        if cal_constant_loss:
            constant_weights = weights.new_ones(size=weights.size())
            constant_loss, _ = label_smoothed_nll_loss(
                lprobs=F.log_softmax(scores, dim=-1),
                target=target,
                epsilon=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduce=False,
            )
            constant_loss = constant_loss * constant_weights
            constant_loss = (constant_loss * label_mask).sum() / sample_size
            logging_output['constant_diff_loss'] = constant_loss.data

        if watch_t1_t2_loss:
            logging_output['weight_diff_t1_loss'] = t1_loss.data
            logging_output['weight_diff_t2_loss'] = t2_loss.data
        
        return loss, logging_output

# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/tree/master
# name="cosine_annealing_warmup",
# version="2.0",
# author="Naoki Katsura",
# packages=['cosine_annealing_warmup'],
# description="Cosine Annealing with Warmup for PyTorch",
# long_description=open("README.md").read(),


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr