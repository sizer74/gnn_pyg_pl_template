# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gin_module
   Description :
   Author :       s1zer
   date：          2022/1/27 21:18
-------------------------------------------------
   Change Activity:
                   2022/1/27 21:18
-------------------------------------------------
"""
__author__ = 's1zer'

from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_geometric.nn import GIN, global_add_pool
from torch_geometric.nn.conv.gen_conv import MLP
from torchmetrics import Accuracy


class GINModule(pl.LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('GIN')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

        parser.add_argument('--in_channels', type=int, help='input dimension')
        parser.add_argument('--out_channels', type=int, help='output dimension')
        parser.add_argument('--hidden_channels', type=int, default=64, help='hidden dimension')
        parser.add_argument('--num_layers', type=int, default=3, help='number of message passing layers')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
        return parent_parser

    def __init__(self, in_channels: int, out_channels: int, args: dict):
        super(GINModule, self).__init__()
        # in_channels = args['in_channels']
        hidden_channels = args['hidden_channels']
        # out_channels = args['out_channels']
        num_layers = args['num_layers']
        dropout = args['dropout']

        self.lr = args['lr']

        self.gnn = GIN(in_channels=in_channels,  hidden_channels=hidden_channels, num_layers=num_layers,
                       out_channels=hidden_channels, dropout=dropout, jk='cat')
        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              norm='batch', dropout=dropout)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        print(self.gnn)
        print(self.classifier)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)
        return x

    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log(f'train_acc/fold{self.trainer.fit_loop.current_fold}', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        y_hat = self(data.x, data.edge_index, data.batch)
        self.val_acc(y_hat, data.y)
        self.log(f'val_acc/fold{self.trainer.fit_loop.current_fold}', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
