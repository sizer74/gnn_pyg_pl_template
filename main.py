# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       s1zer
   date：          2022/1/17 23:15
-------------------------------------------------
   Change Activity:
                   2022/1/17 23:15
-------------------------------------------------
"""
__author__ = 's1zer'

from torch_geometric import transforms as T
from pytorch_lightning import seed_everything, Trainer
from torch_geometric.datasets import TUDataset


from args import parser_args
from data import TUDatasetModule
from loop.k_fold_loop import get_k_fold_trainer
from model import GINModule

if __name__ == '__main__':
    seed_everything(777)

    args = parser_args()
    args_dict = vars(args)

    dataset = TUDataset(root=args_dict['root'], name=args_dict['name'], pre_transform=T.OneHotDegree(135))

    datamodule = TUDatasetModule(args_dict)
    model = GINModule(in_channels=dataset.num_node_features, out_channels=dataset.num_classes, args=args_dict)
    trainer = get_k_fold_trainer(args)

    trainer.fit(model, datamodule)
