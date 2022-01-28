# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tu_dataset_module
   Description :
   Author :       s1zer
   date：          2022/1/27 21:14
-------------------------------------------------
   Change Activity:
                   2022/1/27 21:14
-------------------------------------------------
"""
__author__ = 's1zer'

from argparse import ArgumentParser
from typing import Optional

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from data.base_module import BaseKFoldDataModule


class TUDatasetModule(BaseKFoldDataModule):
    @staticmethod
    def add_module_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('TUDataset')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--num_workers', type=int, default=40, help='number of workers')
        parser.add_argument('--name', type=str, default='MUTAG',
                            choices=['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY'], help='data name')
        parser.add_argument('--root', type=str, default='/lab_data/datasets/TUDataset', help='dataset root path')
        return parent_parser

    def __init__(self, args: dict):
        super(TUDatasetModule, self).__init__()
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.dataset_name = args['name']
        self.dataset_root = args['root']

        self.n_splits: Optional[int] = None
        self.n_repeats: Optional[int] = None
        self.random_state: Optional[int] = None
        self.splits: Optional[list] = None

        self.dataset: Optional[Dataset] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        TUDataset(root=self.dataset_root, name=self.dataset_name, pre_transform=T.OneHotDegree(135))

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = TUDataset(root=self.dataset_root, name=self.dataset_name, pre_transform=T.OneHotDegree(135))

    def setup_folds(self, n_splits: int, n_repeats: Optional[int] = None, random_state: Optional[int] = None) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.splits = [split for split in
                       RepeatedStratifiedKFold(n_splits=self.n_splits,
                                               n_repeats=self.n_repeats,
                                               random_state=self.random_state).split(self.dataset,
                                                                                     self.dataset.data['y'])]

    def setup_folds_index(self, split_index: int) -> None:
        train_indices, val_indices = self.splits[split_index]
        self.train_dataset, self.val_dataset = Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
