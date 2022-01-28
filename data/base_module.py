# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     base_module
   Description :
   Author :       s1zer
   date：          2022/1/27 21:11
-------------------------------------------------
   Change Activity:
                   2022/1/27 21:11
-------------------------------------------------
"""
__author__ = 's1zer'

from abc import ABC, abstractmethod
from typing import Optional

from pytorch_lightning import LightningDataModule


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, n_splits: int, n_repeats: Optional[int] = None, random_state: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def setup_folds_index(self, split_index: int) -> None:
        pass

