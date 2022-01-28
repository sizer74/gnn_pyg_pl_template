# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     args
   Description :
   Author :       s1zer
   date：          2022/1/17 21:35
-------------------------------------------------
   Change Activity:
                   2022/1/17 21:35
-------------------------------------------------
"""
__author__ = 's1zer'

import argparse

from pytorch_lightning import Trainer

from data import TUDatasetModule
from loop import KFoldLoop
from model import GINModule


def parser_args():
    parser = argparse.ArgumentParser()

    # add Program level args
    parser.add_argument('--notification', type=str, default='None', help='notification')

    # add module specific args
    parser = TUDatasetModule.add_module_specific_args(parser)
    parser = GINModule.add_module_specific_args(parser)

    # add loop specific args
    parser = KFoldLoop.add_module_specific_args(parser)

    # add all the available trainer option to argparse
    # i.e. now --gpus --num_nodes ... --fast_dev_rn all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    return args

