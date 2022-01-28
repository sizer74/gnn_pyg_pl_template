# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     base_module
   Description :
   Author :       s1zer
   date：          2022/1/27 21:16
-------------------------------------------------
   Change Activity:
                   2022/1/27 21:16
-------------------------------------------------
"""
__author__ = 's1zer'

from os.path import join as opj
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Any, OrderedDict, Optional, Dict, Union

from pytorch_lightning import Trainer
from pytorch_lightning.loops import Loop, FitLoop
from pytorch_lightning.trainer.states import TrainerFn

from data.base_module import BaseKFoldDataModule


class KFoldLoop(Loop):
    @staticmethod
    def add_module_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('KFoldLoop')
        parser.add_argument('--n_splits', type=int, default=10, help='n_splits')
        parser.add_argument('--n_repeats', type=int, default=1, help='n_repeats')
        parser.add_argument('--random_state', type=int, default=777, help='random state')
        parser.add_argument('--export_path', type=str, default='.', help='export path')
        return parent_parser

    def __init__(self, args) -> None:
        super(KFoldLoop, self).__init__()

        self.n_splits = args['n_splits']
        self.n_repeats = args['n_repeats']
        self.export_path = args['export_path']
        self.random_state = args['random_state']

        self.num_folds = self.n_splits * self.n_repeats

        self.current_fold: int = 0

        self.fit_loop: Optional[FitLoop] = None
        self.lightning_module_state_dict: Optional[OrderedDict] = None

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    @property
    def global_step(self):
        return self.fit_loop.global_step

    @global_step.setter
    def global_step(self, value):
        self.fit_loop.global_step = value

    def connect(self, fit_loop: FitLoop) -> None:
        """Optionally connect one or multiple loops to this one."""
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """
            Reset the internal state  of the loop at the beginning of each call to run.

            Nothing to reset in this loop.
        """

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """
            Hook to be called as the first thing after entering `run` (except the state reset).

            In this case, used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the
            original weights of the model.
        """
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.n_splits, self.n_repeats, self.random_state)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """
            Hook to be called each time before `advance` is called.

            In this case, used to call `setup_fold_index` for the `BaseKFoldDataModule` instance.
        """
        print(f'STRATING  FOLD {self.current_fold}')
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """
            Performs a signal step.

            In this case, used to run a fitting and testing on current hold.
        """
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        # self._reset_testing()  # requires to reset the tracking stage.
        # self.trainer.test_loop.run()

        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """
            Hook to be called each time after `Advance` is called.

            In this case, used to save the weights of the current fold
            land reset the LightningModule and its optimizers.
        """
        self.trainer.save_checkpoint(opj(self.export_path, f'model.{self.current_fold}.pt'))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        """
            Optionally replace one or multiple of this loop's sub-loops.
            This methods takes care of instantiating the class (if necessary) with all
            existing arguments, connecting all sub-loops of the old loop to the new instance, 
            setting the `Trainer` reference, and connecting the new loop to the parent.
        """
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """
            Hook to be called at the end of the run.

            In this case, used to compute the performance of the ensemble model on the test set.
        """

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        """Resets the train/val dataloader and initialises required variables"""
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        """Set the trainer state into fitting"""
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        """Resets the test dataloader and initialises requires variables"""
        self.trainer.reset_test_dataloader()
        """Set the triner state into testing"""
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]


def get_k_fold_trainer(args: Union[Namespace, ArgumentParser]):
    trainer = Trainer.from_argparse_args(args)
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(vars(args))
    trainer.fit_loop.connect(internal_fit_loop)
    return trainer
