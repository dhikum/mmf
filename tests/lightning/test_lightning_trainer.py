# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock

import torch
from mmf.common.report import Report
from mmf.trainers.lightning_trainer import LightningTrainer
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.base import Callback
from tests.test_utils import NumbersDataset, SimpleLightningModel, SimpleModelLogits
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


trainer_config = OmegaConf.create(
    {
        "run_type": "train",
        "training": {
            "detect_anomaly": False,
            "evaluation_interval": 4,
            "log_interval": 2,
            "update_frequency": 1,
            "fp16": False,
        },
        "optimizer": {"type": "adam_w", "params": {"lr": 5e-5, "eps": 1e-8}},
    }
)


class LightningTrainerMock(LightningTrainer):
    def __init__(self, callback, num_data_size=100, batch_size=1):
        self.data_module = MagicMock()
        self._benchmark = False
        self._callbacks = []
        self._distributed = False
        self._gpus = None
        self._gradient_clip_val = False
        self._num_nodes = 1
        self._deterministic = True
        self._automatic_optimization = False
        self._callbacks = [callback]
        self.config = trainer_config
        dataset = NumbersDataset(num_data_size)
        self.data_module.train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        self.data_module.train_loader.current_dataset = MagicMock(return_value=dataset)

    def load_and_calculate_max_updates_test(self, max_updates, max_epochs):
        self.training_config = self.config.training
        self.training_config["max_updates"] = max_updates
        self.training_config["max_epochs"] = max_epochs

        self._calculate_max_updates()
        self._load_trainer()
        return self._max_updates


class TestLightningTrainer(unittest.TestCase, Callback):
    def setUp(self):
        self.lightning_losses = []
        self.mmf_losses = []
        self._callback_enabled = False

    def get_lightning_trainer(self, model_size=1):
        torch.random.manual_seed(2)
        trainer = LightningTrainerMock(callback=self)
        trainer.model = SimpleLightningModel(model_size, config=trainer_config)
        trainer.model.train()
        return trainer

    def get_mmf_trainer(
        self,
        size=1,
        num_data_size=100,
        max_updates=5,
        max_epochs=None,
        on_update_end_fn=None,
    ):
        torch.random.manual_seed(2)
        model = SimpleModelLogits(size)
        model.train()
        optimizer = build_optimizer(model, trainer_config)
        trainer = TrainerTrainingLoopMock(
            num_data_size,
            max_updates,
            max_epochs,
            config=trainer_config,
            optimizer=optimizer,
            on_update_end_fn=on_update_end_fn,
        )
        model.to(trainer.device)
        trainer.model = model
        return trainer

    def test_epoch_over_updates(self):
        trainer = self.get_lightning_trainer()
        max_updates = trainer.load_and_calculate_max_updates_test(2, 0.04)
        self.assertEqual(max_updates, 4)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 4, 0)

    def test_fractional_epoch(self):
        trainer = self.get_lightning_trainer()
        max_updates = trainer.load_and_calculate_max_updates_test(None, 0.04)
        self.assertEqual(max_updates, 4)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 4, 0)

    def test_updates(self):
        trainer = self.get_lightning_trainer()
        max_updates = trainer.load_and_calculate_max_updates_test(2, None)
        self.assertEqual(max_updates, 2)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 2, 0)

    def _check_values(self, trainer, current_iteration, current_epoch):
        self.assertEqual(trainer.trainer.global_step, current_iteration)
        self.assertEqual(trainer.trainer.current_epoch, current_epoch)

    def test_loss_computation(self):
        # check to see the same losses between the two trainers
        self._callback_enabled = True

        # compute mmf_trainer training losses
        def _on_update_end(report, meter, should_log):
            self.mmf_losses.append(report["losses"]["loss"].item())

        mmf_trainer = self.get_mmf_trainer(
            max_updates=5, max_epochs=None, on_update_end_fn=_on_update_end
        )
        mmf_trainer.training_loop()

        # compute lightning_trainer training losses
        trainer = self.get_lightning_trainer()
        trainer.load_and_calculate_max_updates_test(max_updates=5, max_epochs=None)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self._callback_enabled:
            output = outputs[0][0]["extra"]
            report = Report(output["input_batch"], output)
            self.lightning_losses.append(report["losses"]["loss"].item())

    def on_train_end(self, trainer, pl_module):
        if self._callback_enabled:
            for lightning_loss, mmf_loss in zip(self.lightning_losses, self.mmf_losses):
                self.assertEqual(lightning_loss, mmf_loss)
