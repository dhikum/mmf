# Copyright (c) Facebook, Inc. and its affiliates.

import logging

import torch
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.trainers.core.reporting import TrainerReportingMixin
from mmf.utils.checkpoint import Checkpoint
from mmf.utils.logger import calculate_time_left, summarize_report
from mmf.utils.timer import Timer
from pytorch_lightning.callbacks.base import Callback


logger = logging.getLogger(__name__)


class LightningLoopCallback(Callback, TrainerReportingMixin):
    def __init__(self, lightning_trainer):
        super().__init__()
        self.lightning_trainer = lightning_trainer
        self.training_config = lightning_trainer.training_config
        self.total_timer = Timer()
        self.snapshot_iterations = len(lightning_trainer.data_module.val_loader)
        self.snapshot_iterations //= self.training_config.batch_size

    def on_init_start(self, trainer):
        self._checkpoint = Checkpoint(self.lightning_trainer)

    def on_train_start(self, trainer, pl_module):
        registry.register("current_epoch", trainer.current_epoch)
        # ```combined_report``` is only used for train. Keeping the name
        # consistent with the mmf_trainer counterpart, though ideally
        # its name should have 'train' in it to make it clear.
        self.combined_report = None
        self.train_timer = Timer()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.train_loader.change_dataloader()

        step_output = outputs[0][0]["extra"]
        input_batch = step_output["input_batch"]
        report = Report(input_batch, step_output)

        should_accumulate = not (batch_idx % self.training_config.update_frequency == 0)

        if not should_accumulate or self.combined_report is None:
            self.combined_report = report
        else:
            self.combined_report.accumulate_tensor_fields(
                report, self.lightning_trainer.metrics.required_params
            )
            self.combined_report.accumulate_loss(report)
            self.combined_report.batch_size += report.batch_size

        # log
        if trainer.global_step % self.training_config.log_interval == 0:
            if self.training_config.evaluate_metrics:
                self.combined_report.metrics = self.lightning_trainer.metrics(
                    self.combined_report, self.combined_report
                )
            self.update_meter(self.combined_report, self.meter)
            self._log(trainer)

        # eval
        if trainer.global_step % self.training_config.evaluation_interval == 0:
            self._start_eval(trainer)

        # save checkpoints
        if trainer.global_step % self.training_config.checkpoint_interval == 0:
            self._save_checkpoint(trainer)

    def on_train_end(self, trainer, pl_module):
        self._start_eval(trainer)

    def on_validation_start(self, trainer, pl_module):
        logger.info("Evaluation time. Running on full validation set...")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.val_loader.change_dataloader()

    def _start_eval(self, trainer):
        trainer.test(
            model=self.lightning_trainer.model,
            test_dataloaders=self.lightning_trainer.data_module.val_loader,
        )

    def _save_checkpoint(self, trainer):
        logger.info("Checkpoint time. Saving a checkpoint.")
        return
        # TODO: sash Needs implementation - coming soon

    def _log(self, trainer):
        extra = {}
        if "cuda" in str(trainer.model.device):
            extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
            extra["max mem"] //= 1024

        if self.training_config.experiment_name:
            extra["experiment"] = self.training_config.experiment_name

        assert (
            len(trainer.optimizers) == 1
        ), "mmf lightning_trainer supports 1 optimizer per model for now."
        optimizer = trainer.optimizers[0]
        extra.update(
            {
                "epoch": trainer.current_epoch,
                "num_updates": trainer.global_step,
                "iterations": trainer.batch_idx,
                "max_updates": trainer.max_steps,
                "lr": "{:.5f}".format(optimizer.param_groups[0]["lr"]).rstrip("0"),
                "ups": "{:.2f}".format(
                    self.training_config.log_interval
                    / self.train_timer.unix_time_since_start()
                ),
                "time": self.train_timer.get_time_since_start(),
                "time_since_start": self.total_timer.get_time_since_start(),
                "eta": calculate_time_left(
                    max_updates=trainer.max_steps,
                    num_updates=trainer.global_step,
                    timer=self.train_timer,
                    num_snapshot_iterations=self.snapshot_iterations,
                    log_interval=self.training_config.log_interval,
                    eval_interval=self.training_config.evaluation_interval,
                ),
            }
        )
        self.train_timer.reset()
        summarize_report(
            current_iteration=trainer.batch_idx,
            num_updates=trainer.global_step,
            max_updates=trainer.max_steps,
            meter=self.meter,
            extra=extra,
            tb_writer=self.lightning_trainer.tb_writer,
        )
