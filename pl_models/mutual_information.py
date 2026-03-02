# -*- coding: utf-8 -*-
from .base import DensityRatioEstimationModel

class MIEstimationModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.reload_interval = args.reload_interval

    def on_train_epoch_end(self):
        if self.datamodule.__class__.__name__ == "MutualInformationDataModule":
            if self.current_epoch % self.reload_interval == 0:
                self.datamodule.setup()
        super().on_train_epoch_end()
