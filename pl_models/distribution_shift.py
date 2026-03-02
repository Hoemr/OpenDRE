# -*- coding: utf-8 -*-
from .base import DensityRatioEstimationModel

class DistributionShiftMeasureModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
