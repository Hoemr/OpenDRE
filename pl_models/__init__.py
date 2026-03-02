# -*- coding: utf-8 -*-
from .base import DensityEstimationModel, DensityRatioEstimationModel
from .density_estimation_2d_synthetic import DensityEstimation2DSyntheticModel
from .density_estimation_tabular import DensityEstimationTabularModel
from .density_estimation_real_image import DensityEstimationRealImageModel
from .distribution_shift import DistributionShiftMeasureModel
from .mutual_information import MIEstimationModel
from .mi_real import MIRealDataModel
from .two_sample_test import TwoSampleTestModel
from .change_point_detection import ChangePointDetectionModel
from .ood_detection import OutOfDistributionDetectionModel

SUBTASK_MODLE_MODULES = {
    "synthetic2d": DensityEstimation2DSyntheticModel,
    "tabular": DensityEstimationTabularModel,
    "real_image": DensityEstimationRealImageModel,
    "distribution_shift": DistributionShiftMeasureModel,
    "mutual_information": MIEstimationModel,
    "mi_realdata": MIRealDataModel,
    "two_sample_test": TwoSampleTestModel,
    "change_point_detection": ChangePointDetectionModel,
    "ood_detection": OutOfDistributionDetectionModel,
}

__all__ = [
    "DensityEstimationModel",
    "DensityRatioEstimationModel",
    "DensityEstimation2DSyntheticModel",
    "DensityEstimationTabularModel",
    "DensityEstimationRealImageModel",
    "DistributionShiftMeasureModel",
    "MIEstimationModel",
    "MIRealDataModel",
    "TwoSampleTestModel",
    "ChangePointDetectionModel",
    "OutOfDistributionDetectionModel",
    "SUBTASK_MODLE_MODULES",
]
