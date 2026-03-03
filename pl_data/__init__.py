# -*- coding: utf-8 -*-
from .base import DataModule, IterDataModule

from .density_estimation import (
    TABULAR_DATASETS,
    SYNTHETIC2D_DATASETS,
    TabularDataModule,
    Synthetic2DIterDataModule,
)
from .density_estimation_image import REAL_IMAGE_DATASETS, SPECS, DATA_SHAPES, RealImageDataModule
from .distribution_shift import DISTRIBUTION_SHIFT_DATASETS, DistributionShiftIterDataModule
from .mutual_information import MI_DATASETS, MutualInformationIterDataModule, MutualInformationDataModule
from .mi_real import MI_REAL_DATASETS, MIRealDataModule
from .two_sample_test import TwoSampleTestIterDataModule
from .change_point_detection import CHANGEPOINT_DATASETS, ChangePointDetectionDataModule
from .ood_detection import (
    OOD_IN_DIST_DATASETS,
    OOD_BASE_TYPES,
    OOD_EVAL_CONFIG,
    OOD_IMGLIST_NEAR_FAR,
    OutOfDistributionDetectionDataModule,
)

SUBTASK_DATA_MODULES = {
    "synthetic2d": Synthetic2DIterDataModule,
    "tabular": TabularDataModule,
    "real_image": RealImageDataModule,
    "distribution_shift": DistributionShiftIterDataModule,
    "mutual_information": MutualInformationIterDataModule,
    "mi_realdata": MIRealDataModule,
    "two_sample_test": TwoSampleTestIterDataModule,
    "change_point_detection": ChangePointDetectionDataModule,
    "ood_detection": OutOfDistributionDetectionDataModule,
}

__all__ = [
    "DataModule",
    "IterDataModule",
    "TABULAR_DATASETS",
    "SYNTHETIC2D_DATASETS",
    "TabularDataModule",
    "Synthetic2DIterDataModule",
    "REAL_IMAGE_DATASETS",
    "SPECS",
    "DATA_SHAPES",
    "RealImageDataModule",
    "DISTRIBUTION_SHIFT_DATASETS",
    "DistributionShiftIterDataModule",
    "MI_DATASETS",
    "MutualInformationIterDataModule",
    "MutualInformationDataModule",
    "MI_REAL_DATASETS",
    "MIRealDataModule",
    "TwoSampleTestIterDataModule",
    "CHANGEPOINT_DATASETS",
    "ChangePointDetectionDataModule",
    "OOD_IN_DIST_DATASETS",
    "OOD_BASE_TYPES",
    "OOD_EVAL_CONFIG",
    "OutOfDistributionDetectionDataModule",
    "OOD_IMGLIST_NEAR_FAR",
    "SUBTASK_DATA_MODULES",
]
