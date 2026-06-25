# -*- coding: utf-8 -*-
import numpy as np
import nn_models as base_models
from dre.path import ImageInterpXt
import dre.losses.losses as train_step_fns
from .base import DensityRatioEstimationModel

class MIRealDataModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.save_hyperparameters()

    def _build_sde(self, args):
        return ImageInterpXt(args=args)

    def set_basic_functions(self, args):
        self.path = self._build_sde(args)
        self.density_ratio_fn = self.get_dr_fn()
        self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.path)
        self.data_type = args.data
        if self.data_type == "image":
            self.data_shape = self._get_image_shape(args)
        elif self.data_type == "text":
            self.data_shape = (args.mi_dr,)
        else:
            self.data_shape = (args.mi_dr,)
        if self.data_type in ["image", "mixture"]:
            self.model = self.build_image_model()
        else:
            self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda: None

    def _get_image_shape(self, args):
        img_size = int(np.sqrt(args.mi_dr // args.mi_image_channels))
        mi_image_patches = eval(args.mi_image_patches)
        return (args.mi_image_channels * mi_image_patches[0], img_size, img_size)

    def build_image_model(self):
        hidden_channels = list(map(int, self.args.dims.split("-")))
        in_channels = self.data_shape[0] * 2
        image_backbone = getattr(self.args, "image_backbone", "resnet18")
        image_dropout = getattr(self.args, "image_dropout", 0.3)

        if "unet" == image_backbone: 
            score_model = base_models.ImageJointScoreModel
        else:
            score_model = base_models.ImageJointScoreModelResNet
            
        return score_model(
            args=self.args, 
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            embed_dim=self.args.embed_dim, 
            nonlinearity=self.args.nonlinearity, 
            backbone=image_backbone, 
            dropout=image_dropout
        )
