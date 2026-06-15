# -*- coding: utf-8 -*-
import os
import sys

import torch
import torchvision.utils as vutils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data import SUBTASK_DATA_MODULES
from pl_models import SUBTASK_MODLE_MODULES

from config import parse_args, get_save_path
from dre_utils.utils import set_random_seeds, makedirs
import dre_utils.logger as logger_utils

import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument.*")
warnings.filterwarnings("ignore", ".*torch.meshgrid: in an upcoming release.*")


def setup_tensor_cores():
    """
    Enable Tensor Cores optimization if the GPU supports it.
    
    Tensor Cores are available on GPUs with compute capability >= 7.0:
    - Volta (V100), Turing (RTX 20xx, T4), Ampere (A100, A6000, RTX 30xx),
      Hopper (H100), Ada Lovelace (RTX 40xx)
    
    This sets float32 matmul precision to 'high' for better performance
    on supported hardware, while maintaining compatibility on older GPUs.
    """
    if not torch.cuda.is_available():
        return
    
    # Get compute capability of the current device
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    compute_capability = major + minor * 0.1
    
    # Tensor Cores require SM >= 7.0 (Volta, Turing, Ampere, Hopper, Ada)
    if major >= 7:
        torch.set_float32_matmul_precision('high')
        print(f"[Tensor Cores] Enabled on {torch.cuda.get_device_name(device)} (SM {major}.{minor})")
    else:
        print(f"[Tensor Cores] Not available on {torch.cuda.get_device_name(device)} (SM {major}.{minor})")

def get_trainer_config(args, need_strategy=False):
    """
    Unified configuration for Trainer's accelerator/devices/strategy/sync_batchnorm.
    
    Args:
        args: Command-line arguments
        need_strategy: Whether to return strategy and sync_batchnorm (required only for 
                       likelihood_estimation training with multi-GPU support)
    
    Returns:
        For CPU: (accelerator="cpu", devices=1, [strategy="auto", sync_batchnorm=False] if needed)
        For GPU: (accelerator="gpu", devices=[gpu_id] or list, [strategy, sync_batchnorm] if needed)
    """
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    
    if accelerator == "cpu":
        # CPU mode: devices MUST be 1 (strict requirement in PyTorch Lightning 2.1.2)
        devices = 1
        if need_strategy:
            return accelerator, devices, "auto", False
        return accelerator, devices
    
    # GPU mode: preserve original logic - convert integer gpu_id to list for explicit device selection
    devices = [args.gpu_id] if isinstance(args.gpu_id, int) else args.gpu_id
    
    if need_strategy:
        # Calculate actual number of devices (list length or integer value)
        num_devices = len(devices) if isinstance(devices, (list, tuple)) else devices
        strategy = "ddp" if num_devices > 1 else "auto"
        sync_batchnorm = num_devices > 1
        return accelerator, devices, strategy, sync_batchnorm
    
    return accelerator, devices
    
def setup_test_environment(datamodule, args, save_path):
    """
    Common setup for test-only mode: stage setup, GPU cleanup, and logging configuration.
    """
    datamodule.setup(stage="test")
    if args.subtask == "real_image":
        args.in_channels = datamodule.channel_size
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Savepath:", save_path)
    print("Command:", "python " + ' '.join(sys.argv))
    logger_utils.setup_silent_mode()

def train_likelihood_estimation(args):
    # Setup directories and logging
    save_path = get_save_path(args)
    checkpoint_dir = os.path.join(save_path, "checkpoints")
    
    # Prepare dataset and dataloader
    datamodule_class_name = SUBTASK_DATA_MODULES[args.subtask]
    datamodule = datamodule_class_name(args)
    datamodule.prepare_data()
    
    model_module_name = SUBTASK_MODLE_MODULES[args.subtask]
    
    if args.test_only:
        import glob
        setup_test_environment(datamodule, args, save_path)
        
        if args.subtask == "synthetic2d":
            target_pattern = "final_model*.ckpt"
        else:
            target_pattern = "best_ckpt*.ckpt"
            
        search_path = os.path.join(checkpoint_dir, target_pattern)
        list_of_files = glob.glob(search_path)
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getmtime)
            ckpt_path = latest_file
            print(f"[Info] Auto-detected latest checkpoint: {os.path.basename(ckpt_path)}")
        else:
            default_name = "final_model.ckpt" if args.subtask == "synthetic2d" else "best_ckpt.ckpt"
            ckpt_path = os.path.join(checkpoint_dir, default_name)
            print(f"[Warning] No checkpoint found matching '{target_pattern}', trying default: {default_name}")
        
        model = model_module_name.load_from_checkpoint(ckpt_path, strict=False, map_location=args.device, args=args)
        
        if args.sampling_only:  # "AIS-HMC" sampling mode
            model.sampling(256, method="HMC", num_steps=500, leapfrog_steps=30, step_size=0.05, filename="best")
            return

        # Fixed device configuration for CPU/GPU compatibility
        accelerator, devices = get_trainer_config(args)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger_utils.NullLogger(),
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0, 
        )

        trainer.test(model, datamodule=datamodule)
        return
    
    # Training mode setup
    logger = logger_utils.setup_txt_logger(save_path)
    logger_utils.log_args_and_metrics(logger, args)

    early_stopping_callback = EarlyStopping(monitor="val_nll", mode="min", verbose=True, patience=args.early_stopping+1)
    txt_logger_callback = logger_utils.TxtLoggerCallback(logger, args=args, early_stopping_callback=early_stopping_callback)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, 
        filename="best_ckpt", 
        monitor="val_nll",
        mode="min", 
        save_top_k=1, 
        save_last=True,
        enable_version_counter=False,  # Don't create -v1, -v2 versions
    )
    
    # Task-specific hyperparameters and callbacks
    precision = 32
    end_test = True
    max_epochs = args.epochs
    enable_progress_bar = False
    accumulate_grad_batches = 1
    
    if args.subtask == "tabular":
        callbacks = [checkpoint_callback, early_stopping_callback, txt_logger_callback]
        max_epochs = 1000000
        
    elif args.subtask == "synthetic2d":
        callbacks = [txt_logger_callback]
        accumulate_grad_batches = 5
        
    elif args.subtask == "real_image":
        args.in_channels = datamodule.channel_size
        callbacks = [checkpoint_callback, early_stopping_callback, txt_logger_callback]
        max_epochs = 1000000
        # precision = "16-mixed"
        
    # Fixed device configuration with strategy support for multi-GPU
    accelerator, devices, strategy, sync_batchnorm = get_trainer_config(args, need_strategy=True)
    enable_checkpointing = checkpoint_callback is not None
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        max_epochs=max_epochs,
        default_root_dir=save_path,
        logger=logger_utils.NullLogger(),  
        check_val_every_n_epoch=args.eval_freq,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar, 
        num_sanity_val_steps=-1,  # -1: validate on entire dataset; 0: skip pre-validation
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=accumulate_grad_batches, 
        precision=precision,
        # detect_anomaly=True,
    )
    
    pl_model = model_module_name(args, save_path)
    # trainer.validate(pl_model, dataloaders=datamodule)
    
    # Train the model
    trainer.fit(pl_model, datamodule=datamodule)
    if trainer.is_global_zero:
        trainer.save_checkpoint(os.path.join(checkpoint_dir, "final_model.ckpt"))
    
    # Post-training evaluation on last and best checkpoints
    if end_test:
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Evaluate last saved model
        trainer.test(pl_model, datamodule=datamodule)
        
        # Evaluate best model based on validation metric
        best_model = model_module_name.load_from_checkpoint(checkpoint_callback.best_model_path, strict=False)
        trainer.test(best_model, datamodule=datamodule)

def train_fdiv_estimation(args):
    # Setup directories and logging
    save_path = get_save_path(args)
    checkpoint_dir = os.path.join(save_path, "checkpoints")
    
    datamodule_class_name = SUBTASK_DATA_MODULES[args.subtask]
    datamodule = datamodule_class_name(args)
    
    model_module_name = SUBTASK_MODLE_MODULES[args.subtask]
    
    # Initialize checkpoint_callback (may be set by subtask-specific code)
    checkpoint_callback = None
    
    if args.test_only:
        import glob
        setup_test_environment(datamodule, args, save_path)
        
        # Find all best_ckpt*.ckpt files
        best_pattern = os.path.join(checkpoint_dir, "best_ckpt*.ckpt")
        best_candidates = sorted(glob.glob(best_pattern), key=os.path.getmtime, reverse=True)
        
        # Try each checkpoint and find the most compatible one
        ckpt_path = None
        for candidate in best_candidates:
            # Quick check: load checkpoint and inspect key structure
            try:
                ckpt_data = torch.load(candidate, map_location='cpu', weights_only=False)
                state_dict = ckpt_data.get('state_dict', {})
                model_keys = [k[6:] for k in state_dict.keys() if k.startswith('model.')]
                
                # Check if time_head has the new structure (time_head.0.weight instead of time_head.0.layers.0.weight)
                has_new_time_head = any('time_head.0.weight' in k for k in model_keys)
                has_old_time_head = any('time_head.0.layers.0.weight' in k for k in model_keys)
                
                if has_new_time_head and not has_old_time_head:
                    ckpt_path = candidate
                    print(f"[Info] Using compatible checkpoint: {os.path.basename(ckpt_path)}")
                    break
            except Exception:
                continue
        
        if ckpt_path is None:
            if best_candidates:
                ckpt_path = best_candidates[0]
                print(f"[Warning] No fully compatible checkpoint found, using: {os.path.basename(ckpt_path)}")
            else:
                ckpt_path = os.path.join(checkpoint_dir, "final_model.ckpt")
                print(f"[Warning] No best checkpoint found, falling back to {os.path.basename(ckpt_path)}")
        
        best_model = model_module_name.load_from_checkpoint(
            ckpt_path, 
            strict=False, 
            map_location=args.device, 
            args=args, 
            datamodule=datamodule
        )

        # Fixed device configuration for CPU/GPU compatibility
        accelerator, devices = get_trainer_config(args)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger_utils.NullLogger(),
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0, 
        )

        trainer.test(best_model, datamodule=datamodule)
        return
    
    # Training mode setup
    logger = logger_utils.setup_txt_logger(save_path)
    logger_utils.log_args_and_metrics(logger, args)
    txt_logger_callback = logger_utils.TxtLoggerCallback(logger, args=args)

    enable_progress_bar = False
    max_epochs = args.epochs
    precision = 32
    accumulate_grad_batches = 1
    end_test = False
    
    # Task-specific hyperparameters and callbacks
    if args.subtask == "distribution_shift":
        callbacks = [
            txt_logger_callback,
            logger_utils.ShiftControllerCallback(datamodule=datamodule, reload_interval=args.reload_interval)
        ]
        max_epochs = args.reload_interval * args.shift_steps
        
    elif args.subtask == "change_point_detection":
        callbacks = [txt_logger_callback]
        
    elif args.subtask == "mutual_information":
        callbacks = [txt_logger_callback]
        accumulate_grad_batches = 5
        
    elif args.subtask == "mi_realdata":
        callbacks = [txt_logger_callback]
        accumulate_grad_batches = 5

    elif args.subtask == "ood_detection":
        early_stopping_callback = EarlyStopping(
            monitor="val_near_auroc", 
            mode="max", 
            verbose=True, 
            patience=args.early_stopping
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, 
            filename="best_ckpt", 
            monitor="val_near_auroc",
            mode="max", 
            save_top_k=1, 
            save_last=True,
            enable_version_counter=False,  # Don't create -v1, -v2 versions
        )
        txt_logger_callback.early_stopping_callback = early_stopping_callback
        callbacks = [checkpoint_callback, early_stopping_callback, txt_logger_callback]
        end_test = True
        # precision = "16-mixed"
    else:
        raise ValueError(f"Invalid subtask {args.subtask}")

    # Fixed device configuration
    accelerator, devices = get_trainer_config(args)
    # Enable checkpointing only if ModelCheckpoint callback is used
    enable_checkpointing = checkpoint_callback is not None
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        default_root_dir=save_path,
        logger=logger_utils.NullLogger(),  # Optionally add logger for TensorBoard : pl.loggers.TensorBoardLogger("./")
        check_val_every_n_epoch=args.eval_freq,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar, 
        num_sanity_val_steps=0,   # -1: all val datasets; 0: no pre test
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=accumulate_grad_batches, 
        precision=precision,
        reload_dataloaders_every_n_epochs=args.reload_interval,
        # detect_anomaly=True,
    )
    
    # Initial validation setup
    datamodule.setup()
    pl_model = model_module_name(args, save_path, datamodule)
    
    # Train the model
    trainer.fit(pl_model, datamodule=datamodule)
    trainer.save_checkpoint(os.path.join(checkpoint_dir, "final_model.ckpt"))

    if end_test:
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Evaluate last saved model
        trainer.test(pl_model, datamodule=datamodule)
        
        # Evaluate best model if checkpoint_callback exists and has a best model
        if checkpoint_callback and checkpoint_callback.best_model_path:
            best_model = model_module_name.load_from_checkpoint(
                checkpoint_callback.best_model_path, 
                strict=False, 
                map_location=args.device, 
                args=args, 
                datamodule=datamodule
            )
            trainer.test(best_model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_args()
    set_random_seeds(args.seed)
    
    # Enable Tensor Cores optimization if available
    setup_tensor_cores()
    
    if args.task == "likelihood_estimation":
        train_likelihood_estimation(args)
    elif args.task == "fdiv_estimation":
        train_fdiv_estimation(args)
    else:
        raise ValueError(f"Task {args.task} is not implemented!")
