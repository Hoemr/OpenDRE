import os
import sys
import logging
import warnings
import pytorch_lightning as pl
from typing import Optional, Dict, List, Any

def setup_txt_logger(save_path):
    log_file = os.path.join(save_path, "training_log.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add a FileHandler in write mode
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add a StreamHandler for console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_args_and_metrics(logger, args):
    logger.info("Run command:")
    command = "python " + ' '.join(sys.argv)
    logger.info(command)
    
    logger.info("Run parameters:")
    args_str = ""
    for attr in dir(args):
        if not attr.startswith("__") and not attr.startswith("_"):
            value = getattr(args, attr)
            args_str += f"{attr}: {value}\n"
    logger.info(args_str)

class TxtLoggerCallback(pl.Callback):
    def __init__(self, logger, args, early_stopping_callback=None):
        self.logger = logger
        self.args = args
        self.task = args.task
        self.subtask = args.subtask
        self.early_stopping_callback = early_stopping_callback
        
        self._init_handlers()
        
    def _init_handlers(self):
        """Initialize task-specific handlers based on task and subtask."""
        # Common handlers
        self._handle_train_epoch_end = self._noop
        self._handle_validation_epoch_end = self._noop
        self._handle_test_epoch_end = self._noop
        
        # Task-specific overrides
        if self.task == "likelihood_estimation":
            if self.subtask == "tabular":
                self._handle_train_epoch_end = self._likelihood_tabular_train
                self._handle_validation_epoch_end = self._likelihood_tabular_val
                self._handle_test_epoch_end = self._likelihood_tabular_test
            elif self.subtask == "real_image":
                self._handle_train_epoch_end = self._likelihood_real_image_train
                self._handle_validation_epoch_end = self._likelihood_real_image_val
                self._handle_test_epoch_end = self._likelihood_real_image_test
            else:
                self._handle_train_epoch_end = self._likelihood_other_train
                self._handle_validation_epoch_end = self._likelihood_other_val

        elif self.task == "fdiv_estimation":
            self._handle_validation_epoch_end = self._fdiv_val
            if self.subtask == "mi_realdata":
                self._handle_train_epoch_end = self._fdiv_mi_train
            elif self.subtask == "ood_detection":
                self._handle_train_epoch_end = self._ood_train
                self._handle_validation_epoch_end = self._ood_val
                self._handle_test_epoch_end = self._ood_test

    def _noop(self, trainer, pl_module):
        pass
    
    # =====================================================
    # =============== Core logging utility =================
    # =====================================================
    def _log_epoch(
        self,
        *,
        trainer,
        metrics: Optional[Dict[str, Any]] = None,
        formatters: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None,
        epoch_offset: int = 1,
        use_max_epoch: bool = True,
        extra_parts: Optional[List[str]] = None,
        global_zero: bool = False,
    ):
        if global_zero and not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + epoch_offset
        parts = []

        # ---- Epoch header ----
        head = f"{prefix + ' ' if prefix else ''}Epoch {epoch}"
        if use_max_epoch:
            head += f"/{trainer.max_epochs}"
        parts.append(head)

        # ---- Metrics ----
        if metrics and formatters:
            for k, fmt in formatters.items():
                v = metrics.get(k)
                if v is not None:
                    try:
                        parts.append(fmt(v))
                    except Exception:
                        parts.append(f"{k}={v}")

        # ---- Extra info ----
        if extra_parts:
            parts.extend(extra_parts)

        self.logger.info(", ".join(parts))

    def _get_metrics(self, trainer, keys):
        return {k: trainer.callback_metrics.get(k) for k in keys}
        
    def on_fit_start(self, trainer, pl_module):
        device_type = trainer.strategy.root_device.type
        self.logger.info(
            f"GPU available: {device_type == 'cuda'}, used: {device_type == 'cuda'}\n"
            f"TPU available: {device_type == 'xla'}, using: {device_type == 'xla'}"
        )
        
        # Log LOCAL_RANK and CUDA_VISIBLE_DEVICES
        self.logger.info(
            f"LOCAL_RANK: {trainer.local_rank} - "
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}"
        )
        
        # Log model summary
        trainable_params = 0
        non_trainable_params = 0
        total_size = 0
        for param in pl_module.parameters():
            num_params = param.numel()
            param_size = num_params * param.element_size()
            total_size += param_size
            
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params

        total_params = trainable_params + non_trainable_params
        size_mb = total_size / (1024 ** 2)
        self.logger.info(
            f"Model Summary:\n{str(pl_module)}\n"
            f"Trainable params: {trainable_params}\n"
            f"Non-trainable params: {non_trainable_params}\n"
            f"Total params: {total_params}\n"
            f"Total estimated model params size: {size_mb:.3f} MB"
        )
    
    # =====================================================
    # ================= Likelihood ========================
    # =====================================================
    def _likelihood_tabular_train(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['train_loss', 'train_nll']),
            formatters={
                'train_loss': lambda v: f"Average Loss {v:.4f}",
                'train_nll':  lambda v: f"Average NLL {v:.4f}",
            }
        )
    
    def _likelihood_tabular_val(self, trainer, pl_module):
        m = self._get_metrics(trainer, ['val_nll', 'val_nfe', 'val_time'])
        self._log_epoch(
            trainer=trainer,
            metrics=m,
            formatters={
                'val_nll':  lambda v: f"Eval NLL {v:.4f}",
                'val_nfe':  lambda v: f"NFE {v}",
                'val_time': lambda v: f"time {v:.4f}s",
            },
            extra_parts=[
                f"EarlyStopping {self.early_stopping_callback.wait_count}/"
                f"{self.early_stopping_callback.patience - 1}"
            ],
            use_max_epoch=False
        )
        
    def _likelihood_tabular_test(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            prefix="Test",
            metrics=self._get_metrics(trainer, ['test_nll', 'test_bpd', 'test_nfe', 'test_time']),
            formatters={
                'test_nll':  lambda v: f"NLL {v:.4f}",
                'test_bpd':  lambda v: f"BPD {v:.4f}",
                'test_nfe':  lambda v: f"NFE {v}",
                'test_time': lambda v: f"time {v:.4f}s",
            },
            use_max_epoch=False
        )
    
    def _likelihood_real_image_train(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['train_loss', 'train_nll', 'train_bpd']),
            formatters={
                'train_loss': lambda v: f"Loss {v:.4f}",
                'train_nll':  lambda v: f"NLL {v:.4f}",
                'train_bpd':  lambda v: f"BPD {v:.4f}",
            },
            global_zero=True
        )
        
    def _likelihood_real_image_val(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['val_nll', 'val_bpd', 'val_nfe', 'val_time']),
            formatters={
                'val_nll':  lambda v: f"Eval NLL {v:.4f}",
                'val_bpd':  lambda v: f"Eval BPD {v:.4f}",
                'val_nfe':  lambda v: f"NFE {v}",
                'val_time': lambda v: f"time {v:.4f}s",
            },
            global_zero=True
        )
        
    def _likelihood_real_image_test(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['test_nll', 'test_bpd', 'test_nfe', 'test_time']),
            formatters={
                'test_nll':  lambda v: f"NLL {v:.4f}",
                'test_bpd':  lambda v: f"BPD {v:.4f}",
                'test_nfe':  lambda v: f"NFE {v}",
                'test_time': lambda v: f"time {v:.4f}s",
            },
            use_max_epoch=False,
            global_zero=True
        )
            
    def _likelihood_other_train(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['train_loss', 'train_nll']),
            formatters={
                'train_loss': lambda v: f"Loss {v:.4f}",
                'train_nll':  lambda v: f"NLL {v:.4f}",
            }
        )
    
    def _likelihood_other_val(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['val_nfe', 'val_time']),
            formatters={
                'val_nfe':  lambda v: f"NFE {v}",
                'val_time': lambda v: f"time {v:.4f}s",
            }
        )
    
    # =====================================================
    # =================== f-div ===========================
    # =====================================================
    def _fdiv_mi_train(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['train_loss']),
            formatters={'train_loss': lambda v: f"Loss {v:.4f}"}
        )
    
    def _fdiv_val(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(
                trainer, ['val_est_fdiv', 'val_true_fdiv', 'val_mae', 'val_nfe']
            ),
            formatters={
                'val_est_fdiv':  lambda v: f"est. fDiv {v:.4f}",
                'val_true_fdiv': lambda v: f"true fDiv {v:.4f}",
                'val_mae':       lambda v: f"MAE {v:.4f}",
                'val_nfe':       lambda v: f"NFE {v}",
            }
        )
    
    # =====================================================
    # =================== OOD Detection ==================
    # =====================================================
    def _ood_train(self, trainer, pl_module):
        self._log_epoch(
            trainer=trainer,
            metrics=self._get_metrics(trainer, ['train_loss']),
            formatters={'train_loss': lambda v: f"Loss {v:.4f}"}
        )
    
    def _ood_val(self, trainer, pl_module):
        # OOD validation is logged by the model in on_validation_epoch_end (pl_models/ood_detection.py).
        # Log early stopping counter here after the model has logged its metrics.
        if self.early_stopping_callback and trainer.is_global_zero:
            wait_count = self.early_stopping_callback.wait_count
            patience = self.early_stopping_callback.patience
            best_score = self.early_stopping_callback.best_score
            self.logger.info(f"EarlyStopping counter: {wait_count}/{patience} (best near-AUROC: {best_score:.4f})")
    
    def _ood_test(self, trainer, pl_module):
        # OOD test is logged by the model in on_test_epoch_end; skip here to avoid duplicate/stale lines.
        pass
    
    # =====================================================
    # ================= Lightning hooks ===================
    # =====================================================
    def on_train_epoch_end(self, trainer, pl_module):
        self._handle_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._handle_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._handle_test_epoch_end(trainer, pl_module)
        
class ShiftControllerCallback(pl.Callback):
    def __init__(self, datamodule, reload_interval):
        self.dm = datamodule
        self.reload_interval = reload_interval

    def on_train_epoch_start(self, trainer, pl_module):
        new_step = trainer.current_epoch // self.reload_interval
        if new_step != self.dm.generator.get_current_step():
            print(f"[Shift] set_step({new_step}), current_epoch {trainer.current_epoch}")
            self.dm.generator.set_step(new_step)
    
class NullLogger(pl.loggers.Logger):
    @property
    def name(self):
        return "NullLogger"
    @property
    def version(self):
        return "0.0"
    def log_metrics(self, *args, **kwargs):
        pass
    def log_hyperparams(self, *args, **kwargs):
        pass

def setup_silent_mode():
    # setup silent mode
    logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.CRITICAL)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.CRITICAL)
    logging.getLogger("pytorch_lightning.trainer.connectors.data_connector").setLevel(logging.CRITICAL)
    
    # filter warnings
    warnings.filterwarnings("ignore")
    
    # set environment variables
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['PL_DISABLE_FORK_WARNING'] = '1'