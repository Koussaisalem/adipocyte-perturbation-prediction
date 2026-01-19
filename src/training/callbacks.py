"""
Training callbacks for checkpointing and early stopping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    
    Saves top-k models based on a monitored metric.
    """
    
    def __init__(
        self,
        dirpath: str | Path,
        monitor: str = "val/mmd",
        mode: str = "min",
        save_top_k: int = 3,
        filename: str = "model-{epoch:03d}-{val_mmd:.4f}",
        save_last: bool = True,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            dirpath: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Number of best models to keep
            filename: Checkpoint filename pattern
            save_last: Whether to save last checkpoint
        """
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.save_last = save_last
        
        self.best_models = []  # List of (metric_value, filepath)
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ):
        """Save a checkpoint."""
        # Format filename - convert metric names for format string
        format_metrics = {k.replace('/', '_'): v for k, v in metrics.items()}
        format_metrics['epoch'] = epoch
        filename = self.filename.format(**format_metrics)
        
        filepath = self.dirpath / f"{filename}.ckpt"
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        if is_best:
            best_path = self.dirpath / "best.ckpt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def on_validation_end(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ):
        """Called at the end of validation."""
        # Get monitored metric
        current_metric = metrics.get(self.monitor)
        
        if current_metric is None:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return
        
        # Check if this is a best model
        is_best = self._is_better(current_metric, self.best_metric)
        
        if is_best:
            self.best_metric = current_metric
        
        # Save checkpoint
        self.save_checkpoint(model, optimizer, epoch, metrics, is_best)
        
        # Manage top-k models
        self.best_models.append((current_metric, epoch))
        self.best_models.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
        
        # Remove worst models if exceeding top-k
        if len(self.best_models) > self.save_top_k:
            # Keep top-k
            self.best_models = self.best_models[:self.save_top_k]
    
    def on_train_end(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ):
        """Called at the end of training."""
        if self.save_last:
            last_path = self.dirpath / "last.ckpt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }
            torch.save(checkpoint, last_path)
            logger.info(f"Saved last checkpoint: {last_path}")


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = "val/mmd",
        mode: str = "min",
        patience: int = 15,
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)
    
    def on_validation_end(self, metrics: dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            metrics: Validation metrics
            
        Returns:
            True if training should stop
        """
        current_metric = metrics.get(self.monitor)
        
        if current_metric is None:
            logger.warning(f"Metric {self.monitor} not found")
            return False
        
        if self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered! Best {self.monitor}: {self.best_metric:.4f}")
                self.should_stop = True
                return True
        
        return False


class LearningRateScheduler:
    """
    Adjust learning rate during training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "cosine_warmup",
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        initial_lr: float = 1e-4,
    ):
        """
        Initialize LR scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            mode: Scheduling mode ('cosine_warmup', 'step', 'plateau')
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
            initial_lr: Initial learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # Explicitly convert to float in case they come from YAML as strings
        self.min_lr = float(min_lr)
        self.initial_lr = float(initial_lr)
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        
        if self.mode == "cosine_warmup":
            # Cosine annealing after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return lr.item()
        
        return self.initial_lr
    
    def step(self, epoch: int):
        """Update learning rate."""
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        logger.debug(f"Learning rate: {lr:.6f}")
