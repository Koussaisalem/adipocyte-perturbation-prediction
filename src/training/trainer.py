"""
Trainer class for perturbation prediction model.

Handles training loop, validation, and metric tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    
    # Training loop
    epochs: int = 100
    batch_size: int = 64
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Validation
    val_frequency: int = 1
    n_val_samples: int = 100
    
    # Loss weights
    cfm_weight: float = 1.0
    mmd_weight: float = 0.1
    pearson_weight: float = 0.05
    proportion_weight: float = 0.5
    
    # MMD computation
    mmd_frequency: int = 10  # Compute MMD every N steps
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    monitor: str = "val/mmd"
    mode: str = "min"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42


class Trainer:
    """
    Trainer for perturbation flow model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        callbacks: Optional[list] = None,
        logger_obj: Optional[object] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PerturbationFlowModel
            loss_fn: CombinedLoss
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            optimizer: Optional pre-initialized optimizer
            scheduler: Optional learning rate scheduler
            callbacks: Optional list of callbacks (checkpointing, early stopping)
            logger_obj: Optional experiment logger (wandb, mlflow)
        """
        self.model = model.to(config.device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger_obj = logger_obj
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf') if config.mode == 'min' else float('-inf')
        
        logger.info(f"Trainer initialized on device: {config.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'cfm_loss': 0.0,
            'mmd_loss': 0.0,
            'pearson_loss': 0.0,
            'proportion_loss': 0.0,
            'total_loss': 0.0,
        }
        
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            xt = batch['xt'].to(self.config.device)
            t = batch['t'].to(self.config.device)
            velocity = batch['velocity'].to(self.config.device)
            
            # Get perturbation names and encode them
            # Note: This requires knowledge graph and node features
            # For now, we'll assume perturbation embeddings are pre-computed
            # or we'll need to modify the batch to include z_p
            
            # Forward pass (simplified - assuming z_p is in batch)
            # In practice, you'll need to encode perturbations via the graph
            if 'z_p' in batch:
                z_p = batch['z_p'].to(self.config.device)
            else:
                # This is a placeholder - implement proper perturbation encoding
                raise NotImplementedError("Perturbation encoding not implemented in batch")
            
            # Predict velocity
            pred_velocity = self.model.predict_velocity(xt, t, z_p)
            
            # Compute CFM loss
            loss, loss_dict = self.loss_fn(
                predicted_velocity=pred_velocity,
                target_velocity=velocity,
                predicted_proportions=None,  # Add proportion prediction if available
                target_proportions=None,
                compute_mmd=(batch_idx % self.config.mmd_frequency == 0),
                compute_pearson=False,
            )
            
            # Backward pass
            loss = loss / self.config.accumulate_grad_batches
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
            
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'cfm': loss_dict['cfm_loss'].item(),
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = {
            'cfm_loss': 0.0,
            'mmd_loss': 0.0,
            'proportion_loss': 0.0,
            'total_loss': 0.0,
        }
        
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            xt = batch['xt'].to(self.config.device)
            t = batch['t'].to(self.config.device)
            velocity = batch['velocity'].to(self.config.device)
            
            if 'z_p' in batch:
                z_p = batch['z_p'].to(self.config.device)
            else:
                raise NotImplementedError("Perturbation encoding not implemented")
            
            # Predict velocity
            pred_velocity = self.model.predict_velocity(xt, t, z_p)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                predicted_velocity=pred_velocity,
                target_velocity=velocity,
                compute_mmd=True,  # Always compute MMD in validation
                compute_pearson=False,
            )
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in val_losses:
                    val_losses[key] += value.item()
            
            n_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def fit(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log training metrics
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['total_loss']:.4f}")
            
            if self.logger_obj is not None:
                for key, value in train_metrics.items():
                    self.logger_obj.log({f"train/{key}": value}, step=epoch)
            
            # Validation
            if epoch % self.config.val_frequency == 0:
                val_metrics = self.validate()
                
                logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['total_loss']:.4f}")
                logger.info(f"Epoch {epoch} - Val MMD: {val_metrics.get('mmd_loss', 0.0):.4f}")
                
                if self.logger_obj is not None:
                    for key, value in val_metrics.items():
                        self.logger_obj.log({f"val/{key}": value}, step=epoch)
                
                # Callbacks
                metrics_dict = {f"val/{k}": v for k, v in val_metrics.items()}
                
                for callback in self.callbacks:
                    if hasattr(callback, 'on_validation_end'):
                        if callback.__class__.__name__ == 'EarlyStopping':
                            should_stop = callback.on_validation_end(metrics_dict)
                            if should_stop:
                                logger.info("Early stopping triggered!")
                                break
                        else:
                            callback.on_validation_end(
                                self.model,
                                self.optimizer,
                                epoch,
                                metrics_dict,
                            )
            
            # Check early stopping
            if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                break
        
        # Training finished
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes")
        
        # Final callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(
                    self.model,
                    self.optimizer,
                    self.current_epoch,
                    {},
                )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
