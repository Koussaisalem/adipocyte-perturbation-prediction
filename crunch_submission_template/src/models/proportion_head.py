"""
Proportion Head for Multi-Task Learning.

Predicts cell program proportions (pre_adipo, adipo, lipo, other)
and the lipo_adipo ratio from perturbation embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProportionHead(nn.Module):
    """
    Multi-task head for predicting cell program proportions.
    
    Outputs:
    - State proportions: [pre_adipo, adipo, lipo, other] (softmax)
    - Lipo/adipo ratio (sigmoid, scaled)
    """
    
    def __init__(
        self,
        perturbation_dim: int = 256,
        hidden_dims: list[int] = [128, 64],
        n_programs: int = 4,
        predict_ratio: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize the proportion head.
        
        Args:
            perturbation_dim: Dimension of perturbation embedding
            hidden_dims: Hidden layer dimensions
            n_programs: Number of cell programs (4 for pre_adipo, adipo, lipo, other)
            predict_ratio: Whether to predict lipo_adipo ratio
            dropout: Dropout rate
        """
        super().__init__()
        
        self.perturbation_dim = perturbation_dim
        self.n_programs = n_programs
        self.predict_ratio = predict_ratio
        
        # Shared hidden layers
        layers = []
        prev_dim = perturbation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # State proportion head
        self.state_head = nn.Linear(prev_dim, n_programs)
        
        # Ratio head
        if predict_ratio:
            self.ratio_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        z_p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict proportions from perturbation embedding.
        
        Args:
            z_p: Perturbation embedding (batch_size, perturbation_dim)
            
        Returns:
            Dictionary with:
            - 'state_proportions': (batch_size, n_programs) with softmax
            - 'lipo_adipo_ratio': (batch_size, 1) if predict_ratio=True
        """
        # Shared features
        h = self.shared_layers(z_p)
        
        # State proportions (softmax)
        state_logits = self.state_head(h)
        state_probs = F.softmax(state_logits, dim=-1)
        
        result = {
            "state_proportions": state_probs,
        }
        
        # Lipo/adipo ratio
        if self.predict_ratio:
            ratio_logit = self.ratio_head(h)
            # Sigmoid to [0, 1], then scale to reasonable range
            ratio = torch.sigmoid(ratio_logit)
            result["lipo_adipo_ratio"] = ratio
        
        return result
    
    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
        state_weight: float = 0.75,
        ratio_weight: float = 0.25,
        loss_fn: str = "l1",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute multi-task proportion loss.
        
        Args:
            predictions: Output from forward()
            targets: Ground truth proportions (batch_size, n_programs + 1)
                     Columns: [pre_adipo, adipo, lipo, other, lipo_adipo_ratio]
            state_weight: Weight for state proportion loss
            ratio_weight: Weight for ratio loss
            loss_fn: Loss function ('l1', 'l2', 'huber')
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Split targets
        state_targets = targets[:, :self.n_programs]
        
        # State proportion loss
        if loss_fn == "l1":
            state_loss = F.l1_loss(predictions["state_proportions"], state_targets)
        elif loss_fn == "l2":
            state_loss = F.mse_loss(predictions["state_proportions"], state_targets)
        elif loss_fn == "huber":
            state_loss = F.smooth_l1_loss(predictions["state_proportions"], state_targets)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        loss_dict = {"state_loss": state_loss}
        total_loss = state_weight * state_loss
        
        # Ratio loss
        if self.predict_ratio and targets.shape[1] > self.n_programs:
            ratio_targets = targets[:, self.n_programs:self.n_programs + 1]
            
            if loss_fn == "l1":
                ratio_loss = F.l1_loss(predictions["lipo_adipo_ratio"], ratio_targets)
            elif loss_fn == "l2":
                ratio_loss = F.mse_loss(predictions["lipo_adipo_ratio"], ratio_targets)
            elif loss_fn == "huber":
                ratio_loss = F.smooth_l1_loss(predictions["lipo_adipo_ratio"], ratio_targets)
            
            loss_dict["ratio_loss"] = ratio_loss
            total_loss = total_loss + ratio_weight * ratio_loss
        
        loss_dict["total_proportion_loss"] = total_loss
        
        return total_loss, loss_dict


class JointProportionHead(nn.Module):
    """
    Joint head that predicts proportions directly from the flow field.
    
    Instead of using z_p, this aggregates information from generated cells.
    """
    
    def __init__(
        self,
        x_dim: int = 500,
        hidden_dims: list[int] = [256, 128],
        n_programs: int = 4,
        predict_ratio: bool = True,
    ):
        """
        Initialize joint proportion head.
        
        Args:
            x_dim: Dimension of cell state (PCA components)
            hidden_dims: Hidden layer dimensions
            n_programs: Number of cell programs
            predict_ratio: Whether to predict ratio
        """
        super().__init__()
        
        self.x_dim = x_dim
        self.n_programs = n_programs
        self.predict_ratio = predict_ratio
        
        # Encoder for cells
        layers = []
        prev_dim = x_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.cell_encoder = nn.Sequential(*layers)
        
        # Proportion head
        self.state_head = nn.Linear(prev_dim, n_programs)
        
        if predict_ratio:
            self.ratio_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        cells: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict proportions from generated cells.
        
        Args:
            cells: Generated cell states (n_cells, x_dim)
            
        Returns:
            Dictionary with state_proportions and lipo_adipo_ratio
        """
        # Encode cells
        h = self.cell_encoder(cells)
        
        # Aggregate across cells (mean pooling)
        h_agg = h.mean(dim=0, keepdim=True)
        
        # Predict proportions
        state_logits = self.state_head(h_agg)
        state_probs = F.softmax(state_logits, dim=-1)
        
        result = {
            "state_proportions": state_probs,
        }
        
        if self.predict_ratio:
            ratio = torch.sigmoid(self.ratio_head(h_agg))
            result["lipo_adipo_ratio"] = ratio
        
        return result


class ProgramClassifier(nn.Module):
    """
    Per-cell program classifier.
    
    Assigns discrete program labels to individual cells based on their state.
    Useful for computing proportions from generated cells.
    """
    
    def __init__(
        self,
        x_dim: int = 500,
        n_programs: int = 4,
        hidden_dim: int = 128,
    ):
        """
        Initialize program classifier.
        
        Args:
            x_dim: Cell state dimension
            n_programs: Number of programs
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_programs),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify cells into programs.
        
        Args:
            x: Cell states (batch_size, x_dim)
            
        Returns:
            Program logits (batch_size, n_programs)
        """
        return self.classifier(x)
    
    def predict_proportions(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict program proportions from a batch of cells.
        
        Args:
            x: Cell states (n_cells, x_dim)
            
        Returns:
            Proportions (1, n_programs)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        proportions = probs.mean(dim=0, keepdim=True)
        return proportions
