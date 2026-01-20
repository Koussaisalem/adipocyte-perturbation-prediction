"""
Pearson Delta Loss.

Measures the directional shift correlation between predicted
and true perturbation effects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        x: First variable (batch_size, n_features) or (n_features,)
        y: Second variable (same shape as x)
        dim: Dimension along which to compute correlation
        
    Returns:
        Pearson correlation coefficient(s)
    """
    # Center the data
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    
    # Compute correlation
    numerator = (x_centered * y_centered).sum(dim=dim)
    denominator = torch.sqrt(
        (x_centered ** 2).sum(dim=dim) * (y_centered ** 2).sum(dim=dim)
    )
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    
    correlation = numerator / denominator
    
    return correlation


def pearson_delta_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Pearson delta loss.
    
    The delta is defined as: Δ = X_perturbed - X_control
    
    Loss = 1 - pearson(predicted_Δ, target_Δ)
    
    This encourages the predicted perturbation effect to have the
    correct direction in gene expression space.
    
    Args:
        predicted_delta: Predicted perturbation effect (batch_size, n_genes)
        target_delta: True perturbation effect (batch_size, n_genes)
        
    Returns:
        Pearson delta loss (scalar)
    """
    # Compute Pearson correlation
    correlation = pearson_correlation(predicted_delta, target_delta, dim=-1)
    
    # Average over batch
    correlation = correlation.mean()
    
    # Loss: 1 - correlation (higher correlation = lower loss)
    loss = 1.0 - correlation
    
    return loss


class PearsonDeltaLoss(nn.Module):
    """
    Pearson Delta Loss module.
    
    Measures directional similarity of perturbation effects.
    """
    
    def __init__(
        self,
        use_cosine: bool = False,
    ):
        """
        Initialize Pearson delta loss.
        
        Args:
            use_cosine: If True, use cosine similarity instead of Pearson
        """
        super().__init__()
        self.use_cosine = use_cosine
    
    def forward(
        self,
        predicted_delta: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predicted_delta: Predicted Δ = X_pred - X_control
            target_delta: True Δ = X_true - X_control
            
        Returns:
            Loss value
        """
        if self.use_cosine:
            # Cosine similarity
            similarity = F.cosine_similarity(
                predicted_delta,
                target_delta,
                dim=-1,
            ).mean()
            loss = 1.0 - similarity
        else:
            # Pearson correlation
            loss = pearson_delta_loss(predicted_delta, target_delta)
        
        return loss


class DirectionalLoss(nn.Module):
    """
    Directional loss combining multiple metrics.
    
    Measures both magnitude and direction of perturbation effects.
    """
    
    def __init__(
        self,
        pearson_weight: float = 0.5,
        cosine_weight: float = 0.3,
        magnitude_weight: float = 0.2,
    ):
        """
        Initialize directional loss.
        
        Args:
            pearson_weight: Weight for Pearson correlation
            cosine_weight: Weight for cosine similarity
            magnitude_weight: Weight for magnitude similarity
        """
        super().__init__()
        self.pearson_weight = pearson_weight
        self.cosine_weight = cosine_weight
        self.magnitude_weight = magnitude_weight
    
    def forward(
        self,
        predicted_delta: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined directional loss.
        
        Args:
            predicted_delta: Predicted perturbation effect
            target_delta: True perturbation effect
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Pearson correlation
        pearson_corr = pearson_correlation(predicted_delta, target_delta, dim=-1).mean()
        pearson_loss = 1.0 - pearson_corr
        
        # Cosine similarity
        cosine_sim = F.cosine_similarity(predicted_delta, target_delta, dim=-1).mean()
        cosine_loss = 1.0 - cosine_sim
        
        # Magnitude similarity (relative L2 error)
        pred_mag = torch.norm(predicted_delta, dim=-1)
        tgt_mag = torch.norm(target_delta, dim=-1)
        mag_loss = F.l1_loss(pred_mag, tgt_mag) / (tgt_mag.mean() + 1e-8)
        
        # Combined loss
        total_loss = (
            self.pearson_weight * pearson_loss
            + self.cosine_weight * cosine_loss
            + self.magnitude_weight * mag_loss
        )
        
        loss_dict = {
            "pearson_loss": pearson_loss,
            "pearson_corr": pearson_corr,
            "cosine_loss": cosine_loss,
            "cosine_sim": cosine_sim,
            "magnitude_loss": mag_loss,
        }
        
        return total_loss, loss_dict


def compute_perturbation_delta(
    perturbed_cells: torch.Tensor,
    control_cells: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Compute the perturbation effect Δ.
    
    Δ = aggregate(X_perturbed) - aggregate(X_control)
    
    Args:
        perturbed_cells: Perturbed cell states (n_cells, n_features)
        control_cells: Control cell states (m_cells, n_features)
        method: Aggregation method ('mean', 'median')
        
    Returns:
        Perturbation delta (n_features,)
    """
    if method == "mean":
        delta = perturbed_cells.mean(dim=0) - control_cells.mean(dim=0)
    elif method == "median":
        delta = perturbed_cells.median(dim=0)[0] - control_cells.median(dim=0)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return delta
