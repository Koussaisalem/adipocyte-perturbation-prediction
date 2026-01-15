"""
Maximum Mean Discrepancy (MMD) Loss.

Measures the distance between two distributions using kernel embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: list[float],
) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel with multiple bandwidths.
    
    K(x, y) = sum_σ exp(-||x - y||^2 / (2σ^2))
    
    Args:
        x: Samples from first distribution (n, d)
        y: Samples from second distribution (m, d)
        bandwidths: List of bandwidth parameters σ
        
    Returns:
        Kernel matrix (n, m)
    """
    # Compute pairwise squared distances
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (n, 1)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)  # (m, 1)
    xy = torch.mm(x, y.t())  # (n, m)
    
    dist_sq = xx + yy.t() - 2 * xy  # (n, m)
    dist_sq = torch.clamp(dist_sq, min=0.0)  # Numerical stability
    
    # Sum over multiple bandwidths
    K = torch.zeros_like(dist_sq)
    
    for bandwidth in bandwidths:
        gamma = 1.0 / (2 * bandwidth ** 2)
        K = K + torch.exp(-gamma * dist_sq)
    
    return K


def mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: list[float] = [581.5, 1163.0, 2326.0, 4652.0, 9304.0],
) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy between two distributions.
    
    MMD^2(X, Y) = E[K(x, x')] - 2E[K(x, y)] + E[K(y, y')]
    
    Args:
        x: Samples from first distribution (n, d)
        y: Samples from second distribution (m, d)
        bandwidths: List of bandwidth parameters for Gaussian kernel
        
    Returns:
        MMD loss (scalar)
    """
    # Compute kernel matrices
    K_xx = gaussian_kernel(x, x, bandwidths)
    K_yy = gaussian_kernel(y, y, bandwidths)
    K_xy = gaussian_kernel(x, y, bandwidths)
    
    # Compute MMD^2
    # E[K(x, x')] = mean of off-diagonal elements
    n = x.shape[0]
    m = y.shape[0]
    
    # Remove diagonal for unbiased estimate
    K_xx_sum = K_xx.sum() - K_xx.diag().sum()
    K_yy_sum = K_yy.sum() - K_yy.diag().sum()
    
    mmd_sq = (
        K_xx_sum / (n * (n - 1))
        + K_yy_sum / (m * (m - 1))
        - 2 * K_xy.mean()
    )
    
    # Ensure non-negative (numerical stability)
    mmd_sq = torch.clamp(mmd_sq, min=0.0)
    
    return torch.sqrt(mmd_sq)


class MMDLoss(nn.Module):
    """
    MMD Loss module.
    
    Configurable bandwidths for multi-scale kernel.
    """
    
    def __init__(
        self,
        bandwidths: list[float] = [581.5, 1163.0, 2326.0, 4652.0, 9304.0],
    ):
        """
        Initialize MMD loss.
        
        Args:
            bandwidths: List of bandwidth parameters
        """
        super().__init__()
        self.bandwidths = bandwidths
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMD loss.
        
        Args:
            x: Predicted distribution samples
            y: Target distribution samples
            
        Returns:
            MMD loss
        """
        return mmd_loss(x, y, self.bandwidths)


class ConditionalMMDLoss(nn.Module):
    """
    Conditional MMD loss for perturbation-specific distributions.
    
    Computes MMD separately for each perturbation in a batch.
    """
    
    def __init__(
        self,
        bandwidths: list[float] = [581.5, 1163.0, 2326.0, 4652.0, 9304.0],
        aggregate: str = "mean",
    ):
        """
        Initialize conditional MMD loss.
        
        Args:
            bandwidths: Bandwidth parameters
            aggregate: How to aggregate MMDs ('mean', 'sum', 'max')
        """
        super().__init__()
        self.bandwidths = bandwidths
        self.aggregate = aggregate
    
    def forward(
        self,
        predicted: list[torch.Tensor],
        target: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute MMD for each perturbation.
        
        Args:
            predicted: List of predicted distributions (one per perturbation)
            target: List of target distributions
            
        Returns:
            Aggregated MMD loss
        """
        mmds = []
        
        for pred, tgt in zip(predicted, target):
            mmd = mmd_loss(pred, tgt, self.bandwidths)
            mmds.append(mmd)
        
        mmds = torch.stack(mmds)
        
        if self.aggregate == "mean":
            return mmds.mean()
        elif self.aggregate == "sum":
            return mmds.sum()
        elif self.aggregate == "max":
            return mmds.max()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregate}")


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances.
    
    Args:
        x: First set of points (n, d)
        y: Second set of points (m, d)
        
    Returns:
        Distance matrix (n, m)
    """
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    xy = torch.mm(x, y.t())
    
    distances = xx + yy.t() - 2 * xy
    distances = torch.clamp(distances, min=0.0)
    
    return torch.sqrt(distances)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute energy distance between two distributions.
    
    Alternative to MMD, based on pairwise distances.
    
    Args:
        x: Samples from first distribution
        y: Samples from second distribution
        
    Returns:
        Energy distance
    """
    n = x.shape[0]
    m = y.shape[0]
    
    # Compute pairwise distances
    dxy = pairwise_distances(x, y).mean()
    dxx = pairwise_distances(x, x).sum() / (n * (n - 1))
    dyy = pairwise_distances(y, y).sum() / (m * (m - 1))
    
    energy = 2 * dxy - dxx - dyy
    
    return energy
