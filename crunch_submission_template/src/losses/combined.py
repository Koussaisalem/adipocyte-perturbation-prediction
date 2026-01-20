"""
Combined Loss for Training.

Combines CFM loss, MMD loss, Pearson delta loss, and proportion loss
into a unified training objective.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmd import MMDLoss
from .pearson_delta import PearsonDeltaLoss, compute_perturbation_delta

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined multi-task loss for perturbation prediction.
    
    Components:
    1. CFM loss: Velocity prediction error
    2. MMD loss: Distribution matching (computed periodically)
    3. Pearson delta loss: Directional shift alignment
    4. Proportion loss: Cell program proportion prediction
    """
    
    def __init__(
        self,
        cfm_weight: float = 1.0,
        mmd_weight: float = 0.1,
        pearson_weight: float = 0.05,
        proportion_weight: float = 0.5,
        mmd_bandwidths: list[float] = [581.5, 1163.0, 2326.0, 4652.0, 9304.0],
        proportion_state_weight: float = 0.75,
        proportion_ratio_weight: float = 0.25,
        proportion_loss_fn: str = "l1",
    ):
        """
        Initialize combined loss.
        
        Args:
            cfm_weight: Weight for flow matching velocity loss
            mmd_weight: Weight for MMD distribution loss
            pearson_weight: Weight for Pearson delta loss
            proportion_weight: Weight for proportion prediction loss
            mmd_bandwidths: Bandwidths for MMD Gaussian kernel
            proportion_state_weight: Weight for state proportion loss
            proportion_ratio_weight: Weight for lipo_adipo ratio loss
            proportion_loss_fn: Loss function for proportions ('l1', 'l2', 'huber')
        """
        super().__init__()
        
        self.cfm_weight = cfm_weight
        self.mmd_weight = mmd_weight
        self.pearson_weight = pearson_weight
        self.proportion_weight = proportion_weight
        
        self.proportion_state_weight = proportion_state_weight
        self.proportion_ratio_weight = proportion_ratio_weight
        self.proportion_loss_fn = proportion_loss_fn
        
        # Loss modules
        self.mmd_loss = MMDLoss(bandwidths=mmd_bandwidths)
        self.pearson_loss = PearsonDeltaLoss()
    
    def compute_cfm_loss(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching velocity loss.
        
        Args:
            predicted_velocity: Predicted velocity
            target_velocity: Target velocity (x1 - x0)
            
        Returns:
            CFM loss
        """
        return F.mse_loss(predicted_velocity, target_velocity)
    
    def compute_mmd_loss(
        self,
        predicted_cells: torch.Tensor,
        target_cells: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMD distribution loss.
        
        Args:
            predicted_cells: Generated perturbed cells
            target_cells: True perturbed cells
            
        Returns:
            MMD loss
        """
        return self.mmd_loss(predicted_cells, target_cells)
    
    def compute_pearson_loss(
        self,
        predicted_cells: torch.Tensor,
        target_cells: torch.Tensor,
        control_cells: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Pearson delta loss.
        
        Args:
            predicted_cells: Generated perturbed cells
            target_cells: True perturbed cells
            control_cells: Control (NC) cells
            
        Returns:
            Pearson delta loss
        """
        # Compute deltas
        predicted_delta = compute_perturbation_delta(predicted_cells, control_cells)
        target_delta = compute_perturbation_delta(target_cells, control_cells)
        
        # Add batch dimension if needed
        if predicted_delta.dim() == 1:
            predicted_delta = predicted_delta.unsqueeze(0)
            target_delta = target_delta.unsqueeze(0)
        
        return self.pearson_loss(predicted_delta, target_delta)
    
    def compute_proportion_loss(
        self,
        predicted_proportions: dict[str, torch.Tensor],
        target_proportions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute proportion prediction loss.
        
        Args:
            predicted_proportions: Dict with 'state_proportions' and 'lipo_adipo_ratio'
            target_proportions: Target proportions (batch_size, n_programs + 1)
            
        Returns:
            Tuple of (total_proportion_loss, loss_dict)
        """
        # State proportion loss
        state_pred = predicted_proportions["state_proportions"]
        state_target = target_proportions[:, :4]  # First 4 columns
        
        if self.proportion_loss_fn == "l1":
            state_loss = F.l1_loss(state_pred, state_target)
        elif self.proportion_loss_fn == "l2":
            state_loss = F.mse_loss(state_pred, state_target)
        elif self.proportion_loss_fn == "huber":
            state_loss = F.smooth_l1_loss(state_pred, state_target)
        else:
            raise ValueError(f"Unknown loss: {self.proportion_loss_fn}")
        
        loss_dict = {"state_loss": state_loss}
        total_loss = self.proportion_state_weight * state_loss
        
        # Ratio loss
        if "lipo_adipo_ratio" in predicted_proportions and target_proportions.shape[1] > 4:
            ratio_pred = predicted_proportions["lipo_adipo_ratio"]
            ratio_target = target_proportions[:, 4:5]  # Last column
            
            if self.proportion_loss_fn == "l1":
                ratio_loss = F.l1_loss(ratio_pred, ratio_target)
            elif self.proportion_loss_fn == "l2":
                ratio_loss = F.mse_loss(ratio_pred, ratio_target)
            elif self.proportion_loss_fn == "huber":
                ratio_loss = F.smooth_l1_loss(ratio_pred, ratio_target)
            
            loss_dict["ratio_loss"] = ratio_loss
            total_loss = total_loss + self.proportion_ratio_weight * ratio_loss
        
        return total_loss, loss_dict
    
    def forward(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        predicted_proportions: Optional[dict[str, torch.Tensor]] = None,
        target_proportions: Optional[torch.Tensor] = None,
        predicted_cells: Optional[torch.Tensor] = None,
        target_cells: Optional[torch.Tensor] = None,
        control_cells: Optional[torch.Tensor] = None,
        compute_mmd: bool = False,
        compute_pearson: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            predicted_velocity: Predicted velocity from flow matching
            target_velocity: Target velocity (x1 - x0)
            predicted_proportions: Predicted proportions (optional)
            target_proportions: Target proportions (optional)
            predicted_cells: Generated cells for MMD/Pearson (optional)
            target_cells: True perturbed cells (optional)
            control_cells: NC cells (optional)
            compute_mmd: Whether to compute MMD loss (expensive)
            compute_pearson: Whether to compute Pearson loss
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. CFM velocity loss (always computed)
        cfm_loss = self.compute_cfm_loss(predicted_velocity, target_velocity)
        loss_dict["cfm_loss"] = cfm_loss
        total_loss = total_loss + self.cfm_weight * cfm_loss
        
        # 2. MMD loss (computed periodically)
        if compute_mmd and predicted_cells is not None and target_cells is not None:
            mmd = self.compute_mmd_loss(predicted_cells, target_cells)
            loss_dict["mmd_loss"] = mmd
            total_loss = total_loss + self.mmd_weight * mmd
        
        # 3. Pearson delta loss
        if compute_pearson and predicted_cells is not None and target_cells is not None and control_cells is not None:
            pearson = self.compute_pearson_loss(predicted_cells, target_cells, control_cells)
            loss_dict["pearson_loss"] = pearson
            total_loss = total_loss + self.pearson_weight * pearson
        
        # 4. Proportion loss
        if predicted_proportions is not None and target_proportions is not None:
            prop_loss, prop_dict = self.compute_proportion_loss(
                predicted_proportions,
                target_proportions,
            )
            loss_dict.update(prop_dict)
            loss_dict["proportion_loss"] = prop_loss
            total_loss = total_loss + self.proportion_weight * prop_loss
        
        loss_dict["total_loss"] = total_loss
        
        return total_loss, loss_dict


def create_loss_from_config(config: dict) -> CombinedLoss:
    """
    Create combined loss from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized CombinedLoss
    """
    return CombinedLoss(
        cfm_weight=config["loss"]["cfm_weight"],
        mmd_weight=config["loss"]["mmd_weight"],
        pearson_weight=config["loss"]["pearson_weight"],
        proportion_weight=config["loss"]["proportion_weight"],
        mmd_bandwidths=config["loss"]["mmd_bandwidths"],
        proportion_state_weight=config["loss"]["state_weight"],
        proportion_ratio_weight=config["loss"]["ratio_weight"],
        proportion_loss_fn=config["loss"]["proportion_loss"],
    )
