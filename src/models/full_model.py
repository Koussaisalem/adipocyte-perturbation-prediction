"""
Full Perturbation Flow Model.

Combines GATv2 encoder, flow matching decoder, and proportion head
into a unified architecture.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .gat_encoder import PerturbationEncoder
from .flow_matching import FlowMatchingDecoder
from .proportion_head import ProportionHead

logger = logging.getLogger(__name__)


class PerturbationFlowModel(nn.Module):
    """
    Complete model for zero-shot perturbation prediction.
    
    Architecture:
    1. PerturbationEncoder: Maps gene knockout -> perturbation embedding (z_p)
    2. FlowMatchingDecoder: Transforms NC cells -> perturbed cells using z_p
    3. ProportionHead: Predicts cell program proportions from z_p
    """
    
    def __init__(
        self,
        perturbation_encoder: PerturbationEncoder,
        flow_decoder: FlowMatchingDecoder,
        proportion_head: Optional[ProportionHead] = None,
    ):
        """
        Initialize the full model.
        
        Args:
            perturbation_encoder: Encoder for perturbation embeddings
            flow_decoder: Flow matching decoder
            proportion_head: Optional proportion prediction head
        """
        super().__init__()
        
        self.perturbation_encoder = perturbation_encoder
        self.flow_decoder = flow_decoder
        self.proportion_head = proportion_head
    
    def encode_perturbation(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        perturbation_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode perturbations into latent vectors.
        
        Args:
            node_features: Gene node features
            edge_index: Graph edges
            perturbation_indices: Indices of perturbed genes
            
        Returns:
            Perturbation embeddings (batch_size, perturbation_dim)
        """
        return self.perturbation_encoder(
            node_features,
            edge_index,
            perturbation_indices,
        )
    
    def predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity at state x, time t, conditioned on perturbation z_p.
        
        Args:
            x: Current state
            t: Time
            z_p: Perturbation embedding
            
        Returns:
            Predicted velocity
        """
        return self.flow_decoder(x, t, z_p)
    
    def generate_cells(
        self,
        x0: torch.Tensor,
        z_p: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate perturbed cells from NC cells.
        
        Args:
            x0: NC cells (n_cells, x_dim)
            z_p: Perturbation embedding (1, perturbation_dim) or (n_cells, perturbation_dim)
            n_steps: Number of ODE integration steps
            
        Returns:
            Perturbed cells (n_cells, x_dim)
        """
        return self.flow_decoder.sample(x0, z_p, n_steps)
    
    def predict_proportions(
        self,
        z_p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict cell program proportions.
        
        Args:
            z_p: Perturbation embedding
            
        Returns:
            Dictionary with state_proportions and lipo_adipo_ratio
        """
        if self.proportion_head is None:
            raise ValueError("ProportionHead not initialized")
        
        return self.proportion_head(z_p)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x: Interpolated state x_t
            t: Time
            z_p: Perturbation embedding
            
        Returns:
            Dictionary with velocity and optionally proportions
        """
        # Predict velocity
        velocity = self.predict_velocity(x, t, z_p)
        
        result = {"velocity": velocity}
        
        # Predict proportions if head exists
        if self.proportion_head is not None:
            proportions = self.predict_proportions(z_p)
            result.update(proportions)
        
        return result
    
    def inference(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        perturbation_indices: torch.Tensor,
        x0: torch.Tensor,
        n_cells: int = 100,
        n_ode_steps: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full inference pipeline.
        
        Args:
            node_features: Gene node features
            edge_index: Graph edges
            perturbation_indices: Indices of perturbed genes (batch_size,)
            x0: NC cells to transform (n_cells, x_dim)
            n_cells: Number of cells to generate per perturbation
            n_ode_steps: ODE integration steps
            
        Returns:
            Dictionary with generated_cells and proportions
        """
        # Encode perturbations
        z_p = self.encode_perturbation(
            node_features,
            edge_index,
            perturbation_indices,
        )
        
        # Generate cells for each perturbation
        all_generated = []
        all_proportions = []
        
        for i in range(len(perturbation_indices)):
            # Select subset of NC cells
            x0_subset = x0[:n_cells]
            
            # Expand z_p for broadcasting
            z_p_single = z_p[i:i+1].expand(n_cells, -1)
            
            # Generate cells
            x1 = self.generate_cells(x0_subset, z_p_single, n_ode_steps)
            all_generated.append(x1)
            
            # Predict proportions
            if self.proportion_head is not None:
                props = self.predict_proportions(z_p[i:i+1])
                all_proportions.append(props)
        
        result = {
            "generated_cells": all_generated,  # List of (n_cells, x_dim) tensors
        }
        
        if all_proportions:
            # Stack proportions
            state_props = torch.cat([p["state_proportions"] for p in all_proportions])
            result["state_proportions"] = state_props
            
            if "lipo_adipo_ratio" in all_proportions[0]:
                ratios = torch.cat([p["lipo_adipo_ratio"] for p in all_proportions])
                result["lipo_adipo_ratio"] = ratios
        
        return result


def build_model_from_config(
    config: dict,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
) -> PerturbationFlowModel:
    """
    Build the full model from configuration.
    
    Args:
        config: Configuration dictionary
        node_features: Gene node features for initialization
        edge_index: Graph edges
        
    Returns:
        Initialized PerturbationFlowModel
    """
    from .gat_encoder import GATv2Encoder, PerturbationEncoder
    from .flow_matching import ConditionalVelocityMLP, FlowMatchingDecoder
    from .proportion_head import ProportionHead
    
    # Build GATv2 encoder
    gat_encoder = GATv2Encoder(
        input_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["gat_hidden_dim"],
        output_dim=config["model"]["perturbation_dim"],
        num_layers=config["model"]["gat_layers"],
        heads=config["model"]["gat_heads"],
        dropout=config["model"]["gat_dropout"],
    )
    
    # Build perturbation encoder
    perturbation_encoder = PerturbationEncoder(
        gat_encoder=gat_encoder,
        aggregation=config["model"]["perturbation_aggregation"],
        perturbation_dim=config["model"]["perturbation_dim"],
    )
    
    # Build flow matching decoder
    velocity_mlp = ConditionalVelocityMLP(
        x_dim=config["flow_matching"]["pca_components"],
        perturbation_dim=config["model"]["perturbation_dim"],
        time_dim=config["flow_matching"]["time_embedding_dim"],
        hidden_dims=config["flow_matching"]["hidden_dims"],
        activation=config["flow_matching"]["activation"],
    )
    
    flow_decoder = FlowMatchingDecoder(
        velocity_model=velocity_mlp,
        ode_solver=config["flow_matching"]["ode_solver"],
        ode_steps=config["flow_matching"]["ode_steps"],
        rtol=config["flow_matching"]["ode_rtol"],
        atol=config["flow_matching"]["ode_atol"],
    )
    
    # Build proportion head
    proportion_head = ProportionHead(
        perturbation_dim=config["model"]["perturbation_dim"],
        hidden_dims=config["proportion_head"]["hidden_dims"],
        n_programs=config["proportion_head"]["n_programs"],
        predict_ratio=config["proportion_head"]["predict_ratio"],
    )
    
    # Combine into full model
    model = PerturbationFlowModel(
        perturbation_encoder=perturbation_encoder,
        flow_decoder=flow_decoder,
        proportion_head=proportion_head,
    )
    
    logger.info(f"Built model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model
