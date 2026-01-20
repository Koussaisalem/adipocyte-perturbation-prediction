"""
Inference pipeline for generating perturbation predictions.

Generates predictions for unseen perturbations and formats submission files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerturbationPredictor:
    """
    Generate predictions for perturbations.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        pca_model: object,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_to_idx: dict[str, int],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained PerturbationFlowModel
            pca_model: Fitted PCA model for inverse transform
            node_features: Gene node features
            edge_index: Knowledge graph edges
            node_to_idx: Mapping from gene names to indices
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.pca_model = pca_model
        self.node_features = node_features.to(device)
        self.edge_index = edge_index.to(device)
        self.node_to_idx = node_to_idx
        self.device = device
        
        logger.info(f"Predictor initialized on {device}")
    
    @torch.no_grad()
    def predict_perturbation(
        self,
        perturbation_gene: str,
        nc_cells_pca: np.ndarray,
        n_cells: int = 100,
        n_ode_steps: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions for a single perturbation.
        
        Args:
            perturbation_gene: Name of perturbed gene
            nc_cells_pca: NC cells in PCA space (n_nc_cells, n_components)
            n_cells: Number of cells to generate
            n_ode_steps: Number of ODE integration steps
            
        Returns:
            Dictionary with:
            - 'cells_pca': Generated cells in PCA space (n_cells, n_components)
            - 'cells_full': Generated cells in full gene space (n_cells, n_genes)
            - 'proportions': Predicted proportions (4,) [pre_adipo, adipo, lipo, other]
            - 'lipo_adipo_ratio': Predicted ratio (scalar)
        """
        # Check if gene is in the graph
        if perturbation_gene not in self.node_to_idx:
            logger.warning(f"Gene {perturbation_gene} not in knowledge graph")
            # Return NC cells as fallback
            indices = np.random.choice(len(nc_cells_pca), size=n_cells, replace=True)
            cells_pca = nc_cells_pca[indices]
            cells_full = self.pca_model.inverse_transform(cells_pca)
            
            return {
                'cells_pca': cells_pca,
                'cells_full': cells_full,
                'proportions': np.array([0.25, 0.25, 0.25, 0.25]),
                'lipo_adipo_ratio': np.array([0.5]),
            }
        
        # Get perturbation index
        pert_idx = torch.tensor([self.node_to_idx[perturbation_gene]], device=self.device)
        
        # Encode perturbation
        z_p = self.model.encode_perturbation(
            self.node_features,
            self.edge_index,
            pert_idx,
        )
        
        # Sample NC cells
        indices = np.random.choice(len(nc_cells_pca), size=n_cells, replace=True)
        x0 = torch.tensor(nc_cells_pca[indices], dtype=torch.float32, device=self.device)
        
        # Expand z_p for broadcasting
        z_p = z_p.expand(n_cells, -1)
        
        # Generate perturbed cells
        x1_pca = self.model.generate_cells(x0, z_p, n_ode_steps)
        
        # Convert back to full gene space
        x1_pca_np = x1_pca.cpu().numpy()
        x1_full = self.pca_model.inverse_transform(x1_pca_np)
        
        # Predict proportions
        if self.model.proportion_head is not None:
            props_dict = self.model.predict_proportions(z_p[:1])  # Use first sample
            proportions = props_dict['state_proportions'].cpu().numpy()[0]
            lipo_adipo = props_dict.get('lipo_adipo_ratio', torch.tensor([0.5])).cpu().numpy()[0]
        else:
            proportions = np.array([0.25, 0.25, 0.25, 0.25])
            lipo_adipo = np.array([0.5])
        
        return {
            'cells_pca': x1_pca_np,
            'cells_full': x1_full,
            'proportions': proportions,
            'lipo_adipo_ratio': lipo_adipo,
        }
    
    def predict_batch(
        self,
        perturbation_genes: list[str],
        nc_cells_pca: np.ndarray,
        n_cells: int = 100,
        batch_size: int = 10,
    ) -> dict[str, list]:
        """
        Generate predictions for multiple perturbations.
        
        Args:
            perturbation_genes: List of perturbed gene names
            nc_cells_pca: NC cells in PCA space
            n_cells: Number of cells per perturbation
            batch_size: Number of perturbations to process at once
            
        Returns:
            Dictionary with predictions for all perturbations
        """
        all_cells_full = []
        all_proportions = []
        all_ratios = []
        
        for i in tqdm(range(0, len(perturbation_genes), batch_size), desc="Generating predictions"):
            batch_genes = perturbation_genes[i:i+batch_size]
            
            for gene in batch_genes:
                result = self.predict_perturbation(gene, nc_cells_pca, n_cells)
                all_cells_full.append(result['cells_full'])
                all_proportions.append(result['proportions'])
                all_ratios.append(result['lipo_adipo_ratio'])
        
        return {
            'cells': all_cells_full,
            'proportions': np.array(all_proportions),
            'ratios': np.array(all_ratios),
        }


def generate_submission(
    predictor: PerturbationPredictor,
    perturbation_list_path: str | Path,
    gene_list_path: str | Path,
    nc_cells_pca: np.ndarray,
    output_dir: str | Path,
    n_cells_per_perturbation: int = 100,
    gene_names: Optional[list[str]] = None,
) -> tuple[Path, Path]:
    """
    Generate submission files for the challenge.
    
    Args:
        predictor: Initialized PerturbationPredictor
        perturbation_list_path: Path to predict_perturbations.txt
        gene_list_path: Path to gene_to_predict.txt (output genes)
        nc_cells_pca: NC cells in PCA space
        output_dir: Directory to save submissions
        n_cells_per_perturbation: Number of cells to generate per perturbation
        gene_names: Optional list of gene names for output (in correct order)
        
    Returns:
        Tuple of (expression_matrix_path, proportions_csv_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load perturbation list
    with open(perturbation_list_path) as f:
        perturbations = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Generating predictions for {len(perturbations)} perturbations")
    
    # Load output gene list
    with open(gene_list_path) as f:
        output_genes = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Output dimensions: {len(perturbations)} x {n_cells_per_perturbation} cells x {len(output_genes)} genes")
    
    # Generate predictions
    predictions = predictor.predict_batch(
        perturbations,
        nc_cells_pca,
        n_cells=n_cells_per_perturbation,
    )
    
    # Stack all cells
    all_cells = np.vstack(predictions['cells'])  # (n_perturbations * n_cells, n_genes)
    
    logger.info(f"Generated expression matrix shape: {all_cells.shape}")
    
    # Save expression matrix
    # Expected shape: (286,300, 10,238) for 2,863 perturbations × 100 cells
    expr_path = output_dir / "submission_expression.npy"
    np.save(expr_path, all_cells.astype(np.float32))
    logger.info(f"Saved expression matrix to {expr_path}")
    
    # Create proportions dataframe
    proportions_df = pd.DataFrame({
        'gene': perturbations,
        'pre_adipo': predictions['proportions'][:, 0],
        'adipo': predictions['proportions'][:, 1],
        'lipo': predictions['proportions'][:, 2],
        'other': predictions['proportions'][:, 3],
        'lipo_adipo': predictions['ratios'].flatten(),
    })
    
    proportions_path = output_dir / "submission_proportions.csv"
    proportions_df.to_csv(proportions_path, index=False)
    logger.info(f"Saved proportions to {proportions_path}")
    
    # Validation checks
    logger.info("Submission validation:")
    logger.info(f"  - Expression matrix shape: {all_cells.shape}")
    logger.info(f"  - Expected: ({len(perturbations) * n_cells_per_perturbation}, {len(output_genes)})")
    logger.info(f"  - Proportions CSV shape: {proportions_df.shape}")
    logger.info(f"  - Expected: ({len(perturbations)}, 6)")
    
    # Check for NaNs or Infs
    if np.any(np.isnan(all_cells)) or np.any(np.isinf(all_cells)):
        logger.warning("Expression matrix contains NaN or Inf values!")
    
    if proportions_df.isnull().any().any():
        logger.warning("Proportions CSV contains NaN values!")
    
    return expr_path, proportions_path


def load_model_for_inference(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    """
    Load model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model architecture (will be populated with weights)
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    if 'metrics' in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model
