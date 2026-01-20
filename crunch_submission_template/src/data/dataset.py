"""
PyTorch Dataset for Perturbation Training.

Handles sampling of NC (negative control) and perturbed cells
for conditional flow matching training.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PerturbationDataset(Dataset):
    """
    Dataset for training flow matching on perturbation data.
    
    Each sample consists of:
    - A perturbation identifier (gene name)
    - An NC cell (source distribution)
    - A perturbed cell (target distribution)
    - The perturbation's program proportions (for multi-task learning)
    """
    
    def __init__(
        self,
        X_pca: np.ndarray,
        gene_labels: np.ndarray,
        perturbations: list[str],
        proportions: Optional["pd.DataFrame"] = None,
        n_samples_per_perturbation: int = 1000,
        seed: int = 42,
    ):
        """
        Initialize the dataset.
        
        Args:
            X_pca: PCA-transformed expression matrix (n_cells, n_components)
            gene_labels: Perturbation label for each cell
            perturbations: List of perturbation gene names to include
            proportions: DataFrame with program proportions per perturbation
            n_samples_per_perturbation: Number of samples to generate per perturbation
            seed: Random seed for reproducibility
        """
        self.X_pca = X_pca
        self.gene_labels = gene_labels
        self.perturbations = perturbations
        self.proportions = proportions
        self.n_samples_per_perturbation = n_samples_per_perturbation
        self.rng = np.random.RandomState(seed)
        
        # Pre-compute indices for each perturbation
        self.nc_indices = np.where(gene_labels == "NC")[0]
        self.perturbation_indices = {}
        
        for pert in perturbations:
            indices = np.where(gene_labels == pert)[0]
            if len(indices) > 0:
                self.perturbation_indices[pert] = indices
            else:
                logger.warning(f"No cells found for perturbation: {pert}")
        
        # Filter to perturbations with cells
        self.perturbations = [p for p in perturbations if p in self.perturbation_indices]
        
        logger.info(f"Dataset initialized with {len(self.perturbations)} perturbations")
        logger.info(f"NC cells: {len(self.nc_indices)}")
        
        # Pre-generate sample pairs
        self._generate_samples()
    
    def _generate_samples(self):
        """Pre-generate all training samples."""
        self.samples = []
        
        for pert in self.perturbations:
            pert_indices = self.perturbation_indices[pert]
            
            for _ in range(self.n_samples_per_perturbation):
                # Sample one NC cell and one perturbed cell
                nc_idx = self.rng.choice(self.nc_indices)
                pert_idx = self.rng.choice(pert_indices)
                
                self.samples.append({
                    "perturbation": pert,
                    "nc_idx": nc_idx,
                    "pert_idx": pert_idx,
                })
        
        logger.info(f"Generated {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Get expression vectors
        x0 = self.X_pca[sample["nc_idx"]]  # NC cell (source)
        x1 = self.X_pca[sample["pert_idx"]]  # Perturbed cell (target)
        
        # Sample time
        t = self.rng.uniform(0, 1)
        
        # Interpolate
        xt = (1 - t) * x0 + t * x1
        
        # Target velocity
        velocity = x1 - x0
        
        result = {
            "perturbation": sample["perturbation"],
            "x0": torch.tensor(x0, dtype=torch.float32),
            "x1": torch.tensor(x1, dtype=torch.float32),
            "xt": torch.tensor(xt, dtype=torch.float32),
            "t": torch.tensor(t, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
        }
        
        # Add proportions if available
        if self.proportions is not None and sample["perturbation"] in self.proportions.index:
            props = self.proportions.loc[sample["perturbation"]].values
            result["proportions"] = torch.tensor(props, dtype=torch.float32)
        
        return result


class InferenceDataset(Dataset):
    """
    Dataset for inference - generates NC cells to transform.
    """
    
    def __init__(
        self,
        X_pca_nc: np.ndarray,
        perturbations: list[str],
        n_cells_per_perturbation: int = 100,
        seed: int = 42,
    ):
        """
        Initialize inference dataset.
        
        Args:
            X_pca_nc: PCA-transformed NC cells
            perturbations: List of perturbations to generate predictions for
            n_cells_per_perturbation: Number of cells to generate per perturbation
            seed: Random seed
        """
        self.X_pca_nc = X_pca_nc
        self.perturbations = perturbations
        self.n_cells = n_cells_per_perturbation
        self.rng = np.random.RandomState(seed)
        
        # Pre-sample NC cells for each perturbation
        self.samples = []
        for pert in perturbations:
            nc_indices = self.rng.choice(len(X_pca_nc), size=n_cells_per_perturbation)
            for idx in nc_indices:
                self.samples.append({
                    "perturbation": pert,
                    "nc_idx": idx,
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        x0 = self.X_pca_nc[sample["nc_idx"]]
        
        return {
            "perturbation": sample["perturbation"],
            "x0": torch.tensor(x0, dtype=torch.float32),
        }


def collate_perturbations(batch: list[dict]) -> dict:
    """
    Custom collate function for perturbation batches.
    
    Groups samples by perturbation for efficient GNN encoding.
    """
    # Stack tensors
    result = {
        "perturbation": [b["perturbation"] for b in batch],
        "x0": torch.stack([b["x0"] for b in batch]),
        "xt": torch.stack([b["xt"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "velocity": torch.stack([b["velocity"] for b in batch]),
    }
    
    if "x1" in batch[0]:
        result["x1"] = torch.stack([b["x1"] for b in batch])
    
    if "proportions" in batch[0]:
        result["proportions"] = torch.stack([b["proportions"] for b in batch])
    
    return result


def create_dataloaders(
    training_data: dict,
    batch_size: int = 64,
    n_samples_per_perturbation: int = 1000,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        training_data: Output from prepare_training_data
        batch_size: Batch size
        n_samples_per_perturbation: Samples per perturbation per epoch
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = PerturbationDataset(
        X_pca=training_data["X_pca"],
        gene_labels=training_data["gene_labels"],
        perturbations=training_data["train_perturbations"],
        proportions=training_data["proportions"],
        n_samples_per_perturbation=n_samples_per_perturbation,
        seed=seed,
    )
    
    # Validation dataset
    val_dataset = PerturbationDataset(
        X_pca=training_data["X_pca"],
        gene_labels=training_data["gene_labels"],
        perturbations=training_data["val_perturbations"],
        proportions=training_data["proportions"],
        n_samples_per_perturbation=n_samples_per_perturbation // 10,  # Fewer samples for val
        seed=seed + 1,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_perturbations,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_perturbations,
        pin_memory=True,
    )
    
    return train_loader, val_loader
