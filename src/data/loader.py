"""
Data loader for the Adipocyte Perturbation Challenge.

Handles loading the h5ad file, computing cell program scores using signature genes,
and preparing training/validation splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_h5ad_data(
    h5ad_path: str | Path,
    gene_list_path: Optional[str | Path] = None,
    max_cells: Optional[int] = None,
    seed: int = 42,
    keep_perturbations: Optional[list[str]] = None,
) -> ad.AnnData:
    """
    Load the challenge h5ad file.
    
    Args:
        h5ad_path: Path to obesity_challenge_1.h5ad
        gene_list_path: Optional path to gene_to_predict.txt to subset genes
        
    Returns:
        AnnData object with expression data and perturbation labels
    """
    h5ad_path = Path(h5ad_path)
    logger.info(f"Loading h5ad from {h5ad_path}")
    
    adata = sc.read_h5ad(h5ad_path)
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Check expected structure
    if "gene" not in adata.obs.columns:
        raise ValueError("Expected 'gene' column in .obs for perturbation labels")
    
    # Log perturbation info
    perturbations = adata.obs["gene"].unique()
    n_perturbed = (adata.obs["gene"] != "NC").sum()
    n_nc = (adata.obs["gene"] == "NC").sum()
    logger.info(f"Found {len(perturbations)} unique perturbations ({n_perturbed} cells)")
    logger.info(f"Found {n_nc} negative control (NC) cells")
    
    # Optionally downsample cells to reduce memory
    if max_cells is not None and adata.n_obs > max_cells:
        logger.info(f"Downsampling cells: {adata.n_obs} -> {max_cells} (seed={seed})")
        rng = np.random.default_rng(seed)

        if "gene" in adata.obs.columns:
            labels = adata.obs["gene"].values
            keep_labels = set(keep_perturbations or [])
            keep_mask = np.isin(labels, list(keep_labels))

            # Always keep all requested perturbations; if they exceed budget, sample within
            keep_indices = np.where(keep_mask)[0]
            if len(keep_indices) > max_cells:
                keep_indices = rng.choice(keep_indices, max_cells, replace=False)
                downsample_indices = keep_indices
            else:
                remaining = max_cells - len(keep_indices)
                other_indices = np.where(~keep_mask)[0]
                sampled_others = rng.choice(other_indices, remaining, replace=False)
                downsample_indices = np.concatenate([keep_indices, sampled_others])
            rng.shuffle(downsample_indices)
            adata = adata[downsample_indices, :].copy()
        else:
            keep_idx = rng.choice(adata.n_obs, max_cells, replace=False)
            adata = adata[keep_idx, :].copy()

    # Optionally subset to required genes
    if gene_list_path is not None:
        gene_list_path = Path(gene_list_path)
        with open(gene_list_path) as f:
            genes_to_keep = [line.strip() for line in f if line.strip()]
        
        # Filter to genes present in the data
        genes_in_data = set(adata.var_names)
        genes_to_keep = [g for g in genes_to_keep if g in genes_in_data]
        
        logger.info(f"Subsetting to {len(genes_to_keep)} genes from {gene_list_path.name}")
        adata = adata[:, genes_to_keep].copy()
    
    return adata


def load_signature_genes(signature_path: str | Path) -> dict[str, list[str]]:
    """
    Load signature genes for cell programs.
    
    Args:
        signature_path: Path to signature_genes.csv
        
    Returns:
        Dictionary mapping program names to list of signature genes
    """
    signature_path = Path(signature_path)
    df = pd.read_csv(signature_path)
    
    # Expected format: columns include gene names and program assignments
    # Based on challenge data: gene,program format
    programs = {}
    
    if "program" in df.columns and "gene" in df.columns:
        for program in df["program"].unique():
            programs[program] = df[df["program"] == program]["gene"].tolist()
    else:
        # Alternative format: each column is a program with gene names
        for col in df.columns:
            if col.lower() not in ["unnamed: 0", "index"]:
                genes = df[col].dropna().tolist()
                if genes:
                    programs[col] = genes
    
    logger.info(f"Loaded signature genes for {len(programs)} programs:")
    for prog, genes in programs.items():
        logger.info(f"  {prog}: {len(genes)} genes")
    
    return programs


def compute_program_scores(
    adata: ad.AnnData,
    signature_genes: dict[str, list[str]],
    method: str = "mean_zscore",
) -> pd.DataFrame:
    """
    Compute cell program scores based on signature gene expression.
    
    Args:
        adata: AnnData with expression in .X
        signature_genes: Dict mapping program names to signature genes
        method: Scoring method - 'mean_zscore', 'mean_raw', or 'ssgsea'
        
    Returns:
        DataFrame with program scores per cell (n_cells x n_programs)
    """
    logger.info(f"Computing program scores using method: {method}")
    
    # Get expression matrix
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    scores = {}
    
    for program, genes in signature_genes.items():
        # Filter to genes present in data
        valid_genes = [g for g in genes if g in gene_to_idx]
        
        if len(valid_genes) == 0:
            logger.warning(f"No signature genes found for program {program}")
            scores[program] = np.zeros(adata.n_obs)
            continue
        
        # Get expression of signature genes
        gene_indices = [gene_to_idx[g] for g in valid_genes]
        expr = X[:, gene_indices]  # (n_cells, n_sig_genes)
        
        if method == "mean_zscore":
            # Z-score normalize per gene, then take mean
            scaler = StandardScaler()
            expr_z = scaler.fit_transform(expr)
            scores[program] = np.mean(expr_z, axis=1)
            
        elif method == "mean_raw":
            # Simple mean of expression
            scores[program] = np.mean(expr, axis=1)
            
        elif method == "ssgsea":
            # Simplified ssGSEA-like ranking approach
            # Rank genes per cell, score by rank of signature genes
            ranks = np.argsort(np.argsort(-X, axis=1), axis=1)  # Descending ranks
            sig_ranks = ranks[:, gene_indices]
            # Score inversely proportional to mean rank
            scores[program] = -np.mean(sig_ranks, axis=1)
            
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        logger.info(f"  {program}: {len(valid_genes)}/{len(genes)} genes used")
    
    return pd.DataFrame(scores, index=adata.obs_names)


def assign_cell_programs(
    program_scores: pd.DataFrame,
    include_other: bool = True,
) -> pd.Series:
    """
    Assign discrete program labels to cells based on highest score.
    
    Args:
        program_scores: DataFrame with program scores per cell
        include_other: Whether to include 'other' as a category
        
    Returns:
        Series with program assignment per cell
    """
    # Argmax across programs
    assignments = program_scores.idxmax(axis=1)
    
    # Optionally, cells with low max scores could be assigned to 'other'
    if include_other and "other" not in program_scores.columns:
        # Use threshold based on score distribution
        max_scores = program_scores.max(axis=1)
        threshold = np.percentile(max_scores, 25)  # Bottom 25% -> other
        assignments = assignments.where(max_scores > threshold, "other")
    
    logger.info("Program assignment distribution:")
    for prog, count in assignments.value_counts().items():
        logger.info(f"  {prog}: {count} cells ({100*count/len(assignments):.1f}%)")
    
    return assignments


def load_program_proportions(
    proportions_path: str | Path,
) -> pd.DataFrame:
    """
    Load ground truth program proportions for training.
    
    Args:
        proportions_path: Path to program_proportion.csv
        
    Returns:
        DataFrame with proportions per perturbation
    """
    df = pd.read_csv(proportions_path)
    
    # Set gene as index
    if "gene" in df.columns:
        df = df.set_index("gene")
    
    logger.info(f"Loaded proportions for {len(df)} perturbations")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def prepare_training_data(
    adata: ad.AnnData,
    proportions: pd.DataFrame,
    val_genes: list[str],
    pca_components: int = 500,
) -> dict:
    """
    Prepare training data with PCA transformation.
    
    Args:
        adata: AnnData with expression data
        proportions: DataFrame with program proportions
        val_genes: List of genes to use for validation
        pca_components: Number of PCA components for flow matching
        
    Returns:
        Dictionary with training data, validation data, and PCA model
    """
    logger.info("Preparing training data...")
    
    # Get expression matrix
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    # Fit PCA on NC cells only
    nc_mask = adata.obs["gene"] == "NC"
    X_nc = X[nc_mask]
    
    logger.info(f"Fitting PCA ({pca_components} components) on {X_nc.shape[0]} NC cells")
    pca = PCA(n_components=pca_components)
    pca.fit(X_nc)
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explains {100*explained_var:.1f}% of variance")
    
    # Transform all cells
    X_pca = pca.transform(X)
    
    # Split by training vs validation perturbations
    train_genes = [g for g in proportions.index if g not in val_genes and g != "NC"]
    
    # Get perturbation data
    training_data = {
        "X_pca": X_pca,
        "gene_labels": adata.obs["gene"].values,
        "gene_names": list(adata.var_names),
        "cell_indices": np.arange(adata.n_obs),
        "train_perturbations": train_genes,
        "val_perturbations": val_genes,
        "proportions": proportions,
        "nc_mask": nc_mask.values,
        "pca_model": pca,
    }
    
    logger.info(f"Training perturbations: {len(train_genes)}")
    logger.info(f"Validation perturbations: {len(val_genes)}")
    
    return training_data


def get_perturbation_cells(
    training_data: dict,
    perturbation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get cells for a specific perturbation.
    
    Args:
        training_data: Output from prepare_training_data
        perturbation: Gene name
        
    Returns:
        Tuple of (perturbed cell expression, NC cell expression) in PCA space
    """
    gene_labels = training_data["gene_labels"]
    X_pca = training_data["X_pca"]
    
    perturbed_mask = gene_labels == perturbation
    nc_mask = training_data["nc_mask"]
    
    X_perturbed = X_pca[perturbed_mask]
    X_nc = X_pca[nc_mask]
    
    return X_perturbed, X_nc
