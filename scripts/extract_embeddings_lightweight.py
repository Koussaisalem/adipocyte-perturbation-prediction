#!/usr/bin/env python3
"""
Extract gene embeddings using lightweight PCA + Statistics approach.
No GPU required - runs on CPU in minutes instead of hours.
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Use anndata directly instead of scanpy to avoid dependency hell
try:
    import anndata
except ImportError:
    print("Installing anndata...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anndata", "-q"])
    import anndata

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.paths import PROCESSED_DIR, RAW_DATA_DIR
except ImportError:
    PROCESSED_DIR = Path("data/processed")
    RAW_DATA_DIR = Path("data/raw/Challenge")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_gene_statistics(X, gene_names):
    """
    Compute statistical features for each gene across cells.
    
    Returns:
        np.ndarray: (n_genes, n_features) array of statistics
    """
    logger.info(f"Computing gene statistics for {X.shape[1]} genes across {X.shape[0]} cells")
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Compute statistics per gene (column-wise)
    features = []
    
    # 1. Mean expression
    means = np.mean(X, axis=0)
    features.append(means)
    
    # 2. Standard deviation
    stds = np.std(X, axis=0)
    features.append(stds)
    
    # 3. Expression frequency (% of cells with non-zero expression)
    expr_freq = np.mean(X > 0, axis=0)
    features.append(expr_freq)
    
    # 4. Maximum expression
    maxs = np.max(X, axis=0)
    features.append(maxs)
    
    # 5. 90th percentile
    p90 = np.percentile(X, 90, axis=0)
    features.append(p90)
    
    # 6. Coefficient of variation (std/mean, handle division by zero)
    cv = np.divide(stds, means, out=np.zeros_like(stds), where=means!=0)
    features.append(cv)
    
    # 7. Log mean (log(1 + mean))
    log_means = np.log1p(means)
    features.append(log_means)
    
    # Stack features: (n_genes, n_statistical_features)
    gene_features = np.stack(features, axis=1)
    
    logger.info(f"Computed {gene_features.shape[1]} statistical features per gene")
    return gene_features


def extract_pca_embeddings(X, n_components=505):
    """
    Extract gene embeddings using PCA on transposed expression matrix.
    Genes as samples, cells as features -> gene embeddings in cell-space.
    
    Returns:
        np.ndarray: (n_genes, n_components) PCA embeddings
    """
    logger.info(f"Extracting PCA-based gene embeddings...")
    
    # Convert sparse to dense
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Transpose: genes as rows, cells as columns
    X_t = X.T  # (n_genes, n_cells)
    logger.info(f"Transposed matrix shape: {X_t.shape}")
    
    # Limit components to feasible range
    max_components = min(n_components, X_t.shape[0], X_t.shape[1])
    if max_components < n_components:
        logger.warning(f"Requested {n_components} components, but only {max_components} possible. Using {max_components}.")
        n_components = max_components
    
    # Standardize features (cells) for each gene
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t)
    
    # Apply PCA
    logger.info(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    gene_embeddings = pca.fit_transform(X_scaled)  # (n_genes, n_components)
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA complete. Explained variance: {explained_var:.2%}")
    logger.info(f"Top 10 components explain: {pca.explained_variance_ratio_[:10].sum():.2%}")
    
    return gene_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract lightweight gene embeddings using Statistics + PCA (CPU-friendly)"
    )
    parser.add_argument("--h5ad-file", required=True, help="Path to h5ad file")
    parser.add_argument(
        "--output", 
        default=str(PROCESSED_DIR / "gene_embeddings.pt"), 
        help="Output path for embeddings"
    )
    parser.add_argument(
        "--embedding-dim", 
        type=int, 
        default=512, 
        help="Total embedding dimension (default: 512, same as Geneformer)"
    )
    parser.add_argument(
        "--max-cells", 
        type=int, 
        default=20000,
        help="Max cells to use for embedding computation (default: 20000, use -1 for all)"
    )
    parser.add_argument(
        "--n-statistical-features",
        type=int,
        default=7,
        help="Number of statistical features to compute (default: 7)"
    )
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading {args.h5ad_file}...")
    adata = anndata.read_h5ad(args.h5ad_file)
    logger.info(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Sample cells if requested
    if args.max_cells > 0 and adata.n_obs > args.max_cells:
        logger.info(f"Sampling {args.max_cells} cells for efficiency...")
        np.random.seed(42)
        indices = np.random.choice(adata.n_obs, args.max_cells, replace=False)
        adata_subset = adata[indices, :]
        X = adata_subset.X
    else:
        logger.info("Using all cells")
        X = adata.X
    
    # Compute statistical features
    gene_stats = compute_gene_statistics(X, adata.var_names)
    
    # Compute PCA embeddings for remaining dimensions
    pca_dim = args.embedding_dim - gene_stats.shape[1]
    
    if pca_dim <= 0:
        logger.warning(f"Statistical features ({gene_stats.shape[1]}) >= embedding_dim ({args.embedding_dim})")
        logger.warning(f"Truncating to {args.embedding_dim} dimensions")
        gene_embeddings = gene_stats[:, :args.embedding_dim]
    else:
        logger.info(f"Computing {pca_dim} PCA dimensions to complement {gene_stats.shape[1]} statistical features...")
        pca_embeddings = extract_pca_embeddings(X, n_components=pca_dim)
        
        # Concatenate: [statistics | PCA]
        gene_embeddings = np.concatenate([gene_stats, pca_embeddings], axis=1)
        logger.info(f"Combined embeddings shape: {gene_embeddings.shape}")
    
    # Normalize embeddings (optional but recommended)
    logger.info("Normalizing embeddings...")
    scaler = StandardScaler()
    gene_embeddings = scaler.fit_transform(gene_embeddings)
    
    # Convert to dict format: {gene_name: tensor}
    logger.info("Converting to dictionary format...")
    embeddings_dict = {}
    for i, gene_name in enumerate(adata.var_names):
        embeddings_dict[gene_name] = torch.tensor(gene_embeddings[i], dtype=torch.float32)
    
    # Save
    logger.info(f"Saving {len(embeddings_dict)} gene embeddings to {output_path}")
    torch.save(embeddings_dict, output_path)
    
    # Summary statistics
    sample_embedding = gene_embeddings[0]
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Embedding extraction complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Total genes:           {len(embeddings_dict)}")
    logger.info(f"Embedding dimension:   {args.embedding_dim}")
    logger.info(f"  Statistical features: {args.n_statistical_features}")
    logger.info(f"  PCA components:       {pca_dim if pca_dim > 0 else 0}")
    logger.info(f"Output file:           {output_path}")
    logger.info(f"File size:             {output_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
