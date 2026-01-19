#!/usr/bin/env python3
"""
Evaluate model on validation set with challenge metrics.

Computes:
- Pearson Delta (correlation improvement over NC baseline)
- MMD (Maximum Mean Discrepancy)
- L1 Distance (mean absolute error)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_h5ad_data, prepare_training_data
from src.data.knowledge_graph import load_knowledge_graph, convert_to_pyg
from src.models.full_model import build_model_from_config
from src.inference.predict import load_model_for_inference
from src.losses.mmd import mmd_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_pearson_delta(pred_expr, true_expr, nc_expr):
    """
    Compute Pearson Delta for a single perturbation.
    
    Delta = corr(pred, true) - corr(NC, true)
    """
    # Flatten to 1D arrays
    pred_flat = pred_expr.flatten()
    true_flat = true_expr.flatten()
    nc_flat = nc_expr.flatten()
    
    # Pearson correlations
    r_pred, _ = pearsonr(pred_flat, true_flat)
    r_baseline, _ = pearsonr(nc_flat, true_flat)
    
    return r_pred - r_baseline


def evaluate_model(
    checkpoint_path: str,
    config_path: str = "configs/default.yaml",
    n_cells: int = 100,
):
    """Evaluate model on validation set."""
    
    import yaml
    
    # Load config
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Construct full paths
    from pathlib import Path
    paths = config['paths']
    raw_dir = Path(paths['raw_dir'])
    processed_dir = Path(paths['processed_dir'])
    kg_dir = Path(paths['kg_dir'])
    
    # Load data
    logger.info("Loading h5ad data...")
    adata = load_h5ad_data(
        raw_dir / paths['h5ad_file'],
        processed_dir / "all_genes.txt",
        max_cells=config['data'].get('max_cells'),
        seed=config['training']['seed'],
        keep_perturbations=config['data'].get('val_genes', []),
    )
    
    # Load signature genes and compute scores
    logger.info("Loading signature genes...")
    from src.data.loader import load_signature_genes, compute_program_scores
    signature_genes = load_signature_genes(raw_dir / paths['signature_genes'])
    
    logger.info("Computing program scores...")
    program_scores = compute_program_scores(
        adata,
        signature_genes,
        method=config['data']['program_score_method'],
    )
    
    # Load proportions
    logger.info("Loading program proportions...")
    from src.data.loader import load_program_proportions
    proportions = load_program_proportions(raw_dir / paths['program_proportions'])
    
    # Prepare training data (to get val split)
    logger.info("Preparing training data...")
    training_data = prepare_training_data(
        adata,
        proportions,
        val_genes=config['data'].get('val_genes', []),
        pca_components=config['data']['pca_components'],
        seed=config['training']['seed'],
        val_fraction=config['data'].get('val_fraction', 0.05),
    )
    
    val_perturbations = training_data['val_perturbations']
    pca_model = training_data['pca_model']
    logger.info(f"Validation perturbations: {len(val_perturbations)}")
    
    # Load knowledge graph and embeddings
    logger.info("Loading knowledge graph...")
    G = load_knowledge_graph(kg_dir / "knowledge_graph.gpickle")
    
    logger.info("Loading gene embeddings...")
    gene_embeddings = torch.load(processed_dir / "gene_embeddings.pt")
    
    logger.info("Converting graph to PyG format...")
    pyg_data = convert_to_pyg(G, gene_embeddings)
    
    # Load model
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model, checkpoint = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        config=config,
        pyg_data=pyg_data,
        gene_embeddings=gene_embeddings,
        device='cpu',
    )
    
    # Get NC baseline  
    nc_mask = adata.obs['gene'] == 'NC'
    nc_indices = np.where(nc_mask)[0]
    nc_data = adata[nc_indices].X
    if hasattr(nc_data, 'toarray'):
        nc_data = nc_data.toarray()
    nc_mean = nc_data.mean(axis=0)
    
    # Build perturbation_to_cells dict
    perturbation_to_cells = {}
    for pert in adata.obs['gene'].unique():
        pert_mask = adata.obs['gene'] == pert
        perturbation_to_cells[pert] = np.where(pert_mask)[0].tolist()
    
    # Evaluate on validation perturbations
    results = []
    
    logger.info(f"Evaluating {len(val_perturbations)} validation perturbations...")
    
    for pert_name in val_perturbations:
        logger.info(f"  Processing: {pert_name}")
        
        # Get true perturbed cells
        pert_indices = perturbation_to_cells.get(pert_name, [])
        if len(pert_indices) == 0:
            logger.warning(f"  No cells found for {pert_name}, skipping")
            continue
            
        true_data = adata[pert_indices].X
        if hasattr(true_data, 'toarray'):
            true_data = true_data.toarray()
        true_mean = true_data.mean(axis=0)
        
        # Generate predictions
        # Sample NC cells
        nc_sample_indices = np.random.choice(len(nc_indices), min(n_cells, len(nc_indices)), replace=False)
        nc_sample = nc_data[nc_sample_indices]
        
        # Transform to PCA space
        nc_pca = pca_model.transform(nc_sample)
        
        # Get perturbation embedding
        if pert_name not in gene_embeddings:
            logger.warning(f"  No embedding for {pert_name}, using zeros")
            z_p = torch.zeros(512)
        else:
            z_p = gene_embeddings[pert_name]
        
        # Generate cells
        x0 = torch.FloatTensor(nc_pca)
        z_p_batch = z_p.unsqueeze(0).expand(len(x0), -1)
        
        with torch.no_grad():
            pred_pca = model.generate_cells(x0, z_p_batch).numpy()
        
        # Inverse transform to gene space
        pred_expr = pca_model.inverse_transform(pred_pca)
        pred_mean = pred_expr.mean(axis=0)
        
        # Compute metrics
        pearson_delta = compute_pearson_delta(pred_mean, true_mean, nc_mean)
        l1_dist = mean_absolute_error(true_mean, pred_mean)
        
        # Compute MMD between predicted and true distributions
        pred_tensor = torch.FloatTensor(pred_pca)
        true_pca = pca_model.transform(true_data)
        true_tensor = torch.FloatTensor(true_pca)
        mmd = mmd_loss(pred_tensor, true_tensor).item()
        
        results.append({
            'perturbation': pert_name,
            'pearson_delta': pearson_delta,
            'l1_distance': l1_dist,
            'mmd': mmd,
            'n_true_cells': len(pert_indices),
            'n_pred_cells': len(pred_expr),
        })
        
        logger.info(f"    Pearson Delta: {pearson_delta:.4f}")
        logger.info(f"    L1 Distance: {l1_dist:.4f}")
        logger.info(f"    MMD: {mmd:.4f}")
    
    # Aggregate results
    results_df = pd.DataFrame(results)
    
    avg_pearson_delta = results_df['pearson_delta'].mean()
    avg_l1 = results_df['l1_distance'].mean()
    avg_mmd = results_df['mmd'].mean()
    
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    print(f"Number of perturbations: {len(results_df)}")
    print(f"\nChallenge Metrics:")
    print(f"  Average Pearson Delta: {avg_pearson_delta:.4f}")
    print(f"  Average L1 Distance:   {avg_l1:.4f}")
    print(f"  Average MMD:           {avg_mmd:.4f}")
    print("="*60)
    
    print("\nPer-perturbation breakdown:")
    print(results_df.to_string(index=False))
    
    # Save results
    output_path = Path("experiments/validation_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved detailed results to: {output_path}")
    
    return results_df, avg_pearson_delta, avg_l1, avg_mmd


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=100,
        help="Number of cells to generate per perturbation",
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        n_cells=args.n_cells,
    )


if __name__ == "__main__":
    main()
