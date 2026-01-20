"""
Validation evaluation using official challenge scoring functions.
Generates predictions for validation perturbations and compares against ground truth.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scipy.stats
from sklearn.metrics.pairwise import rbf_kernel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.full_model import PerturbationFlowModel
from src.data.loader import load_h5ad_data, load_program_proportions, prepare_training_data
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# OFFICIAL CHALLENGE SCORING FUNCTIONS (from committee's evaluation.py)
# ============================================================================

def _pearson(ground_truth_X, prediction_X, perturbed_centroid):
    """
    Calculate Pearson correlation between predicted and actual perturbation effects.
    
    Args:
        ground_truth_X: Ground truth cell expressions (n_cells, n_genes)
        prediction_X: Predicted cell expressions (n_cells, n_genes)
        perturbed_centroid: Baseline unperturbed centroid (n_genes,)
    
    Returns:
        Pearson correlation coefficient
    """
    ground_truth_X_target = ground_truth_X.mean(axis=0)
    prediction_X_target = prediction_X.mean(axis=0)
    
    return scipy.stats.pearsonr(
        ground_truth_X_target - perturbed_centroid,
        prediction_X_target - perturbed_centroid,
    ).statistic


def _mmd(ground_truth_X, prediction_X):
    """
    Calculate Maximum Mean Discrepancy between distributions.
    Uses Gaussian kernel with official challenge parameters.
    
    Args:
        ground_truth_X: Ground truth cell expressions (n_cells, n_genes)
        prediction_X: Predicted cell expressions (n_cells, n_genes)
    
    Returns:
        MMD value (lower is better)
    """
    # Official challenge parameters
    fix_sigma = 2326
    kernel_mul = 2.0
    kernel_num = 5
    
    n_samples = int(ground_truth_X.shape[0]) + int(prediction_X.shape[0])
    total = np.concatenate([ground_truth_X, prediction_X], axis=0)
    
    total0 = total.sum(axis=0, keepdims=True)
    total1 = np.sum(total**2, axis=0, keepdims=True)
    total01 = total0**2 / n_samples
    
    L2_distance = total1.T + total1 - 2 * total.T @ total
    L2_distance = np.clip(L2_distance, 0, None)
    
    bandwidth = fix_sigma * np.sqrt(L2_distance.mean() / 2)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [rbf_kernel(total, gamma=1 / (2 * bandwidth**2)) for bandwidth in bandwidth_list]
    kernels = sum(kernel_val)
    
    XX = kernels[:ground_truth_X.shape[0], :ground_truth_X.shape[0]]
    YY = kernels[ground_truth_X.shape[0]:, ground_truth_X.shape[0]:]
    XY = kernels[:ground_truth_X.shape[0], ground_truth_X.shape[0]:]
    
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


def _l1_distance(ground_truth_y, prediction_y):
    """
    Calculate L1 distance between program proportions.
    
    Args:
        ground_truth_y: Ground truth proportions (n_programs,)
        prediction_y: Predicted proportions (n_programs,)
    
    Returns:
        L1 distance (lower is better)
    """
    return np.abs(ground_truth_y - prediction_y).sum()


# ============================================================================
# VALIDATION EVALUATION LOGIC
# ============================================================================

def load_model(checkpoint_path: str, config: dict, device: str):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load embeddings
    processed_dir = Path(config['paths']['processed_dir'])
    embeddings_path = processed_dir / "gene_embeddings.pt"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    gene_embeddings = torch.load(embeddings_path, map_location=device, weights_only=False)
    logger.info(f"Loaded gene embeddings: {gene_embeddings.shape}")
    
    # Load knowledge graph
    kg_dir = Path(config['paths']['kg_dir'])
    kg_path = kg_dir / "knowledge_graph.gpickle"
    if not kg_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found at {kg_path}")
    
    import pickle
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    logger.info(f"Loaded knowledge graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # Initialize model
    model = PerturbationFlowModel(
        gene_embeddings=gene_embeddings,
        knowledge_graph=kg,
        config=config,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    return model


def generate_predictions(model, adata, val_perturbations, device, pca_model, pca_mean):
    """
    Generate predictions for validation perturbations.
    
    Returns:
        predictions: Dict mapping perturbation -> (predicted_cells_pca, predicted_proportions)
    """
    predictions = {}
    model.eval()
    
    logger.info(f"Generating predictions for {len(val_perturbations)} perturbations...")
    
    with torch.no_grad():
        for perturb in val_perturbations:
            # Get cells for this perturbation
            mask = adata.obs['gene'] == perturb
            if mask.sum() == 0:
                logger.warning(f"No cells found for perturbation {perturb}")
                continue
            
            # Get initial state (x0) - unperturbed cells for this perturbation's cell type
            cell_mask = mask
            x0_cells = adata[cell_mask].X
            if hasattr(x0_cells, 'toarray'):
                x0_cells = x0_cells.toarray()
            
            # Transform to PCA space
            x0_pca = pca_model.transform(x0_cells - pca_mean)
            x0 = torch.tensor(x0_pca, dtype=torch.float32, device=device)
            
            # Get perturbation embedding
            if perturb in model.perturbation_embeddings:
                z_p = model.perturbation_embeddings[perturb].unsqueeze(0).repeat(x0.shape[0], 1)
            else:
                logger.warning(f"Perturbation {perturb} not in embeddings, using zero")
                z_p = torch.zeros(x0.shape[0], model.encoder.output_dim, device=device)
            
            # Generate cells using flow matching
            predicted_pca = model.generate_cells(x0, z_p)
            
            # Generate proportions
            predicted_proportions = model.proportion_head(z_p[0:1])  # Just need one
            
            predictions[perturb] = {
                'cells_pca': predicted_pca.cpu().numpy(),
                'proportions': predicted_proportions.cpu().numpy().flatten()
            }
            
            logger.info(f"  {perturb}: {predicted_pca.shape[0]} cells predicted")
    
    return predictions


def evaluate(predictions, adata, ground_truth_df, pca_model, pca_mean):
    """
    Evaluate predictions using official challenge metrics.
    
    Args:
        predictions: Dict from generate_predictions
        adata: Full dataset
        ground_truth_df: Ground truth proportions
        pca_model: PCA model for inverse transform
        pca_mean: PCA mean for inverse transform
    
    Returns:
        results: Dict with metrics for each perturbation
    """
    results = {}
    
    # Calculate unperturbed centroid (baseline)
    unperturbed_mask = adata.obs['gene'] == 'NC'
    unperturbed_cells = adata[unperturbed_mask].X
    if hasattr(unperturbed_cells, 'toarray'):
        unperturbed_cells = unperturbed_cells.toarray()
    unperturbed_centroid = unperturbed_cells.mean(axis=0)
    
    logger.info("\nEvaluating predictions...")
    logger.info("=" * 80)
    
    for perturb, pred_data in predictions.items():
        if perturb not in ground_truth_df['gene'].values:
            logger.warning(f"Skipping {perturb} - not in ground truth")
            continue
        
        # Get ground truth cells
        gt_mask = adata.obs['gene'] == perturb
        gt_cells = adata[gt_mask].X
        if hasattr(gt_cells, 'toarray'):
            gt_cells = gt_cells.toarray()
        
        # Transform predicted cells back to gene space
        pred_cells = pca_model.inverse_transform(pred_data['cells_pca']) + pca_mean
        
        # Calculate metrics
        pearson_score = _pearson(gt_cells, pred_cells, unperturbed_centroid)
        mmd_score = _mmd(gt_cells, pred_cells)
        
        # Get ground truth proportions
        gt_row = ground_truth_df[ground_truth_df['gene'] == perturb].iloc[0]
        gt_proportions = gt_row[['pre_adipo', 'adipo', 'other', 'lipo']].values.astype(float)
        pred_proportions = pred_data['proportions'][:4]  # First 4 programs
        
        l1_score = _l1_distance(gt_proportions, pred_proportions)
        
        results[perturb] = {
            'pearson': pearson_score,
            'mmd': mmd_score,
            'l1_distance': l1_score,
            'gt_proportions': gt_proportions,
            'pred_proportions': pred_proportions
        }
        
        logger.info(f"\n{perturb}:")
        logger.info(f"  Pearson: {pearson_score:.4f}")
        logger.info(f"  MMD: {mmd_score:.6f}")
        logger.info(f"  L1 Distance: {l1_score:.4f}")
        logger.info(f"  GT proportions: {gt_proportions}")
        logger.info(f"  Pred proportions: {pred_proportions}")
    
    return results


def main():
    # Paths
    config_path = Path("configs/default.yaml")
    checkpoint_path = Path("checkpoints/best_true.ckpt")
    gt_path = Path("data/raw/Challenge/program_proportion_local_gtruth.csv")
    
    # Load config
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading h5ad data...")
    raw_dir = Path(config['paths']['raw_dir'])
    h5ad_path = raw_dir / config['paths']['h5ad_file']
    gene_to_predict_path = raw_dir / config['paths']['gene_to_predict']
    
    # Use the same downsampling as in training
    adata = load_h5ad_data(
        h5ad_path,
        gene_to_predict_path,
        max_cells=config['data'].get('max_cells'),
        seed=config['training'].get('seed', 42),
        keep_perturbations=config['data'].get('val_genes', []),
    )
    logger.info(f"Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Load proportions for prepare_training_data
    logger.info("Loading program proportions...")
    proportions = load_program_proportions(
        raw_dir / config['paths']['program_proportions']
    )
    
    # Prepare training data to get validation perturbations
    logger.info("Identifying validation perturbations...")
    training_data = prepare_training_data(
        adata,
        proportions,
        val_genes=config['data'].get('val_genes', []),
        pca_components=config['data']['pca_components'],
        seed=config['training'].get('seed', 42),
        val_fraction=config['data'].get('val_fraction', 0.05),
    )
    val_perturbations = training_data['val_perturbations']
    logger.info(f"Validation perturbations: {val_perturbations}")
    
    # Load PCA model from training_data
    logger.info("Using PCA model from prepare_training_data")
    pca_model = training_data['pca_model']
    pca_mean = pca_model.mean_
    logger.info(f"PCA: {pca_model.n_components} components, {pca_mean.shape[0]} genes")
    
    # Load ground truth
    logger.info(f"Loading ground truth from {gt_path}")
    ground_truth_df = pd.read_csv(gt_path)
    logger.info(f"Ground truth: {len(ground_truth_df)} perturbations")
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    
    # Generate predictions
    predictions = generate_predictions(
        model, adata, val_perturbations, device, pca_model, pca_mean
    )
    
    # Evaluate
    results = evaluate(predictions, adata, ground_truth_df, pca_model, pca_mean)
    
    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE RESULTS (Official Challenge Metrics)")
    logger.info("=" * 80)
    
    pearson_scores = [r['pearson'] for r in results.values()]
    mmd_scores = [r['mmd'] for r in results.values()]
    l1_scores = [r['l1_distance'] for r in results.values()]
    
    logger.info(f"\nPearson Delta (mean): {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
    logger.info(f"MMD (mean): {np.mean(mmd_scores):.6f} ± {np.std(mmd_scores):.6f}")
    logger.info(f"L1 Distance (mean): {np.mean(l1_scores):.4f} ± {np.std(l1_scores):.4f}")
    
    logger.info(f"\nLeaderboard context:")
    logger.info(f"  Top MMD: 0.065 (1st place)")
    logger.info(f"  Our validation MMD: {np.mean(mmd_scores):.6f}")
    logger.info(f"  Ratio: {np.mean(mmd_scores) / 0.065:.2f}x")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'perturbation': k,
            'pearson': v['pearson'],
            'mmd': v['mmd'],
            'l1_distance': v['l1_distance'],
        }
        for k, v in results.items()
    ])
    
    output_path = Path("experiments/validation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved detailed results to {output_path}")


if __name__ == '__main__':
    main()
