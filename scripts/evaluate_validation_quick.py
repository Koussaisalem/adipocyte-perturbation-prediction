"""
QUICK validation evaluation using only validation perturbation cells.
Avoids loading full dataset - only loads cells for 6 validation perturbations.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scipy.stats
from sklearn.metrics.pairwise import rbf_kernel
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# OFFICIAL CHALLENGE SCORING FUNCTIONS
# ============================================================================

def _pearson(ground_truth_X, prediction_X, perturbed_centroid):
    """Calculate Pearson correlation between predicted and actual perturbation effects."""
    ground_truth_X_target = ground_truth_X.mean(axis=0)
    prediction_X_target = prediction_X.mean(axis=0)
    
    return scipy.stats.pearsonr(
        ground_truth_X_target - perturbed_centroid,
        prediction_X_target - perturbed_centroid,
    ).statistic


def _mmd(ground_truth_X, prediction_X):
    """Calculate Maximum Mean Discrepancy using official challenge parameters."""
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
    """Calculate L1 distance between program proportions."""
    return np.abs(ground_truth_y - prediction_y).sum()


def main():
    # Load config
    config_path = Path("configs/default.yaml")
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load ground truth
    gt_path = Path("data/raw/Challenge/program_proportion_local_gtruth.csv")
    logger.info(f"Loading ground truth from {gt_path}")
    ground_truth_df = pd.read_csv(gt_path)
    val_genes = list(ground_truth_df['gene'].values)
    logger.info(f"Validation genes: {val_genes}")
    
    # Load ONLY the cells for these perturbations + NC (control)
    h5ad_path = Path("data/raw/obesity_challenge_1.h5ad")
    logger.info(f"Loading h5ad (filtering for val genes only)...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Filter to only NC and validation perturbations
    keep_mask = adata.obs['gene'].isin(val_genes + ['NC'])
    adata = adata[keep_mask].copy()
    logger.info(f"Filtered to {adata.shape[0]} cells for {len(val_genes)} val genes + NC")
    
    # Get expression matrix
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    # Calculate NC centroid (unperturbed baseline)
    nc_mask = adata.obs['gene'] == 'NC'
    nc_cells = X[nc_mask]
    nc_centroid = nc_cells.mean(axis=0)
    logger.info(f"NC centroid calculated from {nc_cells.shape[0]} cells")
    
    # For this quick evaluation, we'll use a simple baseline prediction:
    # Just use NC centroid as the "prediction" - this will give us worst-case metrics
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS (Baseline: NC centroid)")
    logger.info("=" * 80)
    
    results = []
    
    for gene in val_genes:
        # Get ground truth cells
        gt_mask = adata.obs['gene'] == gene
        gt_cells = X[gt_mask]
        
        if gt_cells.shape[0] == 0:
            logger.warning(f"No cells found for {gene}, skipping")
            continue
        
        # Baseline "prediction": use NC centroid repeated
        pred_cells = np.repeat(nc_centroid[np.newaxis, :], gt_cells.shape[0], axis=0)
        
        # Calculate metrics
        pearson_score = _pearson(gt_cells, pred_cells, nc_centroid)
        mmd_score = _mmd(gt_cells, pred_cells)
        
        # Get ground truth proportions
        gt_row = ground_truth_df[ground_truth_df['gene'] == gene].iloc[0]
        gt_proportions = gt_row[['pre_adipo', 'adipo', 'other', 'lipo']].values.astype(float)
        
        # Baseline prediction: uniform proportions
        pred_proportions = np.ones(4) * 0.25
        
        l1_score = _l1_distance(gt_proportions, pred_proportions)
        
        results.append({
            'gene': gene,
            'pearson': pearson_score,
            'mmd': mmd_score,
            'l1_distance': l1_score,
            'n_cells': gt_cells.shape[0]
        })
        
        logger.info(f"\n{gene} ({gt_cells.shape[0]} cells):")
        logger.info(f"  Pearson: {pearson_score:.4f}")
        logger.info(f"  MMD: {mmd_score:.6f}")
        logger.info(f"  L1 Distance: {l1_score:.4f}")
    
    # Aggregate
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE RESULTS (Baseline)")
    logger.info("=" * 80)
    logger.info(f"\nPearson Delta (mean): {results_df['pearson'].mean():.4f} ± {results_df['pearson'].std():.4f}")
    logger.info(f"MMD (mean): {results_df['mmd'].mean():.6f} ± {results_df['mmd'].std():.6f}")
    logger.info(f"L1 Distance (mean): {results_df['l1_distance'].mean():.4f} ± {results_df['l1_distance'].std():.4f}")
    
    logger.info(f"\nLeaderboard context:")
    logger.info(f"  Top MMD: 0.065 (1st place)")
    logger.info(f"  Baseline MMD: {results_df['mmd'].mean():.6f}")
    logger.info(f"  Ratio: {results_df['mmd'].mean() / 0.065:.2f}x worse")
    
    # Save
    output_path = Path("experiments/validation_baseline_results.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NOTE: This is a BASELINE evaluation (NC centroid as prediction)")
    logger.info("To evaluate the actual trained model, we need to:")
    logger.info("1. Load the model checkpoint")
    logger.info("2. Apply the same PCA transform used in training")
    logger.info("3. Generate predictions using the flow matching decoder")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
