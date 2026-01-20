"""
Zero-shot validation evaluation for held-out genes.
Generates predictions for 5 validation genes that have NO cells in training data.
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
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.full_model import PerturbationFlowModel, build_model_from_config
from src.data.knowledge_graph import load_knowledge_graph, convert_to_pyg


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


def load_model(checkpoint_path, config, device):
    """Load trained model and required data structures."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load embeddings
    processed_dir = Path(config['paths']['processed_dir'])
    embeddings_path = processed_dir / "gene_embeddings.pt"
    gene_embeddings = torch.load(embeddings_path, map_location=device, weights_only=False)
    
    # Handle both dict and tensor formats
    if isinstance(gene_embeddings, dict):
        logger.info(f"Loaded gene embeddings: {len(gene_embeddings)} genes")
    else:
        logger.info(f"Loaded gene embeddings: {gene_embeddings.shape}")
    
    # Load knowledge graph
    kg_dir = Path(config['paths']['kg_dir'])
    kg_path = kg_dir / "knowledge_graph.gpickle"
    kg = load_knowledge_graph(kg_path)
    logger.info(f"Loaded knowledge graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # Convert to PyG format
    logger.info("Converting graph to PyG format...")
    pyg_data = convert_to_pyg(
        kg,
        node_features=gene_embeddings,
        feature_dim=config["model"]["embedding_dim"],
    )
    
    # Build model using config
    logger.info("Building model...")
    model = build_model_from_config(
        config,
        node_features=pyg_data["gene"].x,
        edge_index=pyg_data["gene", "interacts_with", "gene"].edge_index,
    )
    model.to(device)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Store PyG data for later use
    model.pyg_data = pyg_data
    model.kg = kg
    
    # Create gene_to_idx mapping
    model.gene_to_idx = {gene: idx for idx, gene in enumerate(kg.nodes())}
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    
    return model, kg


def generate_zero_shot_predictions(model, genes, nc_cells_pca, device, n_cells=100):
    """
    Generate predictions for genes NOT seen during training (zero-shot).
    
    Args:
        model: Trained PerturbationFlowModel
        genes: List of gene names to predict
        nc_cells_pca: PCA-transformed NC cells to use as x0
        device: torch device
        n_cells: Number of cells to generate per gene
        
    Returns:
        predictions: Dict mapping gene -> (predicted_cells_pca, predicted_proportions)
    """
    predictions = {}
    model.eval()
    
    logger.info(f"\nGenerating zero-shot predictions for {len(genes)} genes...")
    logger.info(f"Using {nc_cells_pca.shape[0]} NC cells as x0, sampling {n_cells} per gene")
    
    with torch.no_grad():
        for gene in genes:
            # Sample initial states from NC cells
            if nc_cells_pca.shape[0] >= n_cells:
                indices = np.random.choice(nc_cells_pca.shape[0], n_cells, replace=False)
            else:
                indices = np.random.choice(nc_cells_pca.shape[0], n_cells, replace=True)
            
            x0 = torch.tensor(nc_cells_pca[indices], dtype=torch.float32, device=device)
            
            # Get perturbation embedding using the encoder
            if gene in model.gene_to_idx:
                gene_idx = model.gene_to_idx[gene]
                logger.info(f"  {gene}: Found in knowledge graph at index {gene_idx}")
                
                # Encode perturbation
                perturbation_indices = torch.tensor([gene_idx], dtype=torch.long, device=device)
                z_p = model.encode_perturbation(
                    model.pyg_data["gene"].x.to(device),
                    model.pyg_data["gene", "interacts_with", "gene"].edge_index.to(device),
                    perturbation_indices
                )
                # Repeat for all cells
                z_p = z_p.repeat(n_cells, 1)
            else:
                logger.warning(f"  {gene}: Not in knowledge graph! Using zero embedding")
                z_p = torch.zeros(n_cells, model.perturbation_encoder.perturbation_dim, device=device)
            
            # Generate cells using flow matching
            predicted_pca = model.generate_cells(x0, z_p)
            
            # Generate proportions
            predicted_proportions = model.proportion_head(z_p[0:1])
            
            predictions[gene] = {
                'cells_pca': predicted_pca.cpu().numpy(),
                'proportions': predicted_proportions.cpu().numpy().flatten()
            }
            
            logger.info(f"    Generated {predicted_pca.shape[0]} cells, proportions: {predicted_proportions.cpu().numpy().flatten()[:4]}")
    
    return predictions


def main():
    # Configuration
    config_path = Path("configs/default.yaml")
    checkpoint_path = Path("checkpoints_old/best_true.ckpt")
    gt_path = Path("data/raw/Challenge/program_proportion_local_gtruth.csv")
    
    logger.info("=" * 80)
    logger.info("ZERO-SHOT VALIDATION EVALUATION")
    logger.info("=" * 80)
    
    # Load config
    logger.info(f"\nLoading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load ground truth
    logger.info(f"\nLoading ground truth from {gt_path}")
    ground_truth_df = pd.read_csv(gt_path)
    val_genes = list(ground_truth_df['gene'].values)
    logger.info(f"Validation genes (zero-shot): {val_genes}")
    
    # Load model
    model, kg = load_model(checkpoint_path, config, device)
    
    # Check which validation genes are in the knowledge graph
    logger.info("\nChecking validation genes in knowledge graph:")
    for gene in val_genes:
        in_kg = gene in kg.nodes()
        in_model = gene in model.gene_to_idx if hasattr(model, 'gene_to_idx') else False
        logger.info(f"  {gene}: in_kg={in_kg}, in_model={in_model}")
    
    # Load NC cells for x0 (initial state)
    # We'll load the downsampled dataset used during training
    logger.info("\nLoading NC cells for initial state...")
    raw_dir = Path(config['paths']['raw_dir'])
    h5ad_path = raw_dir / config['paths']['h5ad_file']
    gene_to_predict_path = raw_dir / config['paths']['gene_to_predict']
    
    # Import loader
    from src.data.loader import load_h5ad_data, load_program_proportions, prepare_training_data
    
    # Load with same parameters as training
    adata = load_h5ad_data(
        h5ad_path,
        gene_to_predict_path,
        max_cells=config['data'].get('max_cells'),
        seed=config['training'].get('seed', 42),
        keep_perturbations=config['data'].get('val_genes', []),
    )
    logger.info(f"Loaded {adata.shape[0]} cells")
    
    # Load proportions
    proportions = load_program_proportions(
        raw_dir / config['paths']['program_proportions']
    )
    
    # Prepare training data to get PCA transform
    logger.info("Computing PCA transform...")
    training_data = prepare_training_data(
        adata,
        proportions,
        val_genes=config['data'].get('val_genes', []),
        pca_components=config['data']['pca_components'],
        seed=config['training'].get('seed', 42),
        val_fraction=config['data'].get('val_fraction', 0.05),
    )
    
    pca_model = training_data['pca_model']
    pca_mean = pca_model.mean_
    
    # Get NC cells in PCA space
    nc_mask = training_data['nc_mask']
    nc_cells_pca = training_data['X_pca'][nc_mask]
    logger.info(f"NC cells in PCA space: {nc_cells_pca.shape}")
    
    # Calculate NC centroid in original space for Pearson metric
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    nc_cells_original = X[nc_mask]
    nc_centroid = nc_cells_original.mean(axis=0)
    logger.info(f"NC centroid shape: {nc_centroid.shape}")
    
    # Generate predictions
    predictions = generate_zero_shot_predictions(
        model, val_genes, nc_cells_pca, device, n_cells=100
    )
    
    # Evaluate predictions
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS (Zero-Shot)")
    logger.info("=" * 80)
    
    results = []
    
    for gene in val_genes:
        if gene not in predictions:
            logger.warning(f"\nSkipping {gene} - no predictions generated")
            continue
        
        pred_data = predictions[gene]
        
        # Transform predicted cells back to gene space
        pred_cells_pca = pred_data['cells_pca']
        pred_cells = pca_model.inverse_transform(pred_cells_pca) + pca_mean
        
        # For ground truth, we don't have actual cells, so we use NC centroid
        # This is a limitation - we can't compute real MMD or Pearson without actual perturbed cells
        # But we can compute proportion L1
        
        # Get ground truth proportions
        gt_row = ground_truth_df[ground_truth_df['gene'] == gene].iloc[0]
        gt_proportions = gt_row[['pre_adipo', 'adipo', 'other', 'lipo']].values.astype(float)
        pred_proportions = pred_data['proportions'][:4]  # First 4 programs
        
        l1_score = _l1_distance(gt_proportions, pred_proportions)
        
        results.append({
            'gene': gene,
            'l1_distance': l1_score,
            'n_cells': pred_cells.shape[0],
            'gt_proportions': gt_proportions,
            'pred_proportions': pred_proportions
        })
        
        logger.info(f"\n{gene}:")
        logger.info(f"  Generated cells: {pred_cells.shape}")
        logger.info(f"  L1 Distance: {l1_score:.4f}")
        logger.info(f"  GT proportions:   {gt_proportions}")
        logger.info(f"  Pred proportions: {pred_proportions}")
        logger.info(f"  Difference:       {pred_proportions - gt_proportions}")
    
    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"\nL1 Distance (mean): {results_df['l1_distance'].mean():.4f} ± {results_df['l1_distance'].std():.4f}")
    logger.info(f"L1 Distance (median): {results_df['l1_distance'].median():.4f}")
    logger.info(f"L1 Distance (min): {results_df['l1_distance'].min():.4f}")
    logger.info(f"L1 Distance (max): {results_df['l1_distance'].max():.4f}")
    
    # Save results
    output_path = Path("experiments/zero_shot_validation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NOTE: MMD and Pearson metrics require actual perturbed cells")
    logger.info("The validation genes have NO cells in the dataset (zero-shot)")
    logger.info("Only program proportion L1 distance can be evaluated")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
