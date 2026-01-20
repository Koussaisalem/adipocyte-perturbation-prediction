"""
Minimal zero-shot evaluation - avoids loading full h5ad.
Uses cached training artifacts and generates predictions for validation genes.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.full_model import build_model_from_config
from src.data.knowledge_graph import load_knowledge_graph, convert_to_pyg


def _l1_distance(ground_truth_y, prediction_y):
    """Calculate L1 distance between program proportions."""
    return np.abs(ground_truth_y - prediction_y).sum()


def main():
    logger.info("=" * 80)
    logger.info("MINIMAL ZERO-SHOT VALIDATION")
    logger.info("=" * 80)
    
    # Configuration
    config_path = Path("configs/default.yaml")
    checkpoint_path = Path("checkpoints_old/best_true.ckpt")
    gt_path = Path("data/raw/Challenge/program_proportion_local_gtruth.csv")
    
    logger.info(f"\nLoading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load ground truth
    logger.info(f"Loading ground truth from {gt_path}")
    ground_truth_df = pd.read_csv(gt_path)
    val_genes = list(ground_truth_df['gene'].values)
    logger.info(f"Validation genes: {val_genes}")
    
    # Load model
    logger.info("\n" + "=" * 80)
    logger.info("Loading model...")
    logger.info("=" * 80)
    
    # Load embeddings
    embeddings_path = Path(config['paths']['processed_dir']) / "gene_embeddings.pt"
    gene_embeddings = torch.load(embeddings_path, map_location=device, weights_only=False)
    logger.info(f"Loaded embeddings for {len(gene_embeddings)} genes")
    
    # Load knowledge graph
    kg_path = Path(config['paths']['kg_dir']) / "knowledge_graph.gpickle"
    kg = load_knowledge_graph(kg_path)
    logger.info(f"Loaded KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # Convert to PyG
    pyg_data = convert_to_pyg(kg, node_features=gene_embeddings, feature_dim=config["model"]["embedding_dim"])
    
    # Build model
    model = build_model_from_config(
        config,
        node_features=pyg_data["gene"].x,
        edge_index=pyg_data["gene", "interacts_with", "gene"].edge_index,
    )
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Val MMD: {checkpoint.get('metrics', {}).get('val/mmd', 'N/A')}")
    
    # Create gene_to_idx mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(kg.nodes())}
    
    # Check validation genes in KG
    logger.info("\nValidation genes in knowledge graph:")
    for gene in val_genes:
        in_kg = gene in gene_to_idx
        logger.info(f"  {gene}: {'✓' if in_kg else '✗'}")
    
    # Generate synthetic NC cells in PCA space (just use random samples for x0)
    # In real use, these should come from actual NC cells, but for quick eval we can use zeros
    logger.info("\n" + "=" * 80)
    logger.info("Generating zero-shot predictions...")
    logger.info("=" * 80)
    logger.info("Using synthetic NC baseline (zeros) for x0")
    
    n_cells = 100
    pca_dim = config['data']['pca_components']
    
    results = []
    
    with torch.no_grad():
        for gene in val_genes:
            if gene not in gene_to_idx:
                logger.warning(f"\n{gene}: NOT in KG, skipping")
                continue
            
            logger.info(f"\n{gene}:")
            
            # x0: Start from zero (NC-like state in PCA space)
            # In practice this should be sampled from actual NC cells
            x0 = torch.zeros(n_cells, pca_dim, device=device)
            
            # Encode perturbation
            gene_idx = gene_to_idx[gene]
            perturbation_indices = torch.tensor([gene_idx], dtype=torch.long, device=device)
            
            z_p = model.encode_perturbation(
                pyg_data["gene"].x.to(device),
                pyg_data["gene", "interacts_with", "gene"].edge_index.to(device),
                perturbation_indices
            )
            z_p = z_p.repeat(n_cells, 1)
            
            # Generate cells
            predicted_cells_pca = model.generate_cells(x0, z_p)
            logger.info(f"  Generated {predicted_cells_pca.shape[0]} cells in PCA space")
            
            # Predict proportions
            proportion_output = model.proportion_head(z_p[0:1])
            pred_props = proportion_output['state_proportions'].cpu().numpy().flatten()[:4]
            
            # Get ground truth
            gt_row = ground_truth_df[ground_truth_df['gene'] == gene].iloc[0]
            gt_props = gt_row[['pre_adipo', 'adipo', 'other', 'lipo']].values.astype(float)
            
            # Calculate L1
            l1 = _l1_distance(gt_props, pred_props)
            
            results.append({
                'gene': gene,
                'l1_distance': l1,
                'pre_adipo_pred': pred_props[0],
                'adipo_pred': pred_props[1],
                'other_pred': pred_props[2],
                'lipo_pred': pred_props[3],
                'pre_adipo_gt': gt_props[0],
                'adipo_gt': gt_props[1],
                'other_gt': gt_props[2],
                'lipo_gt': gt_props[3],
            })
            
            logger.info(f"  Predicted proportions: {pred_props}")
            logger.info(f"  Ground truth:          {gt_props}")
            logger.info(f"  L1 Distance:           {l1:.4f}")
    
    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"\nL1 Distance Statistics:")
    logger.info(f"  Mean:   {results_df['l1_distance'].mean():.4f}")
    logger.info(f"  Std:    {results_df['l1_distance'].std():.4f}")
    logger.info(f"  Median: {results_df['l1_distance'].median():.4f}")
    logger.info(f"  Min:    {results_df['l1_distance'].min():.4f} ({results_df.loc[results_df['l1_distance'].idxmin(), 'gene']})")
    logger.info(f"  Max:    {results_df['l1_distance'].max():.4f} ({results_df.loc[results_df['l1_distance'].idxmax(), 'gene']})")
    
    # Save
    output_path = Path("experiments/zero_shot_minimal_results.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to {output_path}")
    
    # Display results table
    logger.info("\nDetailed Results:")
    logger.info(results_df.to_string(index=False))
    
    logger.info("\n" + "=" * 80)
    logger.info("NOTES:")
    logger.info("- Using synthetic x0 (zeros) instead of actual NC cells")
    logger.info("- Real evaluation would use NC cell samples from training data")
    logger.info("- MMD/Pearson cannot be computed without actual perturbed cell ground truth")
    logger.info("- L1 on proportions is the only metric we can evaluate for zero-shot genes")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
