"""
Generate submission for 2,863 test genes.
Optimized to avoid memory issues by caching PCA and using minimal data loading.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.full_model import build_model_from_config
from src.data.knowledge_graph import load_knowledge_graph, convert_to_pyg


def load_or_create_pca(config, cache_path='data/processed/pca_cache.pkl'):
    """Load PCA model from cache or create it."""
    cache_path = Path(cache_path)
    
    if cache_path.exists():
        logger.info(f"Loading cached PCA from {cache_path}")
        with open(cache_path, 'rb') as f:
            pca_cache = pickle.load(f)
        return pca_cache['pca_model'], pca_cache['pca_mean'], pca_cache['gene_names']
    
    # Need to create PCA - load data minimally
    logger.info("PCA cache not found. Creating from training data...")
    logger.info("This requires loading h5ad file once (may take 1-2 minutes)...")
    
    from src.data.loader import load_h5ad_data, load_program_proportions, prepare_training_data
    import scanpy as sc
    
    raw_dir = Path(config['paths']['raw_dir'])
    h5ad_path = raw_dir / config['paths']['h5ad_file']
    gene_to_predict_path = raw_dir / config['paths']['gene_to_predict']
    
    # Load with same downsampling as training
    adata = load_h5ad_data(
        h5ad_path,
        gene_to_predict_path,
        max_cells=config['data'].get('max_cells'),
        seed=config['training'].get('seed', 42),
        keep_perturbations=config['data'].get('val_genes', []),
    )
    logger.info(f"Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Load proportions
    proportions = load_program_proportions(
        raw_dir / config['paths']['program_proportions']
    )
    
    # Prepare training data to get PCA
    logger.info("Computing PCA...")
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
    gene_names = training_data['gene_names']
    
    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pca_cache = {
        'pca_model': pca_model,
        'pca_mean': pca_mean,
        'gene_names': gene_names
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(pca_cache, f)
    logger.info(f"Cached PCA to {cache_path}")
    
    return pca_model, pca_mean, gene_names


def generate_submission(model, gene_to_idx, genes_to_predict, pca_model, pca_mean, gene_names, 
                       device, n_cells=100, pca_dim=300):
    """
    Generate predictions for all test genes.
    
    Returns:
        expression_df: DataFrame with genes × (n_cells * n_genes) expression values
        proportions_df: DataFrame with genes × 4 program proportions
    """
    pyg_data = model.pyg_data
    
    # Storage
    expression_rows = []
    proportion_rows = []
    
    logger.info(f"\nGenerating predictions for {len(genes_to_predict)} genes...")
    
    with torch.no_grad():
        for gene in tqdm(genes_to_predict, desc="Generating predictions"):
            if gene not in gene_to_idx:
                logger.warning(f"{gene} not in knowledge graph, skipping")
                continue
            
            # x0: Synthetic NC baseline
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
            
            # Generate cells in PCA space
            predicted_cells_pca = model.generate_cells(x0, z_p)
            
            # Transform to gene space
            predicted_cells = pca_model.inverse_transform(predicted_cells_pca.cpu().numpy()) + pca_mean
            
            # Flatten cells for expression matrix: [n_cells * n_genes]
            expression_flat = predicted_cells.flatten()
            expression_rows.append([gene] + expression_flat.tolist())
            
            # Get proportions
            proportion_output = model.proportion_head(z_p[0:1])
            props = proportion_output['state_proportions'].cpu().numpy().flatten()[:4]
            
            proportion_rows.append({
                'gene': gene,
                'pre_adipo': props[0],
                'adipo': props[1],
                'other': props[2],
                'lipo': props[3]
            })
    
    # Create expression dataframe
    # Columns: gene, cell1_gene1, cell1_gene2, ..., cell1_geneN, cell2_gene1, ...
    expression_cols = ['gene']
    for cell_idx in range(n_cells):
        for gene_name in gene_names:
            expression_cols.append(f'cell{cell_idx+1}_{gene_name}')
    
    expression_df = pd.DataFrame(expression_rows, columns=expression_cols)
    
    # Create proportions dataframe
    proportions_df = pd.DataFrame(proportion_rows)
    
    return expression_df, proportions_df


def main():
    logger.info("=" * 80)
    logger.info("SUBMISSION GENERATION")
    logger.info("=" * 80)
    
    # Configuration
    config_path = Path("configs/default.yaml")
    checkpoint_path = Path("checkpoints_old/best_true.ckpt")
    output_dir = Path("submissions")
    predict_file = Path("data/raw/Challenge/predict_perturbations.txt")
    
    logger.info(f"\nLoading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load genes to predict
    logger.info(f"\nLoading genes to predict from {predict_file}")
    with open(predict_file, 'r') as f:
        genes_to_predict = [line.strip() for line in f if line.strip()]
    logger.info(f"Found {len(genes_to_predict)} genes to predict")
    
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
    
    # Store PyG data in model
    model.pyg_data = pyg_data
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Val MMD: {checkpoint.get('metrics', {}).get('val/mmd', 'N/A')}")
    
    # Create gene_to_idx mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(kg.nodes())}
    
    # Check how many test genes are in KG
    genes_in_kg = [g for g in genes_to_predict if g in gene_to_idx]
    logger.info(f"\n{len(genes_in_kg)}/{len(genes_to_predict)} test genes found in knowledge graph")
    
    # Load or create PCA
    logger.info("\n" + "=" * 80)
    logger.info("Loading PCA transformation...")
    logger.info("=" * 80)
    
    pca_model, pca_mean, gene_names = load_or_create_pca(config)
    logger.info(f"PCA: {pca_model.n_components} components, {len(gene_names)} genes")
    
    # Generate submission
    logger.info("\n" + "=" * 80)
    logger.info("Generating submission files...")
    logger.info("=" * 80)
    
    expression_df, proportions_df = generate_submission(
        model=model,
        gene_to_idx=gene_to_idx,
        genes_to_predict=genes_in_kg,
        pca_model=pca_model,
        pca_mean=pca_mean,
        gene_names=gene_names,
        device=device,
        n_cells=100,
        pca_dim=config['data']['pca_components']
    )
    
    # Save submission files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    expression_path = output_dir / "expression_matrix.csv"
    proportions_path = output_dir / "program_proportions.csv"
    
    logger.info(f"\nSaving submission files...")
    logger.info(f"  Expression matrix: {expression_path}")
    expression_df.to_csv(expression_path, index=False)
    
    logger.info(f"  Program proportions: {proportions_path}")
    proportions_df.to_csv(proportions_path, index=False)
    
    # Validation checks
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 80)
    
    logger.info(f"\nExpression matrix shape: {expression_df.shape}")
    logger.info(f"  Expected rows: {len(genes_in_kg)} genes")
    logger.info(f"  Expected cols: 1 (gene) + {100 * len(gene_names)} (100 cells × {len(gene_names)} genes)")
    
    logger.info(f"\nProgram proportions shape: {proportions_df.shape}")
    logger.info(f"  Expected: ({len(genes_in_kg)}, 5) - gene + 4 programs")
    
    # Check for NaNs
    expr_nans = expression_df.iloc[:, 1:].isna().sum().sum()
    prop_nans = proportions_df.iloc[:, 1:].isna().sum().sum()
    
    logger.info(f"\nNaN check:")
    logger.info(f"  Expression matrix: {expr_nans} NaNs {'✓' if expr_nans == 0 else '✗'}")
    logger.info(f"  Proportions: {prop_nans} NaNs {'✓' if prop_nans == 0 else '✗'}")
    
    # Sample data
    logger.info(f"\nSample proportions (first 5 genes):")
    logger.info(proportions_df.head().to_string(index=False))
    
    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nFiles ready for upload:")
    logger.info(f"  {expression_path}")
    logger.info(f"  {proportions_path}")


if __name__ == '__main__':
    main()
