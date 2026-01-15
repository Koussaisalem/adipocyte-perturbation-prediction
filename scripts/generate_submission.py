#!/usr/bin/env python3
"""
Generate submission files for the challenge.

Generates predictions for all test perturbations and formats submission.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_h5ad_data
from src.data.knowledge_graph import load_knowledge_graph, convert_to_pyg
from src.models.full_model import build_model_from_config
from src.inference.predict import (
    PerturbationPredictor,
    generate_submission,
    load_model_for_inference,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate challenge submission")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submissions",
        help="Directory to save submissions",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=100,
        help="Number of cells to generate per perturbation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for inference",
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Construct paths
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    
    # Step 1: Load data
    logger.info("Loading h5ad data...")
    adata = load_h5ad_data(
        raw_dir / config["paths"]["h5ad_file"],
        raw_dir / config["paths"]["gene_to_predict"],
    )
    
    # Step 2: Load training data to get NC cells and PCA
    logger.info("Loading PCA model...")
    pca_path = processed_dir / "pca_model.pkl"
    
    if not pca_path.exists():
        logger.error(f"PCA model not found at {pca_path}")
        logger.error("Please run training first to generate PCA model")
        sys.exit(1)
    
    with open(pca_path, 'rb') as f:
        pca_model = pickle.load(f)
    
    # Get NC cells
    nc_mask = adata.obs["gene"] == "NC"
    from scipy import sparse
    
    if sparse.issparse(adata.X):
        X_nc = adata.X[nc_mask].toarray()
    else:
        X_nc = np.array(adata.X[nc_mask])
    
    # Transform to PCA space
    X_nc_pca = pca_model.transform(X_nc)
    logger.info(f"Loaded {X_nc_pca.shape[0]} NC cells in PCA space")
    
    # Step 3: Load knowledge graph
    logger.info("Loading knowledge graph...")
    kg_path = Path(config["paths"]["kg_dir"]) / "knowledge_graph.gpickle"
    graph = load_knowledge_graph(kg_path)
    
    # Load gene embeddings
    logger.info("Loading gene embeddings...")
    embedding_path = processed_dir / "gene_embeddings.pt"
    gene_embeddings = torch.load(embedding_path)
    
    # Convert to PyG
    pyg_data = convert_to_pyg(
        graph,
        node_features=gene_embeddings,
        feature_dim=config["model"]["embedding_dim"],
    )
    
    # Step 4: Build model and load checkpoint
    logger.info("Building model...")
    model = build_model_from_config(
        config,
        node_features=pyg_data["gene"].x,
        edge_index=pyg_data["gene", "interacts_with", "gene"].edge_index,
    )
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = load_model_for_inference(
        args.checkpoint,
        model,
        device=config["hardware"]["accelerator"],
    )
    
    # Step 5: Create predictor
    logger.info("Creating predictor...")
    predictor = PerturbationPredictor(
        model=model,
        pca_model=pca_model,
        node_features=pyg_data["gene"].x,
        edge_index=pyg_data["gene", "interacts_with", "gene"].edge_index,
        node_to_idx=pyg_data["gene"].node_to_idx,
        device=config["hardware"]["accelerator"],
    )
    
    # Step 6: Generate submissions
    logger.info("Generating submission files...")
    
    expr_path, props_path = generate_submission(
        predictor=predictor,
        perturbation_list_path=raw_dir / config["paths"]["predict_perturbations"],
        gene_list_path=raw_dir / config["paths"]["gene_to_predict"],
        nc_cells_pca=X_nc_pca,
        output_dir=args.output_dir,
        n_cells_per_perturbation=args.n_cells,
        gene_names=list(adata.var_names),
    )
    
    logger.info("Submission generation complete!")
    logger.info(f"Expression matrix: {expr_path}")
    logger.info(f"Proportions CSV: {props_path}")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Validate submission format")
    logger.info("2. Check for NaN/Inf values")
    logger.info("3. Submit to challenge platform")


if __name__ == "__main__":
    main()
