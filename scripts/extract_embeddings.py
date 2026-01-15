#!/usr/bin/env python3
"""
Extract gene embeddings using Geneformer.

This script should be run on a machine with GPU and Geneformer installed.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract Geneformer embeddings")
    parser.add_argument(
        "--h5ad-file",
        type=str,
        required=True,
        help="Path to h5ad file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="geneformer-106m",
        choices=["geneformer-10m", "geneformer-106m"],
        help="Geneformer model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/gene_embeddings.pt",
        help="Path to save embeddings",
    )
    parser.add_argument(
        "--gene-list",
        type=str,
        default=None,
        help="Optional: specific genes to extract (otherwise all genes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    
    args = parser.parse_args()
    
    try:
        from geneformer import TranscriptomeTokenizer, EmbExtractor
    except ImportError:
        logger.error("Geneformer not installed. Install with: pip install geneformer")
        sys.exit(1)
    
    # Load h5ad
    import scanpy as sc
    logger.info(f"Loading h5ad from {args.h5ad_file}")
    adata = sc.read_h5ad(args.h5ad_file)
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Load specific genes if provided
    if args.gene_list:
        with open(args.gene_list) as f:
            genes = [line.strip() for line in f if line.strip()]
        logger.info(f"Filtering to {len(genes)} genes")
        adata = adata[:, [g for g in genes if g in adata.var_names]]
    
    # Initialize Geneformer
    logger.info(f"Loading Geneformer model: {args.model}")
    
    # This is a placeholder - actual Geneformer usage may differ
    # You'll need to adapt based on the actual Geneformer API
    
    logger.warning("Geneformer embedding extraction not fully implemented")
    logger.warning("Please refer to Geneformer documentation for proper usage")
    
    # Placeholder: create random embeddings
    logger.info("Creating placeholder embeddings (random)")
    n_genes = adata.n_vars
    embedding_dim = 512 if "106m" in args.model else 256
    
    embeddings = {}
    for gene in adata.var_names:
        embeddings[gene] = torch.randn(embedding_dim)
    
    # Save embeddings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(embeddings, output_path)
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"  {len(embeddings)} genes x {embedding_dim} dimensions")
    
    logger.info("Done!")
    logger.info("")
    logger.info("NOTE: This script uses placeholder embeddings.")
    logger.info("For production, implement proper Geneformer embedding extraction.")


if __name__ == "__main__":
    main()
