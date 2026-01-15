#!/usr/bin/env python3
"""
Train the perturbation flow model.

Trains the KG-conditioned flow matching model on perturbation data.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    load_h5ad_data,
    load_signature_genes,
    compute_program_scores,
    load_program_proportions,
    prepare_training_data,
)
from src.data.knowledge_graph import load_knowledge_graph, convert_to_pyg
from src.data.dataset import create_dataloaders
from src.models.full_model import build_model_from_config
from src.losses.combined import create_loss_from_config
from src.training.trainer import Trainer, TrainingConfig
from src.training.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train perturbation flow model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set seed
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    
    logger.info(f"Using seed: {seed}")
    
    # Construct paths
    data_dir = Path(config["paths"]["data_dir"])
    raw_dir = Path(config["paths"]["raw_dir"])
    
    # Step 1: Load data
    logger.info("Loading h5ad data...")
    adata = load_h5ad_data(
        raw_dir / config["paths"]["h5ad_file"],
        raw_dir / config["paths"]["gene_to_predict"],
    )
    
    # Load signature genes
    logger.info("Loading signature genes...")
    signature_genes = load_signature_genes(
        raw_dir / config["paths"]["signature_genes"]
    )
    
    # Compute program scores
    logger.info("Computing program scores...")
    program_scores = compute_program_scores(
        adata,
        signature_genes,
        method=config["data"]["program_score_method"],
    )
    
    # Load proportions
    logger.info("Loading program proportions...")
    proportions = load_program_proportions(
        raw_dir / config["paths"]["program_proportions"]
    )
    
    # Prepare training data
    logger.info("Preparing training data...")
    training_data = prepare_training_data(
        adata,
        proportions,
        val_genes=config["data"]["val_genes"],
        pca_components=config["data"]["pca_components"],
    )
    
    # Step 2: Load knowledge graph
    logger.info("Loading knowledge graph...")
    kg_path = Path(config["paths"]["kg_dir"]) / "knowledge_graph.gpickle"
    
    if not kg_path.exists():
        logger.error(f"Knowledge graph not found at {kg_path}")
        logger.error("Please run scripts/build_kg.py first")
        sys.exit(1)
    
    graph = load_knowledge_graph(kg_path)
    
    # Load gene embeddings
    logger.info("Loading gene embeddings...")
    embedding_path = Path(config["paths"]["processed_dir"]) / "gene_embeddings.pt"
    
    if not embedding_path.exists():
        logger.error(f"Gene embeddings not found at {embedding_path}")
        logger.error("Please run scripts/extract_embeddings.py first")
        sys.exit(1)
    
    gene_embeddings = torch.load(embedding_path)
    
    # Convert to PyG
    logger.info("Converting graph to PyG format...")
    pyg_data = convert_to_pyg(
        graph,
        node_features=gene_embeddings,
        feature_dim=config["model"]["embedding_dim"],
    )
    
    # Step 3: Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        training_data,
        batch_size=config["training"]["batch_size"],
        n_samples_per_perturbation=1000,
        num_workers=config["hardware"]["num_workers"],
        seed=seed,
    )
    
    # Step 4: Build model
    logger.info("Building model...")
    model = build_model_from_config(
        config,
        node_features=pyg_data["gene"].x,
        edge_index=pyg_data["gene", "interacts_with", "gene"].edge_index,
    )
    
    # Step 5: Create loss
    logger.info("Creating loss function...")
    loss_fn = create_loss_from_config(config)
    
    # Step 6: Create trainer
    logger.info("Creating trainer...")
    
    # Training config
    train_config = TrainingConfig(
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        gradient_clip=config["training"]["gradient_clip"],
        val_frequency=config["training"]["val_frequency"],
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        save_top_k=config["training"]["save_top_k"],
        monitor=config["training"]["monitor"],
        mode=config["training"]["mode"],
        early_stopping=config["training"]["early_stopping"],
        patience=config["training"]["patience"],
        device=config["hardware"]["accelerator"],
        seed=seed,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=train_config.checkpoint_dir,
            monitor=train_config.monitor,
            mode=train_config.mode,
            save_top_k=train_config.save_top_k,
        ),
    ]
    
    if train_config.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=train_config.monitor,
                mode=train_config.mode,
                patience=train_config.patience,
            )
        )
    
    # Learning rate scheduler
    scheduler = LearningRateScheduler(
        optimizer=None,  # Will be created by trainer
        mode=config["training"]["scheduler"],
        warmup_epochs=config["training"]["warmup_epochs"],
        total_epochs=train_config.epochs,
        min_lr=config["training"]["min_lr"],
        initial_lr=train_config.lr,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        scheduler=scheduler,
        callbacks=callbacks,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Step 7: Train
    logger.info("Starting training...")
    trainer.fit()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
