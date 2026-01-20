"""
Path management utilities for the adipocyte perturbation prediction project.

This module provides a single source of truth for all project paths,
ensuring the code works consistently across different machines and environments.

Usage:
    from src.utils.paths import RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR
    
    h5ad_file = RAW_DATA_DIR / "obesity_challenge_1.h5ad"
    output_file = PROCESSED_DIR / "gene_embeddings.pt"
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root directory by looking for marker files.
    
    Searches upward from the current file location until it finds
    a directory containing one of the marker files (pyproject.toml, .git, etc.).
    
    Returns:
        Path: Absolute path to the project root directory
        
    Raises:
        RuntimeError: If no project root can be found
    """
    # Start from this file's directory
    current = Path(__file__).resolve().parent
    
    # Marker files that indicate the project root
    markers = ["pyproject.toml", ".git", "setup.py", "README.md"]
    
    # Walk up the directory tree
    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent
    
    # If we reach the filesystem root without finding markers, raise an error
    raise RuntimeError(
        "Could not find project root. Expected to find one of: "
        f"{', '.join(markers)}"
    )


# Initialize project root (computed once at import time)
PROJECT_ROOT = get_project_root()

# Core directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw" / "Challenge"
PROCESSED_DIR = DATA_DIR / "processed"
KG_DIR = DATA_DIR / "kg"

# Model subdirectories
GENEFORMER_MODEL_DIR = MODELS_DIR / "geneformer_full"

# Commonly used files
ALL_GENES_FILE = PROCESSED_DIR / "all_genes.txt"
KG_FILE = KG_DIR / "knowledge_graph.gpickle"


def ensure_directories():
    """
    Create all standard project directories if they don't exist.
    
    This is useful when setting up the project on a new machine.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DIR,
        KG_DIR,
        MODELS_DIR,
        GENEFORMER_MODEL_DIR,
        CONFIGS_DIR,
        CHECKPOINTS_DIR,
        SUBMISSIONS_DIR,
        EXPERIMENTS_DIR,
        EXPERIMENTS_DIR / "logs",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_relative_path(path: Path, relative_to: Path = None) -> Path:
    """
    Convert an absolute path to a relative path from a reference point.
    
    Args:
        path: The path to convert
        relative_to: Reference point (defaults to PROJECT_ROOT)
        
    Returns:
        Path: Relative path from the reference point
    """
    if relative_to is None:
        relative_to = PROJECT_ROOT
    
    try:
        return path.relative_to(relative_to)
    except ValueError:
        # If path is not relative to reference, return as-is
        return path


# For backwards compatibility with scripts that use environment variables
def get_data_dir() -> Path:
    """Get data directory, with optional override via environment variable."""
    override = os.environ.get("APP_DATA_DIR")
    if override:
        return Path(override)
    return DATA_DIR


def get_models_dir() -> Path:
    """Get models directory, with optional override via environment variable."""
    override = os.environ.get("APP_MODELS_DIR")
    if override:
        return Path(override)
    return MODELS_DIR


if __name__ == "__main__":
    # For testing: print all paths
    print("Project Root:", PROJECT_ROOT)
    print("Data Dir:", DATA_DIR)
    print("Raw Data Dir:", RAW_DATA_DIR)
    print("Processed Dir:", PROCESSED_DIR)
    print("Models Dir:", MODELS_DIR)
    print("Geneformer Model Dir:", GENEFORMER_MODEL_DIR)
    print("Scripts Dir:", SCRIPTS_DIR)
    print("Configs Dir:", CONFIGS_DIR)
    print("Checkpoints Dir:", CHECKPOINTS_DIR)
    print("Submissions Dir:", SUBMISSIONS_DIR)
    print("Experiments Dir:", EXPERIMENTS_DIR)
