# CrunchDAO Submission Package Script
# Copy required model artifacts to resources/ directory

import shutil
from pathlib import Path

def package_submission():
    """Package model artifacts for CrunchDAO submission."""
    
    print("=" * 80)
    print("PACKAGING SUBMISSION")
    print("=" * 80)
    
    # Source paths (your trained model)
    source_dir = Path(".")
    checkpoint_path = source_dir / "checkpoints_old/best_true.ckpt"
    config_path = source_dir / "configs/default.yaml"
    embeddings_path = source_dir / "data/processed/gene_embeddings.pt"
    kg_path = source_dir / "data/kg/knowledge_graph.gpickle"
    
    # Destination (crunch submission resources/)
    dest_dir = Path("crunch_submission_template/resources")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    files_to_copy = [
        (checkpoint_path, dest_dir / "model.ckpt"),
        (config_path, dest_dir / "config.yaml"),
        (embeddings_path, dest_dir / "gene_embeddings.pt"),
        (kg_path, dest_dir / "knowledge_graph.gpickle"),
    ]
    
    print("\nCopying files:")
    for src, dst in files_to_copy:
        if src.exists():
            print(f"  ✓ {src.name} → {dst.name}")
            shutil.copy2(src, dst)
        else:
            print(f"  ✗ {src.name} NOT FOUND!")
            return False
    
    # Check if PCA cache exists
    pca_cache_path = source_dir / "data/processed/pca_cache.pkl"
    if pca_cache_path.exists():
        print(f"  ✓ pca_cache.pkl → pca_cache.pkl")
        shutil.copy2(pca_cache_path, dest_dir / "pca_cache.pkl")
    else:
        print(f"  ⚠ pca_cache.pkl not found - will be created during first inference")
    
    # Copy src/ directory
    src_code_dir = source_dir / "src"
    dest_src_dir = Path("crunch_submission_template/src")
    if dest_src_dir.exists():
        shutil.rmtree(dest_src_dir)
    print(f"  ✓ src/ → src/")
    shutil.copytree(src_code_dir, dest_src_dir)
    
    print("\n" + "=" * 80)
    print("PACKAGE COMPLETE!")
    print("=" * 80)
    print(f"\nSubmission ready in: crunch_submission_template/")
    print("\nNext steps:")
    print("1. cd crunch_submission_template/")
    print("2. Test locally: python main.py")
    print("3. Setup crunch: crunch setup <competition> <model> --token <token> .")
    print("4. Push: crunch push --message 'First submission'")
    
    return True

if __name__ == '__main__':
    package_submission()
