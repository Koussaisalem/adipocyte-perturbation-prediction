"""
CrunchDAO Submission - Adipocyte Perturbation Prediction
Main interface for train() and infer() functions.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import anndata as ad

# Import utilities from same directory
from utils import load_knowledge_graph, convert_to_pyg

# Import model builder - this is in src which should work
sys.path.insert(0, str(Path(__file__).parent))
from src.models.full_model import build_model_from_config


# Hardcoded config to avoid yaml dependency (matches configs/default.yaml)
CONFIG = {
    "model": {
        "embedding_dim": 512,  # Geneformer embeddings
        "gat_layers": 3,
        "gat_heads": 8,
        "gat_hidden_dim": 128,
        "gat_dropout": 0.1,
        "perturbation_dim": 256,
        "perturbation_aggregation": "attention"
    },
    "flow_matching": {
        "hidden_dims": [1024, 512, 512],
        "time_embedding_dim": 64,
        "pca_components": 300,
        "activation": "silu",
        "ode_solver": "dopri5",
        "ode_steps": 20,
        "ode_rtol": 1e-5,
        "ode_atol": 1e-5,
        "n_cells_per_perturbation": 100
    },
    "proportion_head": {
        "hidden_dims": [128, 64],
        "n_programs": 4,
        "predict_ratio": True,
        "dropout": 0.1,
        "activation": "relu"
    },
    "data": {
        "pca_components": 300
    }
}


def train(X_train, y_train, model_directory_path):
    """
    Train function - called once during submission phase.
    
    Args:
        X_train: Training data (not used - we have pre-trained model)
        y_train: Training labels (not used - we have pre-trained model)
        model_directory_path: Path to resources/ directory where model is saved
    """
    print("=" * 80)
    print("TRAIN PHASE")
    print("=" * 80)
    print("\nNote: Using pre-trained model from checkpoint")
    print("No additional training performed during submission")
    
    # The model is already trained - we just need to copy artifacts to resources/
    # This happens during crunch push when you include files
    
    print("\n✓ Train phase complete")
    print(f"✓ Model artifacts in: {model_directory_path}")


def infer(model_directory_path, data_directory_path=None):
    """
    Inference function - called on new test data.
    
    Args:
        model_directory_path: Path to resources/ directory with saved model
        data_directory_path: Path to data directory with predict_perturbations.txt, etc.
        
    Returns:
        dict with:
            - 'prediction.h5ad': AnnData object with expression predictions
            - 'predict_program_proportion.csv': DataFrame with proportions
    """
    # VERSION MARKER - MUST APPEAR FIRST IN LOGS
    print("\n" + "#" * 80)
    print("### CODE VERSION: V11 - NO LIPO_ADIPO COLUMN ###")
    print("#" * 80 + "\n")
    
    print("=" * 80)
    print("INFERENCE PHASE")
    print("=" * 80)
    
    model_dir = Path(model_directory_path)
    
    # Handle data directory - CrunchDAO provides it as /context/data/
    if data_directory_path is None:
        data_dir = Path("/context/data")
    else:
        data_dir = Path(data_directory_path)
    
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    
    # Read perturbations to predict from file
    perturbations_file = data_dir / "predict_perturbations.txt"
    if perturbations_file.exists():
        with open(perturbations_file, 'r') as f:
            perturbations_to_predict = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(perturbations_to_predict)} perturbations from {perturbations_file}")
    else:
        print(f"Warning: {perturbations_file} not found, using test genes")
        perturbations_to_predict = ['ZBTB20', 'FOXC1', 'SOX6', 'CHD4', 'TRIM5']
    
    # Use hardcoded config
    config = CONFIG
    print(f"\nUsing hardcoded configuration")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load gene embeddings
    embeddings_path = model_dir / "gene_embeddings.pt"
    print(f"Loading embeddings from {embeddings_path}")
    gene_embeddings = torch.load(embeddings_path, map_location=device, weights_only=False)
    
    # Load knowledge graph
    kg_path = model_dir / "knowledge_graph.gpickle"
    print(f"Loading KG from {kg_path}")
    kg = load_knowledge_graph(kg_path)
    
    # Convert to PyG
    pyg_data = convert_to_pyg(
        kg, 
        node_features=gene_embeddings, 
        feature_dim=config["model"]["embedding_dim"]
    )
    
    # Build model
    print("Building model...")
    model = build_model_from_config(
        config,
        node_features=pyg_data["gene"].x,
        edge_index=pyg_data["gene", "interacts_with", "gene"].edge_index,
    )
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = model_dir / "model.ckpt"
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Store PyG data
    model.pyg_data = pyg_data
    
    # Load or create PCA (stored as numpy arrays for sklearn version compatibility)
    pca_path = model_dir / "pca_cache.pkl"
    if pca_path.exists():
        print(f"Loading PCA from {pca_path}")
        with open(pca_path, 'rb') as f:
            pca_cache = pickle.load(f)
        # Check if it's numpy arrays (new format) or sklearn object (old format)
        if 'components' in pca_cache:
            # New format: numpy arrays
            pca_components = pca_cache['components']  # (300, 11046)
            pca_mean = pca_cache['mean']  # (11046,)
            pca_gene_names = pca_cache['gene_names']
        else:
            # Old format: sklearn object (may have version issues)
            pca_model = pca_cache['pca_model']
            pca_mean = pca_cache['pca_mean']
            pca_gene_names = pca_cache['gene_names']
            pca_components = pca_model.components_
    else:
        raise FileNotFoundError(
            f"PCA cache not found at {pca_path}. "
            "This should be created during the train() phase."
        )
    
    # Read genes_to_predict from data directory (output columns for expression matrix)
    genes_to_predict_file = data_dir / "genes_to_predict.txt"
    if genes_to_predict_file.exists():
        with open(genes_to_predict_file, 'r') as f:
            genes_to_predict_output = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(genes_to_predict_output)} output genes from {genes_to_predict_file}")
    else:
        # Fallback to PCA gene names
        genes_to_predict_output = pca_gene_names
        print(f"Using {len(genes_to_predict_output)} genes from PCA cache")
    
    # Create gene_to_idx mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(kg.nodes())}
    
    # perturbations_to_predict was loaded earlier from file
    print(f"\n✓ Setup complete")
    print(f"Predicting for {len(perturbations_to_predict)} perturbations...")
    print(f"Output genes: {len(genes_to_predict_output)}")
    
    # Generate predictions
    n_cells_per_perturbation = 100
    pca_dim = config['data']['pca_components']
    
    # Storage for all predictions
    all_cells_expression = []  # Will be (n_perturbations * 100, n_genes)
    all_perturbation_labels = []  # Gene labels for each cell
    proportion_rows = []
    
    with torch.no_grad():
        for idx, perturbation_gene in enumerate(perturbations_to_predict):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(perturbations_to_predict)}")
            
            if perturbation_gene not in gene_to_idx:
                print(f"  Warning: {perturbation_gene} not in KG, using baseline")
                # Use baseline NC cells
                predicted_cells = np.tile(pca_mean, (n_cells_per_perturbation, 1))
                props = [0.25, 0.25, 0.25, 0.25]  # Uniform
            else:
                # x0: Start from NC baseline
                x0 = torch.zeros(n_cells_per_perturbation, pca_dim, device=device)
                
                # Encode perturbation
                gene_idx = gene_to_idx[perturbation_gene]
                perturbation_indices = torch.tensor([gene_idx], dtype=torch.long, device=device)
                
                z_p = model.encode_perturbation(
                    pyg_data["gene"].x.to(device),
                    pyg_data["gene", "interacts_with", "gene"].edge_index.to(device),
                    perturbation_indices
                )
                z_p = z_p.repeat(n_cells_per_perturbation, 1)
                
                # Generate cells in PCA space
                predicted_cells_pca = model.generate_cells(x0, z_p)
                
                # Transform back to gene expression space using numpy (version-independent)
                # inverse_transform: X = X_pca @ components + mean
                predicted_cells = predicted_cells_pca.cpu().numpy() @ pca_components + pca_mean
                
                # Get proportions
                proportion_output = model.proportion_head(z_p[0:1])
                props = proportion_output['state_proportions'].cpu().numpy().flatten()[:4]
            
            # Store expression for all 100 cells
            all_cells_expression.append(predicted_cells)
            all_perturbation_labels.extend([perturbation_gene] * n_cells_per_perturbation)
            
            # Store proportions with required format
            # Columns: gene, pre_adipo, adipo, lipo, other (NO lipo_adipo - they compute it)
            proportion_rows.append({
                'gene': perturbation_gene,
                'pre_adipo': float(props[0]),
                'adipo': float(props[1]),
                'lipo': float(props[3]),
                'other': float(props[2])
            })
    
    print(f"\n✓ Generated predictions for {len(perturbations_to_predict)} perturbations")
    
    # Concatenate all cells: shape (n_perturbations * 100, n_genes_pca)
    all_cells_expression = np.vstack(all_cells_expression)
    
    print(f"Expression matrix shape (from PCA): {all_cells_expression.shape}")
    
    # Subset to only the genes requested for output
    # pca_gene_names are the full 11046 genes, genes_to_predict_output may be a subset
    if len(genes_to_predict_output) != len(pca_gene_names):
        # Create mapping from pca genes to output genes
        pca_gene_to_idx = {g: i for i, g in enumerate(pca_gene_names)}
        output_indices = [pca_gene_to_idx.get(g) for g in genes_to_predict_output]
        
        # Check for missing genes
        missing = [g for g, i in zip(genes_to_predict_output, output_indices) if i is None]
        if missing:
            print(f"Warning: {len(missing)} genes not in PCA, filling with zeros")
        
        # Select columns, fill with zeros for missing genes
        all_cells_subset = np.zeros((all_cells_expression.shape[0], len(genes_to_predict_output)))
        for j, idx in enumerate(output_indices):
            if idx is not None:
                all_cells_subset[:, j] = all_cells_expression[:, idx]
        all_cells_expression = all_cells_subset
    
    print(f"Final expression matrix shape: {all_cells_expression.shape}")
    print(f"Expected: ({len(perturbations_to_predict) * 100}, {len(genes_to_predict_output)})")
    
    # Create AnnData object
    adata = ad.AnnData(
        X=all_cells_expression,
        obs=pd.DataFrame({'gene': all_perturbation_labels}),
        var=pd.DataFrame(index=genes_to_predict_output)
    )
    
    # Create proportions DataFrame with explicit column order
    proportions_df = pd.DataFrame(proportion_rows)
    proportions_df = proportions_df[['gene', 'pre_adipo', 'adipo', 'lipo', 'other']]
    
    print(f"✓ AnnData shape: {adata.shape}")
    print(f"✓ Proportions shape: {proportions_df.shape}")
    print(f"✓ Proportions columns: {list(proportions_df.columns)}")
    
    # Write outputs to /context/prediction/ for CrunchDAO
    print("\n" + "=" * 80)
    print("FILE WRITING PHASE - DEBUG V8")
    print("=" * 80)
    
    import os
    prediction_dir = Path("/context/prediction")
    print(f"[DEBUG] prediction_dir = {prediction_dir}")
    print(f"[DEBUG] prediction_dir.exists() = {prediction_dir.exists()}")
    print(f"[DEBUG] os.getcwd() = {os.getcwd()}")
    print(f"[DEBUG] os.listdir('/context') = {os.listdir('/context') if os.path.exists('/context') else 'NOT FOUND'}")
    
    try:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] mkdir completed, exists now = {prediction_dir.exists()}")
    except Exception as e:
        print(f"[DEBUG] mkdir FAILED: {e}")
    
    h5ad_path = prediction_dir / "prediction.h5ad"
    csv_path = prediction_dir / "predict_program_proportion.csv"
    
    print(f"[DEBUG] Writing h5ad to {h5ad_path}...")
    try:
        adata.write_h5ad(h5ad_path)
        print(f"[DEBUG] h5ad written, file exists = {h5ad_path.exists()}, size = {h5ad_path.stat().st_size if h5ad_path.exists() else 'N/A'}")
    except Exception as e:
        print(f"[DEBUG] h5ad write FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] Writing csv to {csv_path}...")
    try:
        proportions_df.to_csv(csv_path, index=False)
        print(f"[DEBUG] csv written, file exists = {csv_path.exists()}, size = {csv_path.stat().st_size if csv_path.exists() else 'N/A'}")
    except Exception as e:
        print(f"[DEBUG] csv write FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] Final /context/prediction contents: {os.listdir(prediction_dir) if prediction_dir.exists() else 'DIR NOT FOUND'}")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    
    # Return as dict with required file names
    return {
        'prediction.h5ad': adata,
        'predict_program_proportion.csv': proportions_df
    }


if __name__ == '__main__':
    # Local testing
    print("Testing locally...")
    
    model_dir = Path("resources")
    # For local testing, use test_data with just 5 perturbations
    data_dir = Path("test_data")
    
    if model_dir.exists():
        result = infer(str(model_dir), str(data_dir))
        
        print("\nSample AnnData:")
        print(result['prediction.h5ad'])
        print("\nSample proportions:")
        print(result['predict_program_proportion.csv'].head())
        
        # Save to disk for testing
        result['prediction.h5ad'].write_h5ad("test_prediction.h5ad")
        result['predict_program_proportion.csv'].to_csv("test_predict_program_proportion.csv", index=False)
        print("\n✓ Saved test outputs")
    else:
        print(f"Model directory {model_dir} not found")
        print("Run package_submission.py first to copy model artifacts")

