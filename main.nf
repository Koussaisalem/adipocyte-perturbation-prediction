#!/usr/bin/env nextflow

/*
 * Adipocyte Perturbation Prediction Pipeline
 * Translates the get_started.ipynb notebook into a Nextflow pipeline
 */

nextflow.enable.dsl=2

// ========================================
// Parameters
// ========================================

params.project_dir = "$projectDir"
params.data_dir = "${params.project_dir}/data"
params.raw_data_dir = "${params.data_dir}/raw/Challenge"
params.processed_dir = "${params.data_dir}/processed"
params.models_dir = "${params.project_dir}/models"
params.configs_dir = "${params.project_dir}/configs"
params.checkpoints_dir = "${params.project_dir}/checkpoints"
params.submissions_dir = "${params.project_dir}/submissions"
params.experiments_dir = "${params.project_dir}/experiments"

// Data files
params.h5ad_file = "obesity_challenge_1.h5ad"
params.h5ad_zip = "obesity_challenge_1.h5ad.small.zip"

// Geneformer parameters
params.geneformer_model_dir = "${params.models_dir}/geneformer_full/gf-12L-104M-i4096"
params.max_cells = 2000
params.chunk_cells = 200
params.batch_size = 1

// Training parameters
params.config = "${params.configs_dir}/default.yaml"
params.seed = 42

// Submission parameters
params.n_cells = 100
params.submission_batch_size = 10

// ========================================
// Processes
// ========================================

process SETUP_ENVIRONMENT {
    label 'cpu'
    
    output:
    path "setup_complete.txt"
    
    script:
    """
    echo "Setting up environment..."
    python -m pip install --upgrade pip > /dev/null 2>&1
    pip install -e ${params.project_dir}[dev,notebooks] > /dev/null 2>&1
    pip install -r ${params.project_dir}/requirements.txt > /dev/null 2>&1
    echo "Setup complete" > setup_complete.txt
    """
}

process VERIFY_GPU {
    label 'gpu'
    
    input:
    path setup_flag
    
    output:
    path "gpu_info.txt"
    
    script:
    """
    nvidia-smi > gpu_info.txt || echo "No GPU available" > gpu_info.txt
    python -c "import torch; print('CUDA:', torch.cuda.is_available())" >> gpu_info.txt
    """
}

process UNZIP_DATA {
    label 'cpu'
    publishDir "${params.raw_data_dir}", mode: 'copy'
    
    input:
    path setup_flag
    
    output:
    path "${params.h5ad_file}", optional: true
    
    script:
    """
    if [ -f ${params.raw_data_dir}/${params.h5ad_file} ]; then
        echo "Data already exists"
        ln -s ${params.raw_data_dir}/${params.h5ad_file} ${params.h5ad_file}
    elif [ -f ${params.raw_data_dir}/${params.h5ad_zip} ]; then
        echo "Unzipping data..."
        unzip -o ${params.raw_data_dir}/${params.h5ad_zip} -d .
    else
        echo "Warning: Data file not found"
    fi
    """
}

process SETUP_HELPER {
    label 'cpu'
    
    input:
    path setup_flag
    
    output:
    path "all_genes.txt"
    
    script:
    """
    cd ${params.project_dir}
    bash setup_codespace.sh
    cp ${params.processed_dir}/all_genes.txt .
    """
}

process BUILD_KNOWLEDGE_GRAPH {
    label 'cpu'
    publishDir "${params.data_dir}/kg", mode: 'copy'
    
    input:
    path gene_list
    
    output:
    path "knowledge_graph.gpickle"
    
    script:
    """
    python ${params.project_dir}/scripts/build_kg.py \\
        --gene-list ${gene_list} \\
        --output knowledge_graph.gpickle \\
        --dorothea-levels A B \\
        --string-threshold 700
    """
}

process FIX_GENEFORMER_ASSETS {
    label 'cpu'
    
    input:
    path setup_flag
    
    output:
    path "geneformer_fixed.txt"
    
    script:
    """
    python << 'PYEOF'
import subprocess
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

# Find geneformer package
gf_result = subprocess.run(
    [sys.executable, "-c", "import geneformer; print(geneformer.__path__[0])"],
    capture_output=True, text=True
)
gf_path = Path(gf_result.stdout.strip())

pkl_files = [
    "gene_dictionaries_gc104M/gene_name_id_dict_gc104M.pkl",
    "gene_dictionaries_gc104M/gene_median_dict_gc104M.pkl",
    "gene_dictionaries_gc104M/ensembl_mapping_dict_gc104M.pkl",
    "gene_dictionaries_gc104M/token_dict_gc104M.pkl",
]

def is_lfs_pointer(filepath):
    if not filepath.exists():
        return True
    with open(filepath, "rb") as f:
        return f.read(20).startswith(b"version https://git-lfs")

needs_fix = any(is_lfs_pointer(gf_path / p) for p in pkl_files)

if needs_fix:
    print("Fixing Git-LFS pointers...")
    for rel_path in pkl_files:
        local_path = gf_path / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id="ctheodoris/Geneformer",
            filename=rel_path,
            local_dir="/tmp/geneformer_assets",
            force_download=True
        )
        shutil.copy2(downloaded, local_path)
    print("✓ Fixed!")
else:
    print("✓ Already valid")

with open("geneformer_fixed.txt", "w") as f:
    f.write("done")
PYEOF
    """
}

process DOWNLOAD_GENEFORMER_MODEL {
    label 'cpu'
    publishDir "${params.models_dir}/geneformer_full", mode: 'copy'
    
    input:
    path geneformer_flag
    
    output:
    path "gf-12L-104M-i4096/**"
    
    when:
    !file("${params.geneformer_model_dir}/pytorch_model.bin").exists()
    
    script:
    """
    python << 'PYEOF'
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path(".")
model_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Geneformer model...")
snapshot_download(
    repo_id="ctheodoris/Geneformer",
    local_dir=str(model_dir)
)
print("✓ Downloaded")
PYEOF
    """
}

process PREPARE_H5AD {
    label 'cpu'
    publishDir "${params.raw_data_dir}", mode: 'copy'
    
    input:
    path h5ad_file
    path geneformer_flag
    
    output:
    path "obesity_challenge_1.prepared.h5ad"
    
    script:
    """
    python << 'PYEOF'
import scanpy as sc
import pickle
import numpy as np
from pathlib import Path
import subprocess
import sys

input_h5ad = Path("${h5ad_file}")
output_h5ad = Path("obesity_challenge_1.prepared.h5ad")

if output_h5ad.exists():
    print("✓ Already prepared")
    sys.exit(0)

print(f"Loading {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)
print(f"Shape: {adata.shape}")

# Add ensembl_id
if "ensembl_id" not in adata.var.columns:
    print("Adding ensembl_id...")
    gf_result = subprocess.run(
        [sys.executable, "-c", "import geneformer; print(geneformer.__path__[0])"],
        capture_output=True, text=True
    )
    gf_path = Path(gf_result.stdout.strip())
    gene_name_dict_path = gf_path / "gene_dictionaries_gc104M" / "gene_name_id_dict_gc104M.pkl"
    
    with open(gene_name_dict_path, "rb") as f:
        gene_name_to_ensembl = pickle.load(f)
    
    ensembl_ids = [
        gene_name_to_ensembl.get(gene, f"UNKNOWN_{gene}")
        for gene in adata.var_names
    ]
    adata.var["ensembl_id"] = ensembl_ids
    print("✓ ensembl_id added")

# Add n_counts
if "n_counts" not in adata.obs.columns:
    print("Adding n_counts...")
    X = adata.X
    n_counts = np.array(X.sum(axis=1)).flatten()
    adata.obs["n_counts"] = n_counts
    print("✓ n_counts added")

print("Saving...")
adata.write_h5ad(output_h5ad)
print("✓ Saved")
PYEOF
    """
}

process EXTRACT_EMBEDDINGS {
    label 'gpu_large'
    publishDir "${params.processed_dir}", mode: 'copy'
    
    input:
    path prepared_h5ad
    path model_dir
    path kg
    
    output:
    path "gene_embeddings.pt"
    
    script:
    """
    python ${params.project_dir}/scripts/extract_embeddings_fixed.py \\
        --h5ad-file ${prepared_h5ad} \\
        --model-dir ${params.geneformer_model_dir} \\
        --max-cells ${params.max_cells} \\
        --chunk-cells ${params.chunk_cells} \\
        --batch-size ${params.batch_size} \\
        --output gene_embeddings.pt
    """
}

process TRAIN_MODEL {
    label 'gpu_large'
    publishDir "${params.checkpoints_dir}", mode: 'copy', pattern: "*.ckpt"
    publishDir "${params.experiments_dir}/logs", mode: 'copy', pattern: "*.log"
    
    input:
    path embeddings
    path kg
    
    output:
    path "best.ckpt"
    path "baseline_run.log"
    
    script:
    """
    python ${params.project_dir}/scripts/train.py \\
        --config ${params.config} \\
        --seed ${params.seed} \\
        2>&1 | tee baseline_run.log
    
    # Copy the best checkpoint
    cp ${params.checkpoints_dir}/best.ckpt . || echo "Checkpoint location may vary"
    """
}

process GENERATE_SUBMISSION {
    label 'gpu'
    publishDir "${params.submissions_dir}", mode: 'copy'
    publishDir "${params.experiments_dir}/logs", mode: 'copy', pattern: "*.log"
    
    input:
    path checkpoint
    
    output:
    path "expression_matrix.csv"
    path "program_proportions.csv"
    path "inference.log"
    
    script:
    """
    python ${params.project_dir}/scripts/generate_submission.py \\
        --checkpoint ${checkpoint} \\
        --output-dir . \\
        --n-cells ${params.n_cells} \\
        --batch-size ${params.submission_batch_size} \\
        2>&1 | tee inference.log
    """
}

process VALIDATE_SUBMISSION {
    label 'cpu'
    publishDir "${params.submissions_dir}", mode: 'copy', pattern: "validation_report.txt"
    
    input:
    path expression_matrix
    path program_proportions
    
    output:
    path "validation_report.txt"
    
    script:
    """
    python << 'PYEOF'
import pandas as pd

# Read files
expr_df = pd.read_csv("${expression_matrix}")
prog_df = pd.read_csv("${program_proportions}")

# Validation
report = []
report.append("=== Validation Report ===")
report.append(f"Expression matrix rows: {len(expr_df)}")
report.append(f"Expected: 286,301 (with header)")
report.append(f"NaNs in expression: {expr_df.isna().sum().sum()}")
report.append(f"Program proportions shape: {prog_df.shape}")

with open("validation_report.txt", "w") as f:
    f.write("\\n".join(report))

print("\\n".join(report))
PYEOF
    """
}

// ========================================
// Workflow
// ========================================

workflow {
    // Setup and verification
    setup_flag = SETUP_ENVIRONMENT()
    gpu_info = VERIFY_GPU(setup_flag)
    
    // Data preparation
    h5ad = UNZIP_DATA(setup_flag)
    gene_list = SETUP_HELPER(setup_flag)
    kg = BUILD_KNOWLEDGE_GRAPH(gene_list)
    
    // Geneformer setup
    geneformer_flag = FIX_GENEFORMER_ASSETS(setup_flag)
    model = DOWNLOAD_GENEFORMER_MODEL(geneformer_flag)
    prepared_h5ad = PREPARE_H5AD(h5ad, geneformer_flag)
    
    // Main pipeline
    embeddings = EXTRACT_EMBEDDINGS(prepared_h5ad, model, kg)
    checkpoint = TRAIN_MODEL(embeddings, kg)
    (expression, proportions, log) = GENERATE_SUBMISSION(checkpoint)
    
    // Validation
    validation = VALIDATE_SUBMISSION(expression, proportions)
    
    // Summary
    validation.view { "Pipeline complete! Check ${params.submissions_dir} for results." }
}

workflow.onComplete {
    println """
    ========================================
    Pipeline execution summary
    ========================================
    Completed at: ${workflow.complete}
    Duration    : ${workflow.duration}
    Success     : ${workflow.success}
    Exit status : ${workflow.exitStatus}
    ========================================
    Results in  : ${params.submissions_dir}
    ========================================
    """
}
