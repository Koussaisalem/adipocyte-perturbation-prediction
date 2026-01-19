# Running the Nextflow Pipeline

This document explains how to run the adipocyte perturbation prediction pipeline using Nextflow.

## Prerequisites

1. **Nextflow installed**:
   ```bash
   curl -s https://get.nextflow.io | bash
   sudo mv nextflow /usr/local/bin/
   ```

2. **Data files in place**:
   - Ensure data is in `data/raw/Challenge/`
   - The pipeline expects `obesity_challenge_1.h5ad` or `obesity_challenge_1.h5ad.small.zip`

3. **Python environment**:
   - Python 3.10+
   - All dependencies (will be installed by pipeline)

## Quick Start

### Local Execution (with GPU)

```bash
# Run with default parameters
nextflow run main.nf -profile local_gpu

# Run with custom parameters
nextflow run main.nf -profile local_gpu \
  --max_cells 5000 \
  --chunk_cells 500 \
  --batch_size 2
```

### Camber Cloud Execution

```bash
# Upload to Camber stash first
camber stash upload . stash://koussaisalem/adipocyte-perturbation-prediction/

# Create Nextflow job on Camber
camber job create \
  --engine nextflow \
  --gpu \
  --size medium \
  --pipeline main.nf \
  --params "max_cells=2000,chunk_cells=200" \
  --path stash://koussaisalem/adipocyte-perturbation-prediction/
```

### HPC/SLURM Execution

```bash
nextflow run main.nf -profile hpc \
  --max_cells 10000 \
  --chunk_cells 1000
```

## Configuration Profiles

### `standard` (default)
- Local execution
- 4 CPUs, 16GB RAM
- No GPU

### `local_gpu`
- Local execution with GPU
- GPU processes: 8-16 CPUs, 32-64GB RAM
- Requires CUDA-capable GPU

### `camber`
- Optimized for Camber Cloud
- GPU medium size for heavy processes
- Automatic GPU allocation

### `hpc`
- SLURM job scheduler
- GPU partition with L4 GPUs
- Configurable queue and time limits

## Pipeline Parameters

### Data Parameters
```bash
--project_dir       # Project root (default: $projectDir)
--data_dir          # Data directory (default: data/)
--raw_data_dir      # Raw data location (default: data/raw/Challenge/)
--h5ad_file         # Input h5ad filename
```

### Geneformer Parameters
```bash
--max_cells         # Maximum cells to process (default: 2000)
--chunk_cells       # Cells per chunk (default: 200)
--batch_size        # Batch size for GPU (default: 1)
```

### Training Parameters
```bash
--config            # Training config file (default: configs/default.yaml)
--seed              # Random seed (default: 42)
```

### Submission Parameters
```bash
--n_cells           # Cells for submission (default: 100)
--submission_batch_size  # Batch size for inference (default: 10)
```

## Example Commands

### Run with more cells
```bash
nextflow run main.nf -profile local_gpu \
  --max_cells 5000 \
  --chunk_cells 500
```

### Run with custom config
```bash
nextflow run main.nf -profile local_gpu \
  --config configs/high_mmd.yaml
```

### Resume failed run
```bash
nextflow run main.nf -profile local_gpu -resume
```

### Run specific processes
```bash
# Only extract embeddings (modify workflow section)
nextflow run main.nf -profile local_gpu \
  -entry extract_embeddings_only
```

## Monitoring

### Real-time progress
```bash
# Terminal output shows progress automatically
# Ctrl+C to stop (safely resumes with -resume)
```

### Check reports
After completion, check:
- `experiments/reports/nextflow_report.html` - Execution report
- `experiments/reports/nextflow_timeline.html` - Timeline visualization
- `experiments/reports/nextflow_trace.txt` - Resource usage
- `experiments/reports/nextflow_dag.svg` - Workflow DAG

### Check logs
```bash
# Check work directory for process logs
ls -la work/

# View specific process log
cat work/XX/XXXXXXXXXX/.command.log
```

## Output Structure

```
data/
  processed/
    gene_embeddings.pt         # Extracted embeddings
  kg/
    knowledge_graph.gpickle    # Built KG
    
checkpoints/
  best.ckpt                    # Best model checkpoint
  
submissions/
  expression_matrix.csv        # Final predictions
  program_proportions.csv      # Program proportions
  validation_report.txt        # Validation results
  
experiments/
  logs/
    baseline_run.log          # Training log
    inference.log             # Inference log
  reports/
    nextflow_*.html           # Pipeline reports
```

## Troubleshooting

### Out of Memory
```bash
# Reduce cells per chunk
nextflow run main.nf -profile local_gpu \
  --chunk_cells 100 \
  --batch_size 1
```

### GPU Not Available
```bash
# Use CPU profile (slower)
nextflow run main.nf -profile standard
```

### Process Failed
```bash
# Resume from last successful step
nextflow run main.nf -profile local_gpu -resume

# Check specific process log
cat work/XX/XXXXXXXXXX/.command.log
```

### Clean and Restart
```bash
# Remove all intermediate files
nextflow clean -f

# Remove work directory
rm -rf work/

# Run fresh
nextflow run main.nf -profile local_gpu
```

## Advanced Usage

### Modify Workflow
Edit `main.nf` to customize:
- Add new processes
- Change process dependencies
- Adjust resource requirements
- Add conditional execution

### Add Custom Container
Uncomment in `nextflow.config`:
```groovy
docker {
    enabled = true
    container = 'your-container:latest'
}
```

### Run on Cloud
Configure AWS/GCP credentials and use:
```bash
nextflow run main.nf -profile cloud \
  -bucket-dir s3://your-bucket/results
```

## Tips

1. **Use `-resume`**: Always use when re-running to skip completed steps
2. **Check reports**: HTML reports are very informative
3. **Monitor resources**: Use `htop` or `nvidia-smi` during execution
4. **Test small first**: Use `--max_cells 100` for quick testing
5. **Profile wisely**: Choose the right profile for your environment

## Support

For issues:
1. Check `work/` directory logs
2. Review Nextflow reports in `experiments/reports/`
3. Consult [Nextflow documentation](https://www.nextflow.io/docs/latest/)
4. Check [Camber documentation](https://docs.cambercloud.com/) for cloud execution
