git clone https://github.com/koussai/adipocyte-perturbation-prediction.git
# Zero-Shot Adipocyte Perturbation Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict single-cell transcriptomic responses of human adipocytes to 2,863 unseen gene perturbations using a Knowledge-Graph Conditioned Flow Matching model.

---

## Overview
- **Problem**: Zero-shot prediction of TF perturbations on adipocyte differentiation. Training set: 124 measured perturbations. Test set: 2,863 held-out gene targets.
- **Outputs**: 100 generated cells × ~10k genes per perturbation plus cell program proportions (pre_adipo, adipo, lipo, other).
- **Core components**:
  1. **Knowledge graph**: CollecTRI/DoRothEA + STRING PPI + GO edges.
  2. **GATv2 encoder**: Perturbation embeddings from the graph.
  3. **Flow matching decoder**: ODE-based cell distribution generator.
  4. **Proportion head**: Predicts program proportions alongside expression.

## Repository Layout
```
configs/               # Experiment configs (default.yaml, variants)
data/
  raw/Challenge/       # Challenge inputs (h5ad, gene lists, proportions)
  processed/           # Derived data (gene_embeddings.pt, all_genes.txt)
  kg/                  # knowledge_graph.gpickle
scripts/               # CLI entry points (build_kg, extract_embeddings, train, generate_submission)
src/                   # Library code: data, models, losses, training, inference
checkpoints/           # Saved model weights
experiments/logs/      # Training and inference logs
submissions/           # Generated competition submissions
```

## Environment and Requirements
- Python 3.10+, CUDA-capable GPU recommended for embeddings and training.
- Disk: at least 50–100 GB free for h5ad, model weights, intermediates.
- Install (core runtime): `pip install -r requirements.txt`
- Install (dev + notebooks): `pip install -e ".[dev,notebooks]"`

### PyTorch and Torch Geometric
Install the correct PyTorch build for your system before running the pipeline:
- **CPU**: `pip install --index-url https://download.pytorch.org/whl/cpu torch`
- **CUDA**: follow https://pytorch.org/get-started/locally/ and then `pip install torch-geometric`

### Pretrained Models
- **Geneformer** (required for real embeddings):
  - Package: `pip install git+https://huggingface.co/ctheodoris/Geneformer.git`
  - Model weights are downloaded automatically to `models/geneformer/` on first run of
    [scripts/extract_embeddings.py](scripts/extract_embeddings.py).
  - If you want to skip this step, use `--random` in the embeddings script for placeholder embeddings.

## Data Preparation
1) Place challenge files in `data/raw/Challenge/`:
   - obesity_challenge_1.h5ad
   - signature_genes.csv
   - program_proportion.csv
   - program_proportion_local_gtruth.csv
   - predict_perturbations.txt
   - gene_to_predict.txt
2) Generate combined gene list and verify files:
   ```bash
   bash setup_codespace.sh
   ```
   This also creates `data/processed/all_genes.txt` and required directories.

## Knowledge Graph Construction
Build a KG with CollecTRI/DoRothEA (levels A/B) and STRING (score >= 700):
```bash
python scripts/build_kg.py \
  --gene-list data/processed/all_genes.txt \
  --output data/kg/knowledge_graph.gpickle \
  --dorothea-levels A B \
  --string-threshold 700
```
Expected scale: ~11k nodes, 50k edges (directional, includes tf_activates, tf_represses, interacts_with).

## Geneformer Embedding Extraction
Use chunked tokenization to control memory. Run on a GPU VM with ample RAM/disk.
```bash
python scripts/extract_embeddings.py \
  --h5ad-file data/raw/Challenge/obesity_challenge_1.h5ad \
  --max-cells 20000 \
  --chunk-cells 1000 \
  --batch-size 8 \
  --output data/processed/gene_embeddings.pt
```
Flags:
- `--max-cells`: global cap on cells processed (increase if resources allow).
- `--chunk-cells`: cells per tokenization chunk (lower if OOM; raise for throughput).
- `--batch-size`: forward batch for embedding extraction.

## Training
Baseline training uses settings in [configs/default.yaml](configs/default.yaml) and code in [src/training/trainer.py](src/training/trainer.py), [src/models/full_model.py](src/models/full_model.py).
```bash
python scripts/train.py \
  --config configs/default.yaml \
  --seed 42 \
  2>&1 | tee experiments/logs/baseline_run.log
```
Notes:
- AdamW, lr=1e-4, cosine warmup, epochs=100, batch_size=64, precision=16-mixed by default.
- Early stopping monitors val/mmd; checkpointing keeps top-3 by val/mmd.
- Adjust `training.batch_size` and `training.accumulate_grad_batches` if GPU memory is tight.

### Suggested Experiment Variants
- Higher MMD weight: duplicate `configs/default.yaml` to `configs/high_mmd.yaml` with `losses.mmd_weight: 0.2`, `losses.pearson_weight: 0.1`.
- Deeper encoder: `gat_layers: 4`, `gat_heads: 16`, `gat_hidden_dim: 256` (reduce batch_size if needed).
- Higher PCA: `flow_matching.pca_components: 750` for richer reconstruction; monitor VRAM.

## Inference and Submission
Generate submission from the best checkpoint (lowest val/mmd):
```bash
python scripts/generate_submission.py \
  --checkpoint checkpoints/best.ckpt \
  --output-dir submissions \
  --n-cells 100 \
  --batch-size 10 \
  2>&1 | tee experiments/logs/inference.log
```
Validation checks:
- Expression matrix rows: `wc -l submissions/expression_matrix.csv` should be 286,301 (including header).
- Program proportions format: `head submissions/program_proportions.csv`.
- NaN check: `python - <<'PY'
import pandas as pd
df = pd.read_csv('submissions/expression_matrix.csv')
print('NaNs:', df.isna().sum().sum())
PY`

## Metrics
- **MMD (Maximum Mean Discrepancy)**: distribution alignment of generated vs. observed cells.
- **Pearson Delta**: correlation of expression shifts relative to control.
- **Proportion MAE**: accuracy of predicted program proportions.

## Troubleshooting
- GPU OOM: lower `training.batch_size`, increase `training.accumulate_grad_batches`, or reduce `flow_matching.pca_components`.
- Embedding extraction OOM: lower `--chunk-cells` and `--batch-size`; ensure sufficient disk (>50 GB).
- STRING API timeouts: pre-download STRING links and pass `--string-links`/`--string-info` to `build_kg.py`.

## Citation
```bibtex
@software{adipocyte_perturbation_2026,
  title = {Zero-Shot Adipocyte Perturbation Prediction},
  author = {Koussai},
  year = {2026},
  url = {https://github.com/koussai/adipocyte-perturbation-prediction}
}
```

## License
MIT License. See [LICENSE](LICENSE).
