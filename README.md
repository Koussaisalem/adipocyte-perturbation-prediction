# Zero-Shot Adipocyte Perturbation Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Broad Obesity Challenge 1**: Predict single-cell transcriptomic responses of human adipocytes to 2,863 held-out gene perturbations using Knowledge-Graph Conditioned Flow Matching.

## 🎯 Challenge Overview

- **Task**: Zero-shot prediction of TF knockouts on adipocyte differentiation
- **Training**: 124 perturbations with measured outcomes
- **Test**: 2,863 unseen gene perturbations
- **Output**: 100 cells × 10,238 genes per perturbation + cell program proportions

## 🧬 Approach

We use a **Knowledge-Graph Conditioned Flow Matching** model:

1. **Knowledge Graph**: TF-target networks (DoRothEA/CollecTRI) + PPI (STRING) + GO terms
2. **GATv2 Encoder**: Learns perturbation embeddings from graph structure
3. **Flow Matching Decoder**: Generates cell distribution via ODE integration
4. **Multi-Task Head**: Predicts cell program proportions (pre_adipo, adipo, lipo, other)

## 📁 Project Structure

```
adipocyte-perturbation-prediction/
├── configs/                  # Hydra configs
│   └── default.yaml
├── data/
│   ├── raw/                  # Original challenge files
│   ├── processed/            # Preprocessed data
│   └── kg/                   # Knowledge graph
├── src/
│   ├── data/                 # Data loading & KG construction
│   ├── models/               # GATv2, Flow Matching, Proportion Head
│   ├── losses/               # MMD, Pearson Delta
│   ├── training/             # Training loop
│   └── inference/            # Prediction pipeline
├── notebooks/                # EDA & analysis
├── scripts/                  # CLI entry points
└── checkpoints/              # Model weights
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/koussai/adipocyte-perturbation-prediction.git
cd adipocyte-perturbation-prediction

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e ".[dev,notebooks]"
```

### Data Setup

```bash
# Link raw data
ln -s /path/to/challenge/data data/raw/

# Build knowledge graph
python scripts/build_kg.py

# Extract Geneformer embeddings (on GPU machine)
python scripts/extract_embeddings.py --model geneformer-106m
```

### Training

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py model.hidden_dim=256 training.lr=1e-4

# Resume from checkpoint
python scripts/train.py training.resume=checkpoints/last.ckpt
```

### Inference

```bash
# Generate submission
python scripts/generate_submission.py \
    --checkpoint checkpoints/best.ckpt \
    --output submissions/
```

## 📊 Metrics

- **MMD (Maximum Mean Discrepancy)**: Distribution similarity
- **Pearson Delta**: Directional shift correlation
- **Proportion MAE**: Cell state prediction accuracy

## 🔧 Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
model:
  gat_layers: 3
  gat_heads: 8
  hidden_dim: 128
  perturbation_dim: 256
  
flow_matching:
  pca_components: 500
  ode_steps: 20
  
training:
  lr: 1e-4
  batch_size: 64
  epochs: 100
  mmd_weight: 0.1
  proportion_weight: 0.5
```

## 📝 Citation

```bibtex
@software{adipocyte_perturbation_2026,
  title = {Zero-Shot Adipocyte Perturbation Prediction},
  author = {Koussai},
  year = {2026},
  url = {https://github.com/koussai/adipocyte-perturbation-prediction}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
