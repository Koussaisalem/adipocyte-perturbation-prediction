# Path Management Utilities

This module provides centralized path management for the adipocyte perturbation prediction project, ensuring consistent paths across different machines and environments.

## Usage

### In Python Scripts

```python
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR

# Use path constants directly
h5ad_file = RAW_DATA_DIR / "obesity_challenge_1.h5ad"
output_file = PROCESSED_DIR / "gene_embeddings.pt"
model_path = MODELS_DIR / "geneformer_full"
```

### In Jupyter Notebooks

The notebook should start with the path setup cell (already included in `get_started.ipynb`):

```python
# Setup: Configure project paths (run first!)
import sys
import os
from pathlib import Path

# Auto-detect project root
notebook_dir = Path.cwd()
if notebook_dir.name == "notebooks":
    project_root = notebook_dir.parent
# ... (full setup code in notebook)

# Import path constants
from utils.paths import (
    RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, 
    GENEFORMER_MODEL_DIR, CHECKPOINTS_DIR, SUBMISSIONS_DIR
)
```

### In Shell Commands

Use Python f-strings with path variables:

```python
!ls -lh {RAW_DATA_DIR}
!python scripts/train.py --config {CONFIGS_DIR}/default.yaml
```

## Available Path Constants

| Constant | Description | Default Value |
|----------|-------------|---------------|
| `PROJECT_ROOT` | Project root directory | Auto-detected |
| `DATA_DIR` | Main data directory | `{PROJECT_ROOT}/data` |
| `RAW_DATA_DIR` | Raw challenge data | `{DATA_DIR}/raw/Challenge` |
| `PROCESSED_DIR` | Processed data | `{DATA_DIR}/processed` |
| `KG_DIR` | Knowledge graph data | `{DATA_DIR}/kg` |
| `MODELS_DIR` | Model directory | `{PROJECT_ROOT}/models` |
| `GENEFORMER_MODEL_DIR` | Geneformer model | `{MODELS_DIR}/geneformer_full` |
| `CONFIGS_DIR` | Configuration files | `{PROJECT_ROOT}/configs` |
| `CHECKPOINTS_DIR` | Training checkpoints | `{PROJECT_ROOT}/checkpoints` |
| `SUBMISSIONS_DIR` | Submission outputs | `{PROJECT_ROOT}/submissions` |
| `EXPERIMENTS_DIR` | Experiment logs | `{PROJECT_ROOT}/experiments` |

## Environment Variable Overrides

You can override paths using environment variables:

```bash
# Override data directory location
export APP_DATA_DIR=/mnt/external/data

# Override models directory
export APP_MODELS_DIR=/mnt/models
```

Then use the getter functions:

```python
from src.utils.paths import get_data_dir, get_models_dir

data_dir = get_data_dir()  # Returns override if set, otherwise DATA_DIR
models_dir = get_models_dir()  # Returns override if set, otherwise MODELS_DIR
```

## Testing Paths

To verify all paths are correctly configured:

```bash
python -m src.utils.paths
```

This will print all configured paths.

## Benefits

✅ **Machine-independent**: Works on any machine without hardcoded absolute paths  
✅ **Environment-flexible**: Can override paths via environment variables  
✅ **IDE-friendly**: Autocomplete and type hints for all paths  
✅ **Single source of truth**: All paths defined in one place  
✅ **Easy to maintain**: Change paths in one location  

## Project Structure

The path module expects this structure:

```
adipocyte-perturbation-prediction/
├── pyproject.toml              # Project root marker
├── data/
│   ├── raw/Challenge/          # RAW_DATA_DIR
│   ├── processed/              # PROCESSED_DIR
│   └── kg/                     # KG_DIR
├── models/
│   └── geneformer_full/        # GENEFORMER_MODEL_DIR
├── configs/                    # CONFIGS_DIR
├── checkpoints/                # CHECKPOINTS_DIR
├── submissions/                # SUBMISSIONS_DIR
├── experiments/                # EXPERIMENTS_DIR
└── src/
    └── utils/
        └── paths.py            # This module
```

## Troubleshooting

### "Could not find project root" Error

The path module looks for marker files (`pyproject.toml`, `.git`, `setup.py`, `README.md`). Ensure one of these exists in your project root.

### Paths Not Resolving

Make sure you've run the path setup cell in notebooks or imported the module in scripts:

```python
# Always add this at the top of scripts
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.paths import ...
```
