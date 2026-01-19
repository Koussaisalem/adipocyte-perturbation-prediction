# Path Management Implementation Summary

## ✅ What Was Implemented

### 1. **Centralized Path Management Module** (`src/utils/paths.py`)

Created a comprehensive path management system that:
- Auto-detects project root by looking for marker files (`.git`, `pyproject.toml`, etc.)
- Defines all project paths as constants in one place
- Supports environment variable overrides for flexibility
- Works across different machines without hardcoded paths

**Key Features:**
```python
# All paths computed relative to auto-detected project root
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Challenge"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
GENEFORMER_MODEL_DIR = MODELS_DIR / "geneformer_full"
# ... and more
```

### 2. **Notebook Setup Cells**

Added two setup cells at the top of `get_started.ipynb`:

**Cell 1: Path Configuration** (runs first)
- Auto-detects project root from notebook location
- Changes working directory to project root
- Adds `src/` to Python path
- Imports all path constants
- Prints configuration for verification

**Cell 2: Path Verification** (optional)
- Displays all configured paths
- Helps users verify setup

### 3. **Updated All Hardcoded Paths**

Systematically replaced hardcoded paths throughout the notebook:

| Old (Hardcoded) | New (Path Constant) |
|-----------------|---------------------|
| `data/raw/Challenge` | `{RAW_DATA_DIR}` |
| `data/processed/gene_embeddings.pt` | `{PROCESSED_DIR}/gene_embeddings.pt` |
| `models/geneformer_full` | `{GENEFORMER_MODEL_DIR}` |
| `configs/default.yaml` | `{CONFIGS_DIR}/default.yaml` |
| `checkpoints/best.ckpt` | `{CHECKPOINTS_DIR}/best.ckpt` |
| `submissions/` | `{SUBMISSIONS_DIR}/` |
| `experiments/logs/` | `{EXPERIMENTS_DIR}/logs/` |

### 4. **Updated Python Scripts**

Modified `scripts/extract_embeddings.py` to:
- Import centralized path constants
- Use path constants for default arguments
- Fallback gracefully if paths module unavailable

### 5. **Documentation**

Created comprehensive documentation:
- `src/utils/README.md` - Full usage guide with examples
- Inline docstrings in `paths.py`
- Comments in notebook cells

## 🎯 Benefits

### ✅ Cross-Platform Compatibility
- Works on Windows, Linux, macOS
- No more `/home/koussai/...` absolute paths
- Uses `pathlib.Path` for proper path handling

### ✅ Environment Flexibility
- Override paths via environment variables when needed
- Example: `export APP_DATA_DIR=/mnt/external/data`

### ✅ Developer Experience
- IDE autocomplete for all paths
- Type hints and documentation
- Single place to update paths

### ✅ "Works on My Machine" → "Works Everywhere"
- Another developer can clone the repo
- Run notebooks without changing paths
- Paths automatically adapt to their environment

## 📝 Usage Examples

### In Notebooks (after setup cell)
```python
# Access raw data
h5ad_file = RAW_DATA_DIR / "obesity_challenge_1.h5ad"

# Save processed data
torch.save(embeddings, PROCESSED_DIR / "embeddings.pt")

# Shell commands with path variables
!ls -lh {RAW_DATA_DIR}
!python scripts/train.py --config {CONFIGS_DIR}/default.yaml
```

### In Python Scripts
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR

# Use paths
input_file = RAW_DATA_DIR / "data.h5ad"
output_file = PROCESSED_DIR / "results.pt"
```

### With Environment Variables
```bash
# On a different machine with data elsewhere
export APP_DATA_DIR=/mnt/bigdrive/project_data
export APP_MODELS_DIR=/mnt/models

# Paths will use these overrides automatically
python scripts/train.py
```

## 🧪 Testing

Verified the implementation:
```bash
# Test paths module loads correctly
python3 -c 'import sys; sys.path.insert(0, "src"); from utils.paths import PROJECT_ROOT; print(PROJECT_ROOT)'
# Output: /home/koussai/Challenge/adipocyte-perturbation-prediction
```

## 📦 Files Created/Modified

### Created:
- `src/utils/__init__.py` - Package initialization
- `src/utils/paths.py` - Path management module (140 lines)
- `src/utils/README.md` - Documentation

### Modified:
- `notebooks/get_started.ipynb` - Added setup cells, updated all path references
- `scripts/extract_embeddings.py` - Added path imports and usage

## 🚀 Next Steps for Users

1. **Run the notebook setup cell first** (Cell 0)
   - This configures all paths automatically

2. **Use path variables in shell commands**
   - Use `{RAW_DATA_DIR}` instead of `data/raw/Challenge`
   - Use `{MODELS_DIR}` instead of `models/`

3. **If sharing code with teammates**
   - They only need to run the setup cell
   - All paths will work automatically on their machine

4. **If data is in a different location**
   - Use environment variables: `export APP_DATA_DIR=/path/to/data`
   - Or create a symlink: `ln -s /path/to/data data/raw/Challenge`

## 🔧 Troubleshooting

### Issue: "Could not find project root"
**Solution:** Ensure you have one of these files in project root:
- `pyproject.toml` ✓ (you have this)
- `.git/` ✓ (you have this)
- `README.md` ✓ (you have this)

### Issue: Module not found
**Solution:** Make sure setup cell ran successfully and added `src/` to Python path

### Issue: Paths pointing to wrong location
**Solution:** Check you're running from project root or run the setup cell again

## 📊 Impact

**Before:** 
- Hardcoded paths: `data/raw/Challenge/file.h5ad`
- Breaks on different machines
- Need to manually edit paths

**After:**
- Dynamic paths: `RAW_DATA_DIR / "file.h5ad"`
- Works on any machine
- Zero manual configuration needed

This implementation follows best practices from professional Python projects and ensures your code will run smoothly across different development environments! 🎉
