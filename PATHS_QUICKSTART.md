# Quick Start: Path Management

## For New Developers

When you clone this repo, **everything will work automatically** - no path configuration needed! 🎉

### Step 1: Clone the Repo
```bash
git clone https://github.com/Koussaisalem/adipocyte-perturbation-prediction.git
cd adipocyte-perturbation-prediction
```

### Step 2: Open the Notebook
```bash
jupyter notebook notebooks/get_started.ipynb
```

### Step 3: Run the First Cell (Path Setup)
The very first cell automatically:
- Detects your project location
- Configures all paths
- No manual editing required!

That's it! The notebook will work on your machine. ✅

## Common Questions

### Q: Do I need to change paths?
**A: No!** All paths are auto-detected and configured.

### Q: What if my data is elsewhere?
**A:** Two options:
1. **Symlink** (recommended):
   ```bash
   ln -s /your/data/location data/raw/Challenge
   ```
2. **Environment variable**:
   ```bash
   export APP_DATA_DIR=/your/data/location
   ```

### Q: How do I use paths in my own scripts?
**A:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DIR

# Use paths
data = load_data(RAW_DATA_DIR / "my_file.h5ad")
```

### Q: Can I see all configured paths?
**A:** Yes! Run the second cell in the notebook or:
```bash
python -m src.utils.paths
```

## Available Path Variables

Use these in your notebooks after running the setup cell:

```python
RAW_DATA_DIR           # data/raw/Challenge/
PROCESSED_DIR          # data/processed/
MODELS_DIR             # models/
GENEFORMER_MODEL_DIR   # models/geneformer_full/
CONFIGS_DIR            # configs/
CHECKPOINTS_DIR        # checkpoints/
SUBMISSIONS_DIR        # submissions/
EXPERIMENTS_DIR        # experiments/
```

## Example Usage

### In Notebook Cells (Shell Commands)
```python
!ls {RAW_DATA_DIR}
!python scripts/train.py --config {CONFIGS_DIR}/default.yaml
```

### In Python Code
```python
# Load data
h5ad_file = RAW_DATA_DIR / "obesity_challenge_1.h5ad"
adata = sc.read_h5ad(h5ad_file)

# Save results
output = PROCESSED_DIR / "results.pt"
torch.save(embeddings, output)
```

## For More Details

- **Full documentation**: See [src/utils/README.md](src/utils/README.md)
- **Implementation details**: See [PATH_MANAGEMENT.md](PATH_MANAGEMENT.md)

---

**Bottom line**: Clone, run the notebook, it works. No "this doesn't work on my machine" issues! 🚀
