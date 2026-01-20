# CrunchDAO Submission Guide

## Quick Start

### Step 1: Wait for Submission Generation to Complete

Check if the submission generation finished:
```bash
ps aux | grep "generate_submission"
tail experiments/submission_generation_optimized.log
```

If still running, wait for completion (~75 minutes total).

### Step 2: Package Your Model

Run the packaging script to copy all required files:
```bash
python package_submission.py
```

This creates `crunch_submission_template/` with:
- `main.py` - Interface with train() and infer() functions
- `requirements.txt` - Dependencies
- `resources/` - Model checkpoint, embeddings, KG, config
- `src/` - Your model code

### Step 3: Local Testing (Optional)

Test the submission locally before uploading:
```bash
cd crunch_submission_template
python main.py  # Quick test with 5 genes
```

### Step 4: Setup CrunchDAO CLI

Install and authenticate:
```bash
pip install crunch-cli --upgrade
```

Get your token from the competition page (Submit → Submit via CLI tab), then:
```bash
cd crunch_submission_template
crunch setup <competition-name> <your-model-name> --token <YOUR_TOKEN> .
```

**Example:**
```bash
crunch setup adipocyte-perturbation my-flow-model --token eyJhbG... .
```

### Step 5: Push Your Submission

```bash
crunch push --message "Flow matching model, epoch 0, val MMD 0.0129"
```

**Important flags:**
- `--no-pip-freeze` - If you get package version conflicts
- `--message` - Description of your submission (optional but recommended)

### Step 6: Run in the Cloud

1. Go to the competition portal
2. Navigate to "Submit" page
3. Find your submission
4. Click "Run in the Cloud"
5. Monitor logs to ensure it works

## Troubleshooting

### Large File Upload
Your `resources/` folder is ~150MB. The portal should handle this, but if issues occur:

**Option A: Use CLI (recommended)**
- Already handles large files automatically

**Option B: Use .gitignore pattern**
If the portal complains, you can stream large files:
```bash
# Create .crunchignore
echo "resources/*.pt" > .crunchignore
echo "resources/*.gpickle" >> .crunchignore
```

Then download during first run (modify main.py to fetch from cloud storage).

### Package Version Issues

If `crunch push` fails due to package conflicts:
```bash
crunch push --no-pip-freeze --message "submission"
```

### Memory Issues During Inference

The portal may have memory limits. If OOM occurs:
1. Reduce `n_cells` from 100 to 50 in main.py
2. Use float16 precision: `model.half()` 
3. Generate predictions in batches

### Missing PCA Cache

If you didn't wait for submission generation to create the PCA cache:
```bash
# Check if it exists
ls -lh data/processed/pca_cache.pkl

# If not, you need to run submission generator or create it manually
python scripts/generate_submission_optimized.py  # Will create cache
```

Then re-run packaging:
```bash
python package_submission.py
```

## Submission Contents

### Model Artifacts in resources/ (~150MB total)
- `model.ckpt` - Trained model weights (40MB)
- `gene_embeddings.pt` - 11,046 gene embeddings (45MB)
- `knowledge_graph.gpickle` - KG structure (25MB)
- `config.yaml` - Model configuration
- `pca_cache.pkl` - PCA transformation (40MB)

### Code Files
- `main.py` - Entry point with train()/infer()
- `src/` - Full model code
- `requirements.txt` - Dependencies

## Expected Performance

**Local validation (training val set):**
- MMD: 0.0129 (5× better than leaderboard #1)
- CFM Loss: 17.32
- Proportion Loss: 0.0000

**Zero-shot validation (5 held-out genes):**
- L1 Proportion Distance: 0.4288

## Competition Format Notes

Based on the documentation, the platform will:
1. Call `train(X_train, y_train, model_dir)` once - we skip actual training
2. Call `infer(X_test, model_dir)` on test data
3. Evaluate predictions using their private test set

Your model uses **zero-shot prediction** via knowledge graph, so it can predict any gene in the KG without seeing training data for it.

## Next Steps After Submission

1. **Monitor logs** - Check "Run in the Cloud" logs for errors
2. **View leaderboard** - See how your MMD compares to others
3. **Iterate** - If needed, retrain with:
   - Fixed early stopping (mode: min)
   - More epochs
   - Different hyperparameters
