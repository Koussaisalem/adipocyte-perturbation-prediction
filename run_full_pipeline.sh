#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_full_pipeline.sh [options]

Options:
  --config PATH            Config file (default: configs/default.yaml)
  --data-root PATH         Data root containing raw/ (default: data)
  --h5ad PATH              H5AD file (default: <data-root>/raw/obesity_challenge_1.h5ad)
  --batch-size N           Batch size for submission generation (default: 10)
  --n-cells N              Number of cells per perturbation (default: 100)
  --checkpoint PATH        Checkpoint to use for submission (default: checkpoints/best.ckpt)
  --skip-setup             Skip running setup_codespace.sh
  --skip-kg                Skip knowledge graph build
  --skip-embeddings         Skip embeddings extraction
  --skip-train             Skip training
  --skip-submit            Skip submission generation
  --dry-run                Print commands without executing them
  -h, --help               Show this help
EOF
}

CONFIG="configs/default.yaml"
DATA_ROOT="data"
H5AD=""
BATCH_SIZE=10
N_CELLS=100
CHECKPOINT="checkpoints/best.ckpt"
SKIP_SETUP=0
SKIP_KG=0
SKIP_EMB=0
SKIP_TRAIN=0
SKIP_SUBMIT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --h5ad) H5AD="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --n-cells) N_CELLS="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --skip-setup) SKIP_SETUP=1; shift;;
    --skip-kg) SKIP_KG=1; shift;;
    --skip-embeddings) SKIP_EMB=1; shift;;
    --skip-train) SKIP_TRAIN=1; shift;;
    --skip-submit) SKIP_SUBMIT=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
 done

if [[ -z "$H5AD" ]]; then
  H5AD="$DATA_ROOT/raw/obesity_challenge_1.h5ad"
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$@"
  fi
}

# Ensure repo root
if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: Run this script from the project root." >&2
  exit 1
fi

# Prefer local venv if present
if [[ -f "/home/koussai/Challenge/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "/home/koussai/Challenge/.venv/bin/activate"
fi

run_cmd "mkdir -p $DATA_ROOT/raw $DATA_ROOT/processed $DATA_ROOT/kg checkpoints submissions experiments/logs"

if [[ "$SKIP_SETUP" -eq 0 ]]; then
  run_cmd "./setup_codespace.sh"
fi

# Prepare gene list for KG if missing
if [[ ! -f "$DATA_ROOT/processed/all_genes.txt" ]]; then
  run_cmd "cat $DATA_ROOT/raw/predict_perturbations.txt $DATA_ROOT/raw/gene_to_predict.txt | sort -u > $DATA_ROOT/processed/all_genes.txt"
fi

if [[ "$SKIP_KG" -eq 0 ]]; then
  run_cmd "python scripts/build_kg.py --gene-list $DATA_ROOT/processed/all_genes.txt"
fi

if [[ "$SKIP_EMB" -eq 0 ]]; then
  run_cmd "python scripts/extract_embeddings.py --h5ad-file $H5AD"
fi

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  run_cmd "python scripts/train.py --config $CONFIG 2>&1 | tee experiments/logs/train_$(date +%Y%m%d_%H%M%S).log"
fi

if [[ "$SKIP_SUBMIT" -eq 0 ]]; then
  run_cmd "python scripts/generate_submission.py --config $CONFIG --checkpoint $CHECKPOINT --output-dir submissions --n-cells $N_CELLS --batch-size $BATCH_SIZE 2>&1 | tee experiments/logs/submit_$(date +%Y%m%d_%H%M%S).log"
fi

echo "✅ Pipeline complete."
