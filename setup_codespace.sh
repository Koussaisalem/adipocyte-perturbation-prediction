#!/bin/bash
# Quick setup script for Codespace
# Run this after uploading data files

set -e

echo "🚀 Setting up Adipocyte Perturbation Prediction environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the project root"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[dev,notebooks]" --quiet

# Check data files
echo ""
echo "📁 Checking data files..."
DATA_DIR="data/raw"

files=(
    "obesity_challenge_1.h5ad"
    "signature_genes.csv"
    "program_proportion.csv"
    "program_proportion_local_gtruth.csv"
    "predict_perturbations.txt"
    "gene_to_predict.txt"
)

missing_files=()
for file in "${files[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        size=$(du -h "$DATA_DIR/$file" | cut -f1)
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file - MISSING"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  Missing files: ${missing_files[*]}"
    echo "Please upload them to data/raw/ using VS Code UI:"
    echo "  Right-click on data/raw/ → Upload..."
    exit 1
fi

# Create combined gene list for KG building
echo ""
echo "🧬 Creating combined gene list..."
cat data/raw/predict_perturbations.txt data/raw/gene_to_predict.txt | sort -u > data/processed/all_genes.txt
echo "  ✅ Created data/processed/all_genes.txt ($(wc -l < data/processed/all_genes.txt) genes)"

echo ""
echo "✅ Setup complete! Next steps:"
echo ""
echo "1. Build knowledge graph:"
echo "   python scripts/build_kg.py --gene-list data/processed/all_genes.txt"
echo ""
echo "2. Extract embeddings (requires GPU):"
echo "   python scripts/extract_embeddings.py --h5ad-file data/raw/obesity_challenge_1.h5ad"
echo ""
echo "3. Start training:"
echo "   python scripts/train.py"
echo ""
