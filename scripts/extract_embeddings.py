#!/usr/bin/env python3
"""
Extract gene embeddings using Geneformer with chunked processing for large h5ad files.
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_geneformer():
    try:
        from geneformer import TranscriptomeTokenizer, EmbExtractor
    except ImportError:
        logger.error("Geneformer not installed. Install with: pip install git+https://huggingface.co/ctheodoris/Geneformer.git")
        sys.exit(1)
    return TranscriptomeTokenizer, EmbExtractor


def download_model(model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    logger.info("Downloading Geneformer model from HuggingFace...")
    cache_dir = snapshot_download(
        repo_id="ctheodoris/Geneformer",
        allow_patterns=["gf-*/**", "*.pkl"],
        local_dir="models/geneformer",
    )
    candidate = Path(cache_dir) / "gf-12L-104M-i4096"
    final_dir = candidate if candidate.exists() else Path(cache_dir)
    logger.info(f"Model downloaded to {final_dir}")
    return str(final_dir)


def clean_dir(path: Path):
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def main():
    parser = argparse.ArgumentParser(description="Extract Geneformer gene embeddings with chunking")
    parser.add_argument("--h5ad-file", required=True, help="Path to h5ad file")
    parser.add_argument("--model-dir", default=None, help="Path to Geneformer model directory")
    parser.add_argument("--output", default="data/processed/gene_embeddings.pt", help="Path to save embeddings")
    parser.add_argument("--gene-list", default=None, help="Optional file with genes to keep in output")
    parser.add_argument("--batch-size", type=int, default=8, help="Forward batch size")
    parser.add_argument("--max-cells", type=int, default=2000, help="Total cells to process (global cap)")
    parser.add_argument("--chunk-cells", type=int, default=800, help="Cells per chunk for tokenization")
    parser.add_argument("--emb-layer", type=int, choices=[-1, 0], default=-1, help="Embedding layer: -1 (2nd last) or 0 (last)")
    parser.add_argument("--random", action="store_true", help="Use random embeddings (testing only)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Random mode for quick testing
    if args.random:
        import scanpy as sc
        adata = sc.read_h5ad(args.h5ad_file, backed="r")
        genes = list(adata.var_names)
        embedding_dim = 512
        embeddings = {g: torch.randn(embedding_dim) for g in genes}
        torch.save(embeddings, output_path)
        logger.info(f"Saved {len(embeddings)} random embeddings to {output_path}")
        return

    TranscriptomeTokenizer, EmbExtractor = load_geneformer()
    model_dir = download_model(args.model_dir)

    import scanpy as sc

    # Open backed to avoid loading whole file
    adata = sc.read_h5ad(args.h5ad_file, backed="r")
    n_cells = adata.n_obs
    logger.info(f"Backed dataset: {n_cells} cells x {adata.n_vars} genes")

    # Determine total cells to process
    total_cells = min(args.max_cells, n_cells)
    chunk_size = min(args.chunk_cells, total_cells)
    if chunk_size <= 0:
        logger.error("Chunk size must be positive")
        sys.exit(1)

    agg_sum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    processed = 0
    chunk_idx = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_dir = tmp_path / "data"
        token_dir = tmp_path / "tokenized"
        emb_dir = tmp_path / "embeddings"
        data_dir.mkdir()
        token_dir.mkdir()
        emb_dir.mkdir()

        tokenizer = TranscriptomeTokenizer(nproc=1, model_input_size=4096, special_token=True, model_version="V2")

        while processed < total_cells:
            start = processed
            end = min(processed + chunk_size, total_cells)
            logger.info(f"Processing chunk {chunk_idx}: cells {start} to {end} (total {total_cells})")

            # Load slice into memory
            slice_indices = list(range(start, end))
            chunk_adata = adata[slice_indices, :].to_memory()
            chunk_path = data_dir / f"chunk_{chunk_idx}.h5ad"
            chunk_adata.write_h5ad(chunk_path)
            del chunk_adata

            # Tokenize chunk
            clean_dir(token_dir)
            tokenizer.tokenize_data(
                data_directory=data_dir,
                output_directory=token_dir,
                output_prefix=f"tokenized_{chunk_idx}",
                file_format="h5ad",
                input_identifier=chunk_path.name,
            )

            tokenized_files = list(token_dir.glob(f"tokenized_{chunk_idx}*.dataset"))
            if not tokenized_files:
                logger.error("No tokenized dataset produced for chunk")
                sys.exit(1)
            tokenized_path = tokenized_files[0]

            # Extract embeddings
            clean_dir(emb_dir)
            emb_extractor = EmbExtractor(
                model_type="Pretrained",
                emb_mode="gene",
                gene_emb_style="mean_pool",
                max_ncells=None,
                emb_layer=args.emb_layer,
                forward_batch_size=args.batch_size,
                nproc=1,
                model_version="V2",
            )

            emb_extractor.extract_embs(
                model_directory=model_dir,
                input_data_file=tokenized_path,
                output_directory=emb_dir,
                output_prefix=f"gene_embs_{chunk_idx}",
                output_torch_embs=True,
            )

            # Load embeddings from torch if available, else CSV
            embeddings_chunk: dict[str, torch.Tensor] = {}
            torch_files = list(emb_dir.glob("*.pt"))
            if torch_files:
                data_pt = torch.load(torch_files[0])
                if isinstance(data_pt, dict):
                    embeddings_chunk = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in data_pt.items()}
            if not embeddings_chunk:
                csv_files = list(emb_dir.glob("*.csv"))
                if csv_files:
                    emb_df = pd.read_csv(csv_files[0], index_col=0)
                    embeddings_chunk = {g: torch.tensor(emb_df.loc[g].values, dtype=torch.float32) for g in emb_df.index}
            if not embeddings_chunk:
                logger.error("Failed to load embeddings for chunk")
                sys.exit(1)

            # Aggregate mean across chunks
            for gene, vec in embeddings_chunk.items():
                if gene not in agg_sum:
                    agg_sum[gene] = vec.clone().float()
                    counts[gene] = 1
                else:
                    agg_sum[gene] += vec.float()
                    counts[gene] += 1

            processed = end
            chunk_idx += 1
            clean_dir(data_dir)  # remove chunk file to save space

    adata.file.close()

    # Finalize embeddings (mean across chunks)
    final_embeddings = {g: agg_sum[g] / counts[g] for g in agg_sum}

    # Filter to gene list if provided
    if args.gene_list:
        with open(args.gene_list) as f:
            target = set(line.strip() for line in f if line.strip())
        final_embeddings = {g: v for g, v in final_embeddings.items() if g in target}
        logger.info(f"Filtered to {len(final_embeddings)} genes from gene_list")

    torch.save(final_embeddings, output_path)
    emb_dim = next(iter(final_embeddings.values())).shape[0] if final_embeddings else 0
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"  {len(final_embeddings)} genes x {emb_dim} dimensions (mean over {counts.get(next(iter(counts)), 0)} chunk(s))")


if __name__ == "__main__":
    main()
