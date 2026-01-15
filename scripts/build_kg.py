#!/usr/bin/env python3
"""
Build the knowledge graph from biological databases.

Fetches regulatory networks from DoRothEA/CollecTRI and PPI from STRING,
then constructs a heterogeneous graph for perturbation encoding.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.knowledge_graph import build_knowledge_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph")
    parser.add_argument(
        "--gene-list",
        type=str,
        required=True,
        help="Path to gene list file (predict_perturbations.txt + gene_to_predict.txt combined)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kg/knowledge_graph.gpickle",
        help="Path to save knowledge graph",
    )
    parser.add_argument(
        "--dorothea-levels",
        nargs="+",
        default=["A", "B"],
        help="DoRothEA confidence levels to include",
    )
    parser.add_argument(
        "--string-threshold",
        type=int,
        default=700,
        help="STRING combined score threshold",
    )
    parser.add_argument(
        "--string-links",
        type=str,
        default=None,
        help="Path to local STRING links file (9606.protein.links.v12.0.txt.gz)",
    )
    parser.add_argument(
        "--string-info",
        type=str,
        default=None,
        help="Path to local STRING info file (9606.protein.info.v12.0.txt.gz)",
    )
    parser.add_argument(
        "--include-go",
        action="store_true",
        help="Include GO co-functional edges",
    )
    parser.add_argument(
        "--go-terms",
        nargs="+",
        default=["GO:0045444", "GO:0006629", "GO:0045598"],
        help="GO terms to use for co-functional edges",
    )
    
    args = parser.parse_args()
    
    # Load gene list
    logger.info(f"Loading gene list from {args.gene_list}")
    with open(args.gene_list) as f:
        genes = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(genes)} unique genes")
    
    # Build knowledge graph
    logger.info("Building knowledge graph...")
    
    graph = build_knowledge_graph(
        genes=genes,
        output_path=args.output,
        dorothea_levels=args.dorothea_levels,
        string_score_threshold=args.string_threshold,
        string_local_links=args.string_links,
        string_local_info=args.string_info,
        go_terms=args.go_terms if args.include_go else None,
        include_go=args.include_go,
    )
    
    logger.info(f"Knowledge graph saved to {args.output}")
    logger.info(f"  Nodes: {graph.number_of_nodes()}")
    logger.info(f"  Edges: {graph.number_of_edges()}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
