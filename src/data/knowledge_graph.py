"""
Knowledge Graph Construction Module.

Builds a heterogeneous graph with:
- TF-target regulatory edges from DoRothEA/CollecTRI
- Protein-protein interaction edges from STRING
- Co-functional edges from Gene Ontology
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path
from typing import Optional
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# STRING API base URL
STRING_API_URL = "https://string-db.org/api"


def fetch_collectri_edges(
    organism: str = "human",
    confidence_levels: list[str] = ["A", "B"],
) -> pd.DataFrame:
    """
    Fetch TF-target regulatory edges from CollecTRI via decoupleR.
    
    Args:
        organism: Organism to query ('human' or 'mouse')
        confidence_levels: DoRothEA confidence levels to include
        
    Returns:
        DataFrame with columns: source, target, weight (sign)
    """
    logger.info("Fetching CollecTRI/DoRothEA regulatory network...")
    
    try:
        import decoupler as dc
        
        # Get CollecTRI network (comprehensive TF-target database)
        collectri = dc.get_collectri(organism=organism)
        logger.info(f"CollecTRI: {len(collectri)} interactions")
        
        # Also get DoRothEA for additional coverage
        dorothea = dc.get_dorothea(organism=organism)
        # Filter by confidence level
        if "confidence" in dorothea.columns:
            dorothea = dorothea[dorothea["confidence"].isin(confidence_levels)]
        logger.info(f"DoRothEA (levels {confidence_levels}): {len(dorothea)} interactions")
        
        # Combine and deduplicate
        # Expected columns: source (TF), target (gene), weight/mor (sign)
        edges = []
        
        for df, name in [(collectri, "collectri"), (dorothea, "dorothea")]:
            # Standardize column names
            if "source" in df.columns and "target" in df.columns:
                weight_col = "weight" if "weight" in df.columns else "mor"
                if weight_col in df.columns:
                    subset = df[["source", "target", weight_col]].copy()
                    subset.columns = ["source", "target", "weight"]
                else:
                    subset = df[["source", "target"]].copy()
                    subset["weight"] = 1.0
                subset["database"] = name
                edges.append(subset)
        
        if edges:
            combined = pd.concat(edges, ignore_index=True)
            # Deduplicate, keeping first occurrence
            combined = combined.drop_duplicates(subset=["source", "target"], keep="first")
            logger.info(f"Combined regulatory network: {len(combined)} unique edges")
            return combined
        else:
            logger.warning("No edges found from CollecTRI/DoRothEA")
            return pd.DataFrame(columns=["source", "target", "weight"])
            
    except ImportError:
        logger.error("decoupler not installed. Run: pip install decoupler")
        raise


def fetch_string_edges_api(
    genes: list[str],
    species: int = 9606,
    score_threshold: int = 700,
    batch_size: int = 200,
) -> pd.DataFrame:
    """
    Fetch PPI edges from STRING API.
    
    Args:
        genes: List of gene symbols to query
        species: NCBI taxonomy ID (9606 = human)
        score_threshold: Minimum combined score (0-1000)
        batch_size: Genes per API request
        
    Returns:
        DataFrame with columns: gene1, gene2, combined_score
    """
    logger.info(f"Fetching STRING interactions for {len(genes)} genes via API...")
    
    all_interactions = []
    
    # Process in batches
    for i in tqdm(range(0, len(genes), batch_size), desc="STRING API"):
        batch = genes[i:i + batch_size]
        identifiers = "%0d".join(batch)
        
        url = f"{STRING_API_URL}/tsv/network"
        params = {
            "identifiers": identifiers,
            "species": species,
            "required_score": score_threshold,
            "caller_identity": "adipocyte_perturbation_prediction",
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse TSV response
            lines = response.text.strip().split("\n")
            if len(lines) > 1:  # Has header + data
                header = lines[0].split("\t")
                for line in lines[1:]:
                    fields = line.split("\t")
                    if len(fields) >= 6:
                        interaction = dict(zip(header, fields))
                        all_interactions.append({
                            "gene1": interaction.get("preferredName_A", fields[2]),
                            "gene2": interaction.get("preferredName_B", fields[3]),
                            "combined_score": int(interaction.get("score", fields[5])),
                        })
        except requests.RequestException as e:
            logger.warning(f"STRING API error for batch {i}: {e}")
            continue
    
    if all_interactions:
        df = pd.DataFrame(all_interactions)
        df = df.drop_duplicates(subset=["gene1", "gene2"])
        logger.info(f"STRING API: {len(df)} interactions retrieved")
        return df
    else:
        return pd.DataFrame(columns=["gene1", "gene2", "combined_score"])


def fetch_string_edges_local(
    string_links_path: str | Path,
    string_info_path: str | Path,
    genes: Optional[set[str]] = None,
    score_threshold: int = 700,
) -> pd.DataFrame:
    """
    Load STRING PPI edges from local download.
    
    Download from: https://string-db.org/cgi/download
    Files needed:
    - 9606.protein.links.v12.0.txt.gz
    - 9606.protein.info.v12.0.txt.gz
    
    Args:
        string_links_path: Path to protein.links file
        string_info_path: Path to protein.info file
        genes: Optional set of genes to filter to
        score_threshold: Minimum combined score
        
    Returns:
        DataFrame with columns: gene1, gene2, combined_score
    """
    logger.info("Loading STRING interactions from local files...")
    
    # Load protein info for ENSP -> gene symbol mapping
    logger.info(f"Loading protein info from {string_info_path}")
    
    ensp_to_gene = {}
    open_fn = gzip.open if str(string_info_path).endswith(".gz") else open
    
    with open_fn(string_info_path, "rt") as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ensp_id = parts[0]  # e.g., 9606.ENSP00000000233
                gene_symbol = parts[1]
                ensp_to_gene[ensp_id] = gene_symbol
    
    logger.info(f"Loaded {len(ensp_to_gene)} ENSP -> gene mappings")
    
    # Load interactions
    logger.info(f"Loading interactions from {string_links_path}")
    
    interactions = []
    open_fn = gzip.open if str(string_links_path).endswith(".gz") else open
    
    with open_fn(string_links_path, "rt") as f:
        header = f.readline()  # Skip header
        for line in tqdm(f, desc="Parsing STRING"):
            parts = line.strip().split()
            if len(parts) >= 3:
                ensp1, ensp2, score = parts[0], parts[1], int(parts[2])
                
                if score < score_threshold:
                    continue
                
                gene1 = ensp_to_gene.get(ensp1)
                gene2 = ensp_to_gene.get(ensp2)
                
                if gene1 and gene2:
                    # Filter to genes of interest
                    if genes is None or (gene1 in genes and gene2 in genes):
                        interactions.append({
                            "gene1": gene1,
                            "gene2": gene2,
                            "combined_score": score,
                        })
    
    df = pd.DataFrame(interactions)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["gene1", "gene2"])
    logger.info(f"STRING local: {len(df)} interactions after filtering")
    
    return df


def fetch_string_edges(
    genes: list[str],
    species: int = 9606,
    score_threshold: int = 700,
    local_links_path: Optional[str | Path] = None,
    local_info_path: Optional[str | Path] = None,
    use_local: bool = True,
) -> pd.DataFrame:
    """
    Fetch STRING PPI edges (wrapper for local/API methods).
    
    Args:
        genes: List of gene symbols
        species: NCBI taxonomy ID
        score_threshold: Minimum combined score
        local_links_path: Path to local STRING links file
        local_info_path: Path to local STRING info file
        use_local: Whether to prefer local files
        
    Returns:
        DataFrame with PPI edges
    """
    if use_local and local_links_path and local_info_path:
        if Path(local_links_path).exists() and Path(local_info_path).exists():
            return fetch_string_edges_local(
                local_links_path,
                local_info_path,
                genes=set(genes),
                score_threshold=score_threshold,
            )
        else:
            logger.warning("Local STRING files not found, falling back to API")
    
    return fetch_string_edges_api(
        genes,
        species=species,
        score_threshold=score_threshold,
    )


def fetch_go_edges(
    genes: list[str],
    go_terms: list[str],
    min_shared_terms: int = 3,
) -> pd.DataFrame:
    """
    Create co-functional edges based on shared GO terms.
    
    Args:
        genes: List of gene symbols
        go_terms: GO term IDs to consider (e.g., GO:0045444)
        min_shared_terms: Minimum shared GO annotations to create edge
        
    Returns:
        DataFrame with columns: gene1, gene2, shared_terms
    """
    logger.info(f"Fetching GO annotations for {len(go_terms)} terms...")
    
    try:
        from goatools.obo_parser import GODag
        from goatools.anno.genetogo_reader import Gene2GoReader
    except ImportError:
        logger.warning("goatools not installed, skipping GO edges")
        return pd.DataFrame(columns=["gene1", "gene2", "shared_terms"])
    
    # This is a simplified implementation
    # In practice, you'd download GO annotations for human genes
    
    # For now, return empty DataFrame - will implement with proper GO data
    logger.warning("GO edge construction requires additional data files")
    return pd.DataFrame(columns=["gene1", "gene2", "shared_terms"])


def build_knowledge_graph(
    genes: list[str],
    output_path: str | Path,
    dorothea_levels: list[str] = ["A", "B"],
    string_score_threshold: int = 700,
    string_local_links: Optional[str | Path] = None,
    string_local_info: Optional[str | Path] = None,
    go_terms: Optional[list[str]] = None,
    include_go: bool = False,
) -> nx.DiGraph:
    """
    Build the complete knowledge graph.
    
    Args:
        genes: List of all genes to include
        output_path: Path to save the graph
        dorothea_levels: Confidence levels for DoRothEA
        string_score_threshold: Minimum STRING score
        string_local_links: Path to local STRING links file
        string_local_info: Path to local STRING info file
        go_terms: GO terms for co-functional edges
        include_go: Whether to include GO edges
        
    Returns:
        NetworkX DiGraph with heterogeneous edges
    """
    logger.info(f"Building knowledge graph for {len(genes)} genes...")
    output_path = Path(output_path)
    
    gene_set = set(genes)
    
    # Initialize graph
    G = nx.DiGraph()
    G.add_nodes_from(genes)
    
    # 1. Add regulatory edges (TF -> target)
    reg_edges = fetch_collectri_edges(
        organism="human",
        confidence_levels=dorothea_levels,
    )
    
    # Filter to genes in our set
    reg_edges = reg_edges[
        reg_edges["source"].isin(gene_set) & 
        reg_edges["target"].isin(gene_set)
    ]
    
    for _, row in reg_edges.iterrows():
        edge_type = "tf_activates" if row["weight"] > 0 else "tf_represses"
        G.add_edge(
            row["source"], 
            row["target"],
            edge_type=edge_type,
            weight=abs(row["weight"]),
        )
    
    logger.info(f"Added {len(reg_edges)} regulatory edges")
    
    # 2. Add PPI edges (undirected, but stored as bidirectional)
    ppi_edges = fetch_string_edges(
        genes,
        score_threshold=string_score_threshold,
        local_links_path=string_local_links,
        local_info_path=string_local_info,
        use_local=string_local_links is not None,
    )
    
    # Filter to genes in our set
    ppi_edges = ppi_edges[
        ppi_edges["gene1"].isin(gene_set) & 
        ppi_edges["gene2"].isin(gene_set)
    ]
    
    for _, row in ppi_edges.iterrows():
        # Add bidirectional edges for undirected PPI
        G.add_edge(
            row["gene1"],
            row["gene2"],
            edge_type="interacts_with",
            weight=row["combined_score"] / 1000.0,
        )
        G.add_edge(
            row["gene2"],
            row["gene1"],
            edge_type="interacts_with",
            weight=row["combined_score"] / 1000.0,
        )
    
    logger.info(f"Added {len(ppi_edges) * 2} PPI edges (bidirectional)")
    
    # 3. Add GO co-functional edges (optional)
    if include_go and go_terms:
        go_edges = fetch_go_edges(genes, go_terms)
        
        go_edges = go_edges[
            go_edges["gene1"].isin(gene_set) & 
            go_edges["gene2"].isin(gene_set)
        ]
        
        for _, row in go_edges.iterrows():
            G.add_edge(
                row["gene1"],
                row["gene2"],
                edge_type="co_functional",
                weight=row["shared_terms"],
            )
            G.add_edge(
                row["gene2"],
                row["gene1"],
                edge_type="co_functional",
                weight=row["shared_terms"],
            )
        
        logger.info(f"Added {len(go_edges) * 2} GO edges (bidirectional)")
    
    # Summary
    logger.info(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Count edge types
    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get("edge_type", "unknown")] += 1
    
    for etype, count in edge_types.items():
        logger.info(f"  {etype}: {count} edges")
    
    # Save graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(G, output_path)
    logger.info(f"Saved graph to {output_path}")
    
    return G


def load_knowledge_graph(graph_path: str | Path) -> nx.DiGraph:
    """Load a saved knowledge graph."""
    return nx.read_gpickle(graph_path)


def convert_to_pyg(
    G: nx.DiGraph,
    node_features: Optional[dict[str, np.ndarray]] = None,
    feature_dim: int = 512,
) -> "torch_geometric.data.HeteroData":
    """
    Convert NetworkX graph to PyTorch Geometric HeteroData.
    
    Args:
        G: NetworkX graph
        node_features: Optional dict mapping gene names to feature vectors
        feature_dim: Dimension of node features (if not provided)
        
    Returns:
        PyG HeteroData object
    """
    import torch
    from torch_geometric.data import HeteroData
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Create HeteroData
    data = HeteroData()
    
    # Add node features
    if node_features is not None:
        features = []
        for node in nodes:
            if node in node_features:
                features.append(node_features[node])
            else:
                # Random initialization for missing nodes
                features.append(np.random.randn(feature_dim).astype(np.float32))
        data["gene"].x = torch.tensor(np.stack(features), dtype=torch.float32)
    else:
        # Random initialization
        data["gene"].x = torch.randn(len(nodes), feature_dim)
    
    # Store node mapping
    data["gene"].node_names = nodes
    data["gene"].node_to_idx = node_to_idx
    
    # Group edges by type
    edge_types = defaultdict(list)
    edge_weights = defaultdict(list)
    
    for src, dst, edata in G.edges(data=True):
        etype = edata.get("edge_type", "unknown")
        edge_types[etype].append([node_to_idx[src], node_to_idx[dst]])
        edge_weights[etype].append(edata.get("weight", 1.0))
    
    # Add edge indices for each type
    for etype, edges in edge_types.items():
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights[etype], dtype=torch.float32)
        
        # In PyG HeteroData, edge types are tuples: (src_type, relation, dst_type)
        data["gene", etype, "gene"].edge_index = edge_index
        data["gene", etype, "gene"].edge_weight = edge_weight
    
    return data
