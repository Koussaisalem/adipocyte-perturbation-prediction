"""Utility functions for submission - inlined to avoid import issues."""

import pickle
import networkx as nx
import torch
from torch_geometric.data import HeteroData


def load_knowledge_graph(path):
    """Load knowledge graph from gpickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def convert_to_pyg(kg, node_features, feature_dim):
    """
    Convert NetworkX graph to PyTorch Geometric HeteroData format.
    
    Args:
        kg: NetworkX DiGraph with gene nodes and regulatory edges
        node_features: Dict mapping gene names to embedding vectors
        feature_dim: Dimension of node features
        
    Returns:
        HeteroData object with gene nodes and interaction edges
    """
    data = HeteroData()
    
    # Create node mapping
    genes = sorted(kg.nodes())
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    
    # Node features
    x = torch.zeros(len(genes), feature_dim)
    for gene, idx in gene_to_idx.items():
        if gene in node_features:
            x[idx] = node_features[gene]
    
    data['gene'].x = x
    data['gene'].num_nodes = len(genes)
    
    # Edge indices - combine all edge types into single "interacts_with"
    edge_list = []
    for src, dst, edge_data in kg.edges(data=True):
        if src in gene_to_idx and dst in gene_to_idx:
            edge_list.append([gene_to_idx[src], gene_to_idx[dst]])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        data['gene', 'interacts_with', 'gene'].edge_index = edge_index
    else:
        # Empty graph
        data['gene', 'interacts_with', 'gene'].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return data
