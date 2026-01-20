"""
GATv2 Encoder for Perturbation Embeddings.

Uses Graph Attention Networks to learn perturbation representations
from the biological knowledge graph.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, HeteroConv, Linear
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger(__name__)


class GATv2Layer(nn.Module):
    """Single GATv2 layer with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        add_self_loops: bool = True,
    ):
        super().__init__()
        
        self.conv = GATv2Conv(
            in_channels,
            out_channels // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=add_self_loops,
            concat=True,
        )
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connection."""
        # GATv2 convolution
        out = self.conv(x, edge_index)
        out = F.elu(out)
        out = self.dropout(out)
        
        # Residual connection
        out = out + self.residual(x)
        out = self.norm(out)
        
        return out


class GATv2Encoder(nn.Module):
    """
    Multi-layer GATv2 encoder for gene nodes.
    
    Learns node representations by propagating information
    across the knowledge graph structure.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the GATv2 encoder.
        
        Args:
            input_dim: Dimension of input node features (e.g., Geneformer embeddings)
            hidden_dim: Hidden dimension per head
            output_dim: Output dimension
            num_layers: Number of GATv2 layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GATv2Encoder")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim * heads)
        
        # GATv2 layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * heads
            out_dim = hidden_dim * heads if i < num_layers - 1 else output_dim
            
            self.layers.append(GATv2Layer(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads if i < num_layers - 1 else 1,
                dropout=dropout,
            ))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GATv2 layers.
        
        Args:
            x: Node features (n_nodes, input_dim)
            edge_index: Edge indices (2, n_edges)
            edge_weight: Optional edge weights
            
        Returns:
            Updated node representations (n_nodes, output_dim)
        """
        # Project input
        x = self.input_proj(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GATv2 layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        return x


class HeteroGATv2Encoder(nn.Module):
    """
    Heterogeneous GATv2 encoder for multi-relational knowledge graph.
    
    Handles different edge types (tf_activates, tf_represses, interacts_with).
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        edge_types: list[str] = ["tf_activates", "tf_represses", "interacts_with"],
    ):
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for HeteroGATv2Encoder")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_types = edge_types
        
        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim * heads)
        
        # Heterogeneous GATv2 layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim * heads
            out_dim = hidden_dim * heads if i < num_layers - 1 else output_dim
            n_heads = heads if i < num_layers - 1 else 1
            
            # Create convolution for each edge type
            conv_dict = {}
            for etype in edge_types:
                conv_dict[("gene", etype, "gene")] = GATv2Conv(
                    in_dim,
                    out_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True,
                )
            
            self.layers.append(HeteroConv(conv_dict, aggr="sum"))
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * heads if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: "HeteroData") -> torch.Tensor:
        """
        Forward pass through heterogeneous GATv2 layers.
        
        Args:
            data: PyG HeteroData with node features and edge indices
            
        Returns:
            Updated node representations (n_nodes, output_dim)
        """
        x_dict = {"gene": data["gene"].x}
        
        # Project input
        x_dict["gene"] = self.input_proj(x_dict["gene"])
        x_dict["gene"] = F.elu(x_dict["gene"])
        x_dict["gene"] = self.dropout(x_dict["gene"])
        
        # Heterogeneous convolutions
        for layer, norm in zip(self.layers, self.norms):
            x_residual = x_dict["gene"]
            x_dict = layer(x_dict, data.edge_index_dict)
            x_dict["gene"] = F.elu(x_dict["gene"])
            x_dict["gene"] = self.dropout(x_dict["gene"])
            
            # Residual if dimensions match
            if x_residual.shape[-1] == x_dict["gene"].shape[-1]:
                x_dict["gene"] = x_dict["gene"] + x_residual
            
            x_dict["gene"] = norm(x_dict["gene"])
        
        return x_dict["gene"]


class PerturbationEncoder(nn.Module):
    """
    Encodes perturbations into latent vectors using the knowledge graph.
    
    For a knockout perturbation, we:
    1. Zero out the knocked-out gene's features
    2. Run message passing to propagate the "effect"
    3. Aggregate neighborhood information into a perturbation embedding
    """
    
    def __init__(
        self,
        gat_encoder: GATv2Encoder | HeteroGATv2Encoder,
        aggregation: str = "attention",
        perturbation_dim: int = 256,
        n_hops: int = 2,
    ):
        """
        Initialize the perturbation encoder.
        
        Args:
            gat_encoder: Pre-initialized GATv2 encoder
            aggregation: How to aggregate neighborhood ('attention', 'mean', 'max')
            perturbation_dim: Output dimension for perturbation embedding
            n_hops: Number of hops to consider for neighborhood aggregation
        """
        super().__init__()
        
        self.gat_encoder = gat_encoder
        self.aggregation = aggregation
        self.perturbation_dim = perturbation_dim
        self.n_hops = n_hops
        
        # Aggregation mechanism
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(gat_encoder.output_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        
        # Final projection
        self.output_proj = nn.Linear(gat_encoder.output_dim, perturbation_dim)
    
    def get_neighborhood(
        self,
        node_idx: int,
        edge_index: torch.Tensor,
        n_hops: int,
    ) -> torch.Tensor:
        """Get indices of nodes within n_hops of the target node."""
        neighbors = {node_idx}
        current_frontier = {node_idx}
        
        edge_list = edge_index.t().tolist()
        adj = {}
        for src, dst in edge_list:
            if src not in adj:
                adj[src] = []
            adj[src].append(dst)
        
        for _ in range(n_hops):
            new_frontier = set()
            for node in current_frontier:
                if node in adj:
                    new_frontier.update(adj[node])
            neighbors.update(new_frontier)
            current_frontier = new_frontier
        
        return torch.tensor(list(neighbors), dtype=torch.long)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        perturbation_indices: torch.Tensor,
        node_to_idx: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute perturbation embeddings.
        
        Args:
            node_features: Gene node features (n_genes, feature_dim)
            edge_index: Graph edge indices
            perturbation_indices: Indices of perturbed genes (batch_size,)
            node_to_idx: Optional mapping from gene names to indices
            
        Returns:
            Perturbation embeddings (batch_size, perturbation_dim)
        """
        batch_size = perturbation_indices.shape[0]
        device = node_features.device
        
        perturbation_embeddings = []
        
        for i in range(batch_size):
            pert_idx = perturbation_indices[i].item()
            
            # Create modified features (knockout = zero out the gene)
            x_modified = node_features.clone()
            x_modified[pert_idx] = 0
            
            # Run GATv2 on modified graph
            x_encoded = self.gat_encoder(x_modified, edge_index)
            
            # Get neighborhood of perturbed gene
            neighbor_indices = self.get_neighborhood(pert_idx, edge_index, self.n_hops)
            neighbor_indices = neighbor_indices.to(device)
            neighbor_features = x_encoded[neighbor_indices]
            
            # Aggregate neighborhood
            if self.aggregation == "attention":
                attn_scores = self.attention(neighbor_features)
                attn_weights = F.softmax(attn_scores, dim=0)
                aggregated = (attn_weights * neighbor_features).sum(dim=0)
            elif self.aggregation == "mean":
                aggregated = neighbor_features.mean(dim=0)
            elif self.aggregation == "max":
                aggregated = neighbor_features.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
            
            perturbation_embeddings.append(aggregated)
        
        # Stack and project
        z_p = torch.stack(perturbation_embeddings)
        z_p = self.output_proj(z_p)
        
        return z_p


class CachedPerturbationEncoder(nn.Module):
    """
    Cached version of PerturbationEncoder for efficient inference.
    
    Pre-computes perturbation embeddings for all genes in the graph,
    then retrieves them by index during inference.
    """
    
    def __init__(
        self,
        perturbation_encoder: PerturbationEncoder,
    ):
        super().__init__()
        self.perturbation_encoder = perturbation_encoder
        self.cache = None
        self.node_to_idx = None
    
    def build_cache(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_to_idx: dict[str, int],
    ):
        """Pre-compute embeddings for all genes."""
        self.node_to_idx = node_to_idx
        n_genes = len(node_to_idx)
        
        logger.info(f"Building perturbation embedding cache for {n_genes} genes...")
        
        all_indices = torch.arange(n_genes, device=node_features.device)
        
        with torch.no_grad():
            self.cache = self.perturbation_encoder(
                node_features,
                edge_index,
                all_indices,
            )
        
        logger.info(f"Cache built: {self.cache.shape}")
    
    def forward(
        self,
        perturbation_names: list[str],
    ) -> torch.Tensor:
        """Retrieve cached embeddings by gene name."""
        if self.cache is None:
            raise RuntimeError("Cache not built. Call build_cache first.")
        
        indices = [self.node_to_idx[name] for name in perturbation_names]
        indices = torch.tensor(indices, device=self.cache.device)
        
        return self.cache[indices]
