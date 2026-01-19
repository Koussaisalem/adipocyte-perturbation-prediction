"""
Conditional Flow Matching Decoder.

Learns a time-dependent vector field to transform cells from
the NC distribution to the perturbed distribution.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

logger = logging.getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for time."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal time embedding.
        
        Args:
            t: Time values (batch_size,) or scalar
            
        Returns:
            Time embeddings (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class ConditionalVelocityMLP(nn.Module):
    """
    MLP that predicts velocity conditioned on state, time, and perturbation.
    
    Architecture: [x_dim + perturbation_dim + time_dim] -> hidden -> ... -> x_dim
    """
    
    def __init__(
        self,
        x_dim: int = 500,
        perturbation_dim: int = 256,
        time_dim: int = 64,
        hidden_dims: list[int] = [1024, 512, 512],
        activation: str = "silu",
        dropout: float = 0.1,
    ):
        """
        Initialize the velocity MLP.
        
        Args:
            x_dim: Dimension of state (PCA components)
            perturbation_dim: Dimension of perturbation embedding
            time_dim: Dimension of time embedding
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('silu', 'relu', 'gelu')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.x_dim = x_dim
        self.perturbation_dim = perturbation_dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Activation
        if activation == "silu":
            act_fn = nn.SiLU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP
        input_dim = x_dim + perturbation_dim + time_dim
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, x_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity at state x, time t, conditioned on perturbation z_p.
        
        Args:
            x: Current state (batch_size, x_dim)
            t: Time (batch_size,) or scalar
            z_p: Perturbation embedding (batch_size, perturbation_dim)
            
        Returns:
            Predicted velocity (batch_size, x_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Expand t_emb if needed
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)
        
        # Expand z_p if needed
        if z_p.shape[0] == 1 and x.shape[0] > 1:
            z_p = z_p.expand(x.shape[0], -1)
        
        # Concatenate inputs
        h = torch.cat([x, t_emb, z_p], dim=-1)
        
        # Predict velocity
        v = self.mlp(h)
        
        return v


class FlowMatchingDecoder(nn.Module):
    """
    Flow Matching decoder for generating perturbed cell distributions.
    
    Uses ODE integration to transform NC cells to perturbed cells.
    """
    
    def __init__(
        self,
        velocity_model: ConditionalVelocityMLP,
        ode_solver: str = "dopri5",
        ode_steps: int = 20,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """
        Initialize the flow matching decoder.
        
        Args:
            velocity_model: MLP that predicts velocity
            ode_solver: ODE solver to use ('dopri5', 'euler', 'rk4')
            ode_steps: Number of ODE integration steps (for fixed-step solvers)
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
        """
        super().__init__()
        
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq is required for FlowMatchingDecoder")
        
        self.velocity_model = velocity_model
        self.ode_solver = ode_solver
        self.ode_steps = ode_steps
        # Explicitly convert to float in case they come from YAML as strings
        self.rtol = float(rtol)
        self.atol = float(atol)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity for flow matching training.
        
        Args:
            x: Interpolated state x_t (batch_size, x_dim)
            t: Time t (batch_size,)
            z_p: Perturbation embedding (batch_size, perturbation_dim)
            
        Returns:
            Predicted velocity (batch_size, x_dim)
        """
        return self.velocity_model(x, t, z_p)
    
    def sample(
        self,
        x0: torch.Tensor,
        z_p: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample from the learned flow by ODE integration.
        
        Transforms x0 (NC cells) to x1 (perturbed cells).
        
        Args:
            x0: Initial state (batch_size, x_dim)
            z_p: Perturbation embedding (batch_size, perturbation_dim) or (1, perturbation_dim)
            n_steps: Number of integration steps (overrides default)
            
        Returns:
            Final state x1 (batch_size, x_dim)
        """
        n_steps = n_steps or self.ode_steps
        
        # Define ODE function
        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.velocity_model(x, t, z_p)
        
        # Time points
        t_span = torch.linspace(0, 1, n_steps + 1, device=x0.device)
        
        # Integrate ODE
        if self.ode_solver in ["dopri5", "dopri8"]:
            # Adaptive solver
            trajectory = odeint(
                ode_func,
                x0,
                t_span,
                method=self.ode_solver,
                rtol=self.rtol,
                atol=self.atol,
            )
        else:
            # Fixed-step solver
            trajectory = odeint(
                ode_func,
                x0,
                t_span,
                method=self.ode_solver,
            )
        
        # Return final state
        return trajectory[-1]
    
    def sample_trajectory(
        self,
        x0: torch.Tensor,
        z_p: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Sample full trajectory from t=0 to t=1.
        
        Args:
            x0: Initial state (batch_size, x_dim)
            z_p: Perturbation embedding
            n_steps: Number of time points to return
            
        Returns:
            Trajectory (n_steps + 1, batch_size, x_dim)
        """
        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.velocity_model(x, t, z_p)
        
        t_span = torch.linspace(0, 1, n_steps + 1, device=x0.device)
        
        trajectory = odeint(
            ode_func,
            x0,
            t_span,
            method=self.ode_solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        
        return trajectory


class OptimalTransportCFM(nn.Module):
    """
    Optimal Transport Conditional Flow Matching.
    
    Uses mini-batch OT to pair NC and perturbed cells for training.
    """
    
    def __init__(
        self,
        velocity_model: ConditionalVelocityMLP,
        sigma: float = 0.0,
    ):
        """
        Initialize OT-CFM.
        
        Args:
            velocity_model: MLP that predicts velocity
            sigma: Noise level for stochastic interpolation
        """
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma
    
    def compute_ot_pairing(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal transport pairing between x0 and x1.
        
        Args:
            x0: Source samples (n, d)
            x1: Target samples (m, d)
            
        Returns:
            Paired (x0, x1) samples
        """
        try:
            import ot
            
            # Compute cost matrix
            M = torch.cdist(x0, x1, p=2).pow(2)
            M_np = M.detach().cpu().numpy()
            
            # Compute OT plan
            n, m = x0.shape[0], x1.shape[0]
            a = torch.ones(n) / n
            b = torch.ones(m) / m
            
            plan = ot.emd(a.numpy(), b.numpy(), M_np)
            plan = torch.tensor(plan, device=x0.device)
            
            # Sample pairs according to plan
            # For simplicity, use greedy matching
            pairs = []
            for i in range(n):
                j = plan[i].argmax().item()
                pairs.append((i, j))
            
            indices_0 = torch.tensor([p[0] for p in pairs], device=x0.device)
            indices_1 = torch.tensor([p[1] for p in pairs], device=x0.device)
            
            return x0[indices_0], x1[indices_1]
            
        except ImportError:
            # Fallback to random pairing
            logger.warning("POT not installed, using random pairing")
            perm = torch.randperm(x1.shape[0])[:x0.shape[0]]
            return x0, x1[perm]
    
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        z_p: torch.Tensor,
        use_ot: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CFM loss.
        
        Args:
            x0: NC cells (batch_size, x_dim)
            x1: Perturbed cells (batch_size, x_dim)
            z_p: Perturbation embedding (batch_size, perturbation_dim)
            use_ot: Whether to use OT pairing
            
        Returns:
            Tuple of (predicted velocity, target velocity)
        """
        if use_ot:
            x0, x1 = self.compute_ot_pairing(x0, x1)
        
        # Sample time
        t = torch.rand(x0.shape[0], device=x0.device)
        
        # Interpolate
        xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        
        # Add noise if sigma > 0
        if self.sigma > 0:
            xt = xt + self.sigma * torch.randn_like(xt)
        
        # Target velocity (straight line)
        target = x1 - x0
        
        # Predicted velocity
        pred = self.velocity_model(xt, t, z_p)
        
        return pred, target
