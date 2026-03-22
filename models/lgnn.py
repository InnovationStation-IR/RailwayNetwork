"""
models/lgnn.py — Canonical Liquid Graph Neural Network with RK4 Integrator

This is the single, authoritative LGNN implementation.  All other files
(training/train_rl.py, models/ppo.py) MUST import from here.

Architecture:
    ┌──────────────┐
    │  GAT + GCN   │  ← Graph message-passing (2-layer)
    │   concat s   │
    └──────┬───────┘
           │  xs = [x ‖ s]
    ┌──────┴───────┐
    │ proposal_net │ → proposed (Tanh)
    │ gate_net     │ → g        (Sigmoid)
    │ tau_net      │ → τ        (Softplus + ε)
    └──────────────┘
           │
    dx/dt = (g · proposed  +  (1-g) · (−κ·x)) / τ

Integration:
    Fixed-step 4th-order Runge-Kutta (RK4).
    No dependency on torchdiffeq.

Changes from previous versions:
    • Removed duplicate LGNN copies (was in lgnn.py, ppo.py, train_rl.py).
    • Replaced `torchdiffeq.odeint(method='dopri5')` with a self-contained RK4.
    • Removed dead code: LiquidRailwayControlSystem, RailwayControlAPI, SafetyGuard.
    • Kept RewardCalculator with cleaner interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LGNN  —  Liquid Graph Neural Network  (ODE dynamics)
# ══════════════════════════════════════════════════════════════════════════════

class LGNN(nn.Module):
    """
    Liquid Graph Neural Network.

    Computes  dx/dt  for node features on a graph.  Designed to be called by
    an ODE integrator (see `rk4_integrate` below).

    Combines:
      • GATConv  — multi-head graph attention for local neighbourhood
      • GCNConv  — spectral graph convolution for global structure
    with a gating mechanism inspired by Liquid Time-Constant Networks.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node feature vectors.
    num_heads : int
        Number of attention heads for the GAT layer.
    kappa : float
        Decay rate constant.
    epsilon : float
        Floor added to τ to ensure strict positivity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        kappa: float = 0.1,
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.kappa = float(kappa)
        self.epsilon = epsilon

        # Edge index must be set before forward()
        self.edge_index: Optional[torch.Tensor] = None

        # ── Graph convolution layers ─────────────────────────────────────
        self.gat = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
        )
        self.gcn = GCNConv(hidden_dim, hidden_dim, cached=False)

        # ── Proposal network  (what update to apply) ────────────────────
        self.proposal_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # ── Gate network  (how much of the proposal to accept) ──────────
        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Time-constant network  (adaptive integration speed) ─────────
        self.tau_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt.

        Parameters
        ----------
        t : Tensor
            Current time (unused in the dynamics but required by ODE interface).
        x : Tensor, shape (num_nodes, hidden_dim)
            Current node features.

        Returns
        -------
        dx_dt : Tensor, same shape as x.
        """
        if self.edge_index is None:
            raise ValueError("edge_index must be set before calling forward()")

        # Graph message passing
        s_gat = self.gat(x, self.edge_index)
        s_gcn = self.gcn(x, self.edge_index)
        s = s_gat + s_gcn

        # Concatenate current state with neighbourhood aggregation
        xs = torch.cat([x, s], dim=-1)

        # Proposal, gate, time-constant
        proposed = self.proposal_net(xs)
        g = torch.sigmoid(self.gate_net(xs))
        tau = F.softplus(self.tau_net(xs)) + self.epsilon

        # Decay term
        decay = -self.kappa * x

        # Liquid dynamics
        dx_dt = (g * proposed + (1.0 - g) * decay) / tau
        return dx_dt


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RK4 INTEGRATOR  (replaces torchdiffeq)
# ══════════════════════════════════════════════════════════════════════════════

def rk4_integrate(
    dynamics_fn,
    x0: torch.Tensor,
    t_span: float,
    n_steps: int = 4,
) -> torch.Tensor:
    """
    Fixed-step 4th-order Runge-Kutta integration.

    Parameters
    ----------
    dynamics_fn : callable(t, x) → dx/dt
        The ODE right-hand side (e.g. LGNN.forward).
    x0 : Tensor
        Initial state (num_nodes, hidden_dim).
    t_span : float
        Total integration time.
    n_steps : int
        Number of RK4 steps (dt = t_span / n_steps).

    Returns
    -------
    x_final : Tensor, same shape as x0.
    """
    dt = t_span / n_steps
    x = x0
    t = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)

    for _ in range(n_steps):
        k1 = dynamics_fn(t, x)
        k2 = dynamics_fn(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = dynamics_fn(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = dynamics_fn(t + dt, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt

    return x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STATE ENCODER  (raw features → hidden_dim)
# ══════════════════════════════════════════════════════════════════════════════

class RailwayStateEncoder(nn.Module):
    """
    Encodes raw per-node feature vectors into the LGNN's hidden dimension.

    Input features (per node, designed for 12-dim input):
        [0]  is_station              (0/1)
        [1]  block_occupied          (0/1)
        [2]  block_disrupted         (0/1)
        [3]  signal_aspect_red       (0/1)   — most restrictive signal on this edge
        [4]  signal_aspect_yellow    (0/1)
        [5]  signal_aspect_dbl_yellow(0/1)
        [6]  signal_aspect_green     (0/1)
        [7]  num_trains_here         (normalised)
        [8]  speed_restriction       (0..1)
        [9]  disruption_severity     (0..1)
        [10] time_since_signal_change (normalised)
        [11] edge_distance           (normalised)
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REWARD CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class RewardCalculator:
    """
    Computes a dense, shaped reward from simulation events.

    Components (all per decision-step):
        + throughput_reward     : trains that finished a journey segment
        - delay_penalty         : cumulative waiting time across all trains
        - conflict_penalty      : SPAD events or near-collisions
        + punctuality_bonus     : trains arriving within schedule tolerance
        - energy_penalty        : unnecessary signal changes (churn)
    """

    def __init__(
        self,
        w_throughput: float = 2.0,
        w_delay: float = 0.05,
        w_conflict: float = 50.0,
        w_punctuality: float = 5.0,
        w_energy: float = 0.01,
    ):
        self.w_throughput = w_throughput
        self.w_delay = w_delay
        self.w_conflict = w_conflict
        self.w_punctuality = w_punctuality
        self.w_energy = w_energy

    def compute(
        self,
        finished_trains: int = 0,
        total_wait_time: float = 0.0,
        spad_count: int = 0,
        on_time_arrivals: int = 0,
        signal_changes: int = 0,
    ) -> float:
        """
        Compute reward for a single decision step.

        Returns
        -------
        float — net reward (higher is better).
        """
        reward = 0.0
        reward += self.w_throughput * finished_trains
        reward -= self.w_delay * total_wait_time
        reward -= self.w_conflict * spad_count
        reward += self.w_punctuality * on_time_arrivals
        reward -= self.w_energy * signal_changes
        return reward


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CONVENIENCE:  build the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_lgnn_pipeline(
    input_dim: int = 12,
    hidden_dim: int = 128,
    num_heads: int = 4,
    kappa: float = 0.1,
    device: torch.device = torch.device("cpu"),
):
    """
    Convenience function to create the encoder + LGNN on the specified device.

    Returns (encoder, lgnn) — both on `device`.
    """
    encoder = RailwayStateEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    lgnn = LGNN(hidden_dim=hidden_dim, num_heads=num_heads, kappa=kappa).to(device)
    return encoder, lgnn
