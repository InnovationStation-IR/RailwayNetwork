#!/usr/bin/env python3
"""
training/train_rl.py — End-to-End PPO Training for Railway Signal Control

Architecture (why the LGNN matters):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SimPy Simulation  →  12-dim raw node features  (env observation)       │
  │                              ↓                                          │
  │          ┌──── RailwayActorCritic (the full policy network) ────┐       │
  │          │                                                       │       │
  │          │  RailwayStateEncoder   12-dim → 128-dim               │       │
  │          │         ↓                                             │       │
  │          │  LGNN + RK4 ODE integration on the GRAPH topology     │       │
  │          │    (GAT + GCN → proposal/gate/tau → dx/dt → 4 steps) │       │
  │          │         ↓                                             │       │
  │          │  Graph-aware node embeddings  (num_nodes × 128)       │       │
  │          │         ↓  mean-pool                                  │       │
  │          │  Global state vector  (128,)                          │       │
  │          │         ↓                                             │       │
  │          │  Actor head → per-edge Bernoulli (MultiBinary)        │       │
  │          │  Critic head → scalar V(s)                            │       │
  │          └───────────────────────────────────────────────────────┘       │
  │                              ↓                                          │
  │  PPO loss  backpropagates through  Actor + Critic + LGNN + Encoder      │
  │  → The LGNN LEARNS to propagate railway state across the graph          │
  │  → GAT attention heads learn which neighbour stations matter            │
  │  → Liquid τ adapts integration speed per-node (busy vs quiet areas)     │
  │  → Gate g learns when to update vs decay stale signal state             │
  └─────────────────────────────────────────────────────────────────────────┘

Usage:
    python -m training.train_rl --timesteps 50000
    python -m training.train_rl --datafile PATH --timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

# ── Import from our own modules (NO duplicate code) ─────────────────────────
from models.lgnn import LGNN, RailwayStateEncoder, RewardCalculator, rk4_integrate
from simulation.core import DataRecorder, Simulation, load_railway_graph
from simulation.signals import SignalAspect, SignalType


# ══════════════════════════════════════════════════════════════════════════════
# 0.  TRAINING LOGGER  (records everything for visualization)
# ══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """Collects all metrics needed for learning-curve visualizations."""

    def __init__(self):
        # Per-episode
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_global_steps: List[int] = []
        self.episode_trains_finished: List[int] = []
        self.episode_spads: List[int] = []
        self.episode_wall_time: List[float] = []

        # Per-PPO-update
        self.update_global_steps: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_values: List[float] = []
        self.total_losses: List[float] = []
        self.approx_kls: List[float] = []
        self.clip_fractions: List[float] = []
        self.learning_rates: List[float] = []
        self.ppo_epochs_used: List[int] = []

        # Gradient norms (proves LGNN is being trained)
        self.grad_norm_lgnn: List[float] = []
        self.grad_norm_encoder: List[float] = []
        self.grad_norm_heads: List[float] = []
        self.grad_norm_total: List[float] = []

        # Per-step rewards (for smoothed curve)
        self.step_rewards: List[float] = []

    def log_episode(self, reward: float, length: int, global_step: int,
                    trains_finished: int, spads: int, wall_time: float):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_global_steps.append(global_step)
        self.episode_trains_finished.append(trains_finished)
        self.episode_spads.append(spads)
        self.episode_wall_time.append(wall_time)

    def log_step_reward(self, reward: float):
        self.step_rewards.append(reward)

    def log_update(self, metrics: Dict[str, float], global_step: int, lr: float):
        self.update_global_steps.append(global_step)
        self.policy_losses.append(metrics["policy_loss"])
        self.value_losses.append(metrics["value_loss"])
        self.entropy_values.append(metrics["entropy"])
        self.total_losses.append(metrics["total_loss"])
        self.approx_kls.append(metrics["approx_kl"])
        self.clip_fractions.append(metrics["clip_fraction"])
        self.learning_rates.append(lr)
        self.ppo_epochs_used.append(metrics.get("epochs_used", 0))
        self.grad_norm_lgnn.append(metrics["grad_norm_lgnn"])
        self.grad_norm_encoder.append(metrics["grad_norm_encoder"])
        self.grad_norm_heads.append(metrics["grad_norm_heads"])
        self.grad_norm_total.append(metrics["grad_norm_total"])

    def save(self, path: str):
        data = {
            "episodes": {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "global_steps": self.episode_global_steps,
                "trains_finished": self.episode_trains_finished,
                "spads": self.episode_spads,
                "wall_time": self.episode_wall_time,
            },
            "updates": {
                "global_steps": self.update_global_steps,
                "policy_loss": self.policy_losses,
                "value_loss": self.value_losses,
                "entropy": self.entropy_values,
                "total_loss": self.total_losses,
                "approx_kl": self.approx_kls,
                "clip_fraction": self.clip_fractions,
                "learning_rate": self.learning_rates,
                "epochs_used": self.ppo_epochs_used,
                "grad_norm_lgnn": self.grad_norm_lgnn,
                "grad_norm_encoder": self.grad_norm_encoder,
                "grad_norm_heads": self.grad_norm_heads,
                "grad_norm_total": self.grad_norm_total,
            },
            "step_rewards": self.step_rewards,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Metrics saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PPO CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    learning_rate:          float = 3e-4
    n_steps:                int   = 512        # steps per rollout
    batch_size:             int   = 64
    n_epochs:               int   = 4          # PPO epochs per update
    gamma:                  float = 0.99
    gae_lambda:             float = 0.95
    clip_coef:              float = 0.2
    ent_coef:               float = 0.01
    vf_coef:                float = 0.5
    max_grad_norm:          float = 0.5
    target_kl:              Optional[float] = 0.02  # early-stop PPO epoch if KL exceeds
    device:                 torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Environment
    decision_interval_s:    float = 30.0       # agent acts every 30 sim-seconds
    num_trains_per_episode: int   = 30
    episode_duration_s:     float = 3600.0
    # Model
    input_dim:              int   = 12
    hidden_dim:             int   = 128
    rk4_steps:              int   = 4
    rk4_t_span:             float = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ACTOR-CRITIC NETWORK  (LGNN is the backbone — NOT a dead preprocessor)
# ══════════════════════════════════════════════════════════════════════════════

class RailwayActorCritic(nn.Module):
    """
    Graph-native Actor-Critic: the LGNN IS the policy backbone.

    Forward pass (with FULL gradient flow through the LGNN):
      raw features (N×12) → Encoder (N×128) → LGNN+RK4 (N×128) → mean-pool (128,)
                                                                   ↓
                                                       Actor → per-edge logits
                                                       Critic → V(s)

    PPO backprop updates Encoder + LGNN + Actor + Critic jointly.
    This means the LGNN *learns* to:
      • Propagate congestion / disruption info across the graph topology
      • Use GAT attention to focus on critical neighbour stations
      • Adapt its liquid time-constant τ per-node (fast updates for busy junctions,
        slow decay for quiet stretches)
      • Gate which graph messages to accept vs ignore via learned gate g
    """

    def __init__(self, config: PPOConfig, num_edges: int):
        super().__init__()
        self.config = config
        self.num_edges = num_edges
        h = config.hidden_dim

        # ── Graph backbone (TRAINED by PPO) ──────────────────────────────
        self.encoder = RailwayStateEncoder(
            input_dim=config.input_dim, hidden_dim=h,
        )
        self.lgnn = LGNN(hidden_dim=h)

        # ── Policy heads ─────────────────────────────────────────────────
        self.shared_head = nn.Sequential(
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(256, num_edges)
        self.critic_head = nn.Linear(256, 1)

    def set_edge_index(self, edge_index: torch.Tensor):
        """Set the graph topology for LGNN message passing."""
        self.lgnn.edge_index = edge_index

    def _backbone(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        raw_features: (num_nodes, input_dim) → global state vector (hidden_dim,)

        This runs the LGNN WITH gradients — the entire graph neural ODE is
        part of the computation graph for PPO.
        """
        # Encode raw features → hidden dim
        h0 = self.encoder(raw_features)                # (N, hidden_dim)

        # Integrate liquid dynamics on the graph via RK4
        h_final = rk4_integrate(
            self.lgnn, h0,
            t_span=self.config.rk4_t_span,
            n_steps=self.config.rk4_steps,
        )                                               # (N, hidden_dim)

        # Mean-pool across all nodes → single graph-level embedding
        return h_final.mean(dim=0)                      # (hidden_dim,)

    def forward(self, raw_features: torch.Tensor):
        """Full forward: features → backbone → actor + critic."""
        state = self._backbone(raw_features)
        shared = self.shared_head(state)
        return self.actor_head(shared), self.critic_head(shared)

    def get_value(self, raw_features: torch.Tensor) -> torch.Tensor:
        state = self._backbone(raw_features)
        shared = self.shared_head(state)
        return self.critic_head(shared)

    def get_action_and_value(
        self,
        raw_features: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """
        Sample or evaluate per-edge binary actions.

        Args:
            raw_features: (num_nodes, input_dim) — 12-dim raw node features
            action: if provided, evaluate this action's log_prob (for PPO update)

        Returns: (action, log_prob, entropy, value)
        """
        logits, value = self.forward(raw_features)

        # Independent Bernoulli per edge
        dist = torch.distributions.Bernoulli(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ROLLOUT BUFFER  (stores RAW node features, not embeddings)
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Stores raw node-feature tensors (num_nodes × input_dim) per step.

    During PPO update, each stored observation is re-processed through the
    LGNN backbone WITH gradients — this is what makes the LGNN actually learn.
    """

    def __init__(self, n_steps: int, num_nodes: int, input_dim: int,
                 num_edges: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        # Store raw features per step
        self.obs      = torch.zeros((n_steps, num_nodes, input_dim), device=device)
        self.actions  = torch.zeros((n_steps, num_edges), device=device)
        self.logprobs = torch.zeros((n_steps,), device=device)
        self.rewards  = torch.zeros((n_steps,), device=device)
        self.dones    = torch.zeros((n_steps,), device=device)
        self.values   = torch.zeros((n_steps,), device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.flatten()
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor,
                                        done: float, gamma: float,
                                        gae_lambda: float):
        self.advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t < self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            else:
                next_non_terminal = 1.0 - done
                next_value = last_value
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """
        Yields individual step indices (NOT mini-batches of flattened vectors).

        Each yielded item is a single step's raw features so the LGNN can
        process the full graph topology.  We accumulate gradients across
        `batch_size` steps before each optimizer step.
        """
        indices = np.random.permutation(self.n_steps)
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, self.n_steps, batch_size):
            bi = indices[start:start + batch_size]
            yield bi, self.obs[bi], self.actions[bi], self.logprobs[bi], adv[bi], self.returns[bi]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GYMNASIUM ENVIRONMENT  (returns RAW features — LGNN lives in the agent)
# ══════════════════════════════════════════════════════════════════════════════

class RailwayGymEnv(gym.Env):
    """
    Wraps simulation/core.py Simulation into a Gymnasium env.

    Observation:  raw 12-dim per-node features as flat array (num_nodes × 12)
    Action:       per-edge binary {0, 1}^num_edges
    Reward:       dense shaped reward from RewardCalculator

    The LGNN does NOT live here — it lives in the ActorCritic where PPO can
    actually train it via backpropagation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: PPOConfig, datafile: str):
        super().__init__()
        self.config = config
        self.datafile = datafile
        self.reward_calc = RewardCalculator()

        self.sim: Optional[Simulation] = None
        self.num_nodes: int = 0
        self.num_edges: int = 0

        # Will be set on first reset()
        self.observation_space = None
        self.action_space = None

        # Caches
        self._node_to_idx: Dict[str, int] = {}
        self._edge_keys: List[Tuple[str, str]] = []
        self._edge_key_strs: List[str] = []
        self._block_to_signals: Dict[Tuple, list] = {}
        self._static_station: Optional[torch.Tensor] = None
        self._has_disruption_mgr = False
        self._edge_index: Optional[torch.Tensor] = None

    @property
    def graph_edge_index(self) -> Optional[torch.Tensor]:
        """PyG edge_index tensor — needed by the ActorCritic to set LGNN topology."""
        return self._edge_index

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim = Simulation(
            datafile=self.datafile,
            headless=True,
            seed=random.randint(0, 1_000_000),
            cfg={
                "episode_duration": self.config.episode_duration_s,
                "train_speed_pps": 12.0,
                "snapshot_interval": self.config.decision_interval_s,
            },
        )
        if not self.sim.is_running:
            if self.observation_space is None:
                self.observation_space = spaces.Box(-1, 1, shape=(1, self.config.input_dim), dtype=np.float32)
                self.action_space = spaces.MultiBinary(1)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        # Build node / edge caches
        self._node_to_idx = {nid: i for i, nid in enumerate(self.sim.graph.nodes())}
        self._edge_keys = sorted(self.sim.blocks.keys())
        self.num_nodes = len(self._node_to_idx)
        self.num_edges = len(self._edge_keys)

        # Set spaces (first time only — Gym requires fixed shapes)
        if self.observation_space is None:
            self.observation_space = spaces.Box(
                low=-1.0, high=2.0,
                shape=(self.num_nodes, self.config.input_dim),
                dtype=np.float32,
            )
            self.action_space = spaces.MultiBinary(self.num_edges)

        # Build PyG edge_index for the LGNN
        edge_list = []
        for u, v in self.sim.graph.edges():
            ui, vi = self._node_to_idx[u], self._node_to_idx[v]
            edge_list.append([ui, vi])
            edge_list.append([vi, ui])
        self._edge_index = torch.tensor(
            edge_list, dtype=torch.long,
        ).t().contiguous().to(self.config.device)

        # Pre-compute caches
        self._edge_key_strs = [str(sorted(ek)) for ek in self._edge_keys]
        self._block_to_signals = defaultdict(list)
        for sig in self.sim.signals.values():
            self._block_to_signals[sig.controls_block].append(sig)
        self._static_station = torch.zeros(self.num_nodes, device=self.config.device)
        for nid, idx in self._node_to_idx.items():
            if self.sim.graph.nodes[nid].get("label"):
                self._static_station[idx] = 1.0
        self._has_disruption_mgr = hasattr(self.sim, '_disruption_mgr')

        # Spawn trains and advance one tick
        ods = self.sim.sample_od_pairs(self.sim.stations, self.config.num_trains_per_episode)
        self.sim.spawn_trains_from_ods(ods)
        self.sim.schedule_random_disruptions(disruption_rate=0.0005)
        self.sim.advance(1.0)
        self._step_count = 0

        obs = self._build_raw_features()
        return obs.cpu().numpy(), {}

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, action):
        if self.sim is None or not self.sim.is_running:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

        self._step_count += 1

        # Apply per-edge green mask
        green_mask: Dict[str, bool] = {}
        for i, eks in enumerate(self._edge_key_strs):
            green_mask[eks] = bool(action[i])
        self.sim.apply_signal_actions(green_mask, self.sim.current_sim_time)

        # Advance simulation
        self.sim.advance(self.config.decision_interval_s)

        # Compute reward
        events = self.sim.recorder.get_and_clear_events()
        finished = sum(1 for e in events if e["kind"] == "finish")
        total_wait = sum(
            e["payload"].get("duration", 0.0)
            for e in events if e["kind"] == "wait_end"
        )
        spad_count = sum(1 for e in events if e["kind"] == "spad")
        signal_changes = sum(
            e["payload"].get("count", 0)
            for e in events if e["kind"] == "signal_change"
        )
        reward = self.reward_calc.compute(
            finished_trains=finished,
            total_wait_time=total_wait,
            spad_count=spad_count,
            signal_changes=signal_changes,
        )

        # Next observation
        obs = self._build_raw_features()

        # Termination
        has_trains = len(self.sim.trains) > 0
        all_done = all(not t.is_visible for t in self.sim.trains) if has_trains else False
        terminated = has_trains and all_done
        truncated = self.sim.current_sim_time >= self.config.episode_duration_s

        return obs.cpu().numpy(), float(reward), terminated, truncated, {}

    # ── raw feature builder (NO LGNN here — that's in the ActorCritic) ──

    def _build_raw_features(self) -> torch.Tensor:
        """
        Build 12-dim per-node features from current simulation state.

        Returns: (num_nodes, 12) tensor.  The LGNN processes this in the
        ActorCritic's forward pass, not here.
        """
        dev = self.config.device
        N = self.num_nodes
        features = torch.zeros(N, self.config.input_dim, device=dev)

        # [0] is_station
        features[:, 0] = self._static_station

        # [7] num_trains_here (normalised)
        for t in self.sim.trains:
            if t.is_visible and t.current_index < len(t.route):
                idx = self._node_to_idx.get(t.route[t.current_index])
                if idx is not None:
                    features[idx, 7] = min(features[idx, 7].item() + 0.2, 1.0)

        # Edge / block / signal features → map to endpoint nodes
        _aspect_map = {
            SignalAspect.RED: 3, SignalAspect.YELLOW: 4,
            SignalAspect.DOUBLE_YELLOW: 5, SignalAspect.GREEN: 6,
        }

        for u, v in self.sim.graph.edges():
            a, b = sorted((u, v))
            bk = (a, b)
            ui, vi = self._node_to_idx[u], self._node_to_idx[v]

            block = self.sim.blocks.get(bk)
            if block:
                if block.is_occupied:
                    features[ui, 1] = 1.0; features[vi, 1] = 1.0
                if block.is_disrupted:
                    features[ui, 2] = 1.0; features[vi, 2] = 1.0

            for sig in self._block_to_signals.get(bk, ()):
                aidx = _aspect_map.get(sig.aspect, 3)
                features[ui, aidx] = 1.0; features[vi, aidx] = 1.0

            if self._has_disruption_mgr:
                sr = self.sim._disruption_mgr.get_speed_restriction(bk)
                if sr < 1.0:
                    val = 1.0 - sr
                    features[ui, 8] = max(features[ui, 8].item(), val)
                    features[vi, 8] = max(features[vi, 8].item(), val)

            dist = self.sim.graph.edges[u, v].get("distance", 1.0)
            nd = min(dist / 5000.0, 1.0)
            features[ui, 11] = max(features[ui, 11].item(), nd)
            features[vi, 11] = max(features[vi, 11].item(), nd)

        return features


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PPO AGENT  (LGNN parameters included in optimizer)
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    PPO agent that trains the ENTIRE pipeline: Encoder + LGNN + Actor + Critic.

    Because the LGNN lives inside RailwayActorCritic, calling
    optimizer.step() updates:
      - GAT attention weights (learn which neighbours matter)
      - GCN spectral weights (global graph diffusion)
      - Proposal net (what signal changes to recommend)
      - Gate net (when to accept vs ignore graph messages)
      - Tau net (adaptive integration speed per node)
      - Encoder projection (raw features → hidden space)
      - Actor / Critic heads
    """

    def __init__(self, config: PPOConfig, ac: RailwayActorCritic,
                 num_nodes: int, num_edges: int):
        self.config = config
        self.ac = ac

        # ALL parameters in one optimizer — LGNN included
        total_params = sum(p.numel() for p in ac.parameters())
        lgnn_params = sum(p.numel() for p in ac.lgnn.parameters())
        enc_params = sum(p.numel() for p in ac.encoder.parameters())
        head_params = total_params - lgnn_params - enc_params
        print(f"  Model parameters: {total_params:,d} total")
        print(f"    LGNN (backbone):  {lgnn_params:,d} ({100*lgnn_params/total_params:.1f}%)")
        print(f"    Encoder:          {enc_params:,d} ({100*enc_params/total_params:.1f}%)")
        print(f"    Actor+Critic:     {head_params:,d} ({100*head_params/total_params:.1f}%)")

        self.optimizer = optim.Adam(ac.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(
            config.n_steps, num_nodes, config.input_dim, num_edges, config.device,
        )

    def _compute_grad_norms(self) -> Dict[str, float]:
        """Compute per-module gradient norms (proves LGNN is being trained)."""
        def _norm(params):
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
            return total ** 0.5

        return {
            "grad_norm_lgnn": _norm(self.ac.lgnn.parameters()),
            "grad_norm_encoder": _norm(self.ac.encoder.parameters()),
            "grad_norm_heads": _norm(
                list(self.ac.shared_head.parameters()) +
                list(self.ac.actor_head.parameters()) +
                list(self.ac.critic_head.parameters())
            ),
            "grad_norm_total": _norm(self.ac.parameters()),
        }

    def learn(self, next_obs: torch.Tensor, done: float) -> Dict[str, float]:
        """
        PPO update — backprop flows through LGNN + RK4 ODE integration.

        Returns a metrics dict with loss breakdown and gradient norms.
        """
        with torch.no_grad():
            next_value = self.ac.get_value(next_obs).flatten()
        self.buffer.compute_returns_and_advantages(
            next_value, done, self.config.gamma, self.config.gae_lambda,
        )

        # Accumulators for metrics
        all_pg_loss, all_v_loss, all_ent, all_total = [], [], [], []
        all_clip_frac = []
        all_kl = []
        last_grad_norms = {}
        epochs_used = 0

        for epoch in range(self.config.n_epochs):
            epochs_used = epoch + 1
            approx_kl_sum = 0.0
            n_batches = 0

            for bi, obs_b, act_b, lp_b, adv_b, ret_b in self.buffer.get_batches(self.config.batch_size):
                batch_pg_loss = torch.tensor(0.0, device=self.config.device)
                batch_v_loss = torch.tensor(0.0, device=self.config.device)
                batch_ent = torch.tensor(0.0, device=self.config.device)
                batch_kl = 0.0
                batch_clip = 0.0
                bs = len(bi)

                for j in range(bs):
                    _, new_lp, entropy, new_val = self.ac.get_action_and_value(
                        obs_b[j], act_b[j],
                    )
                    ratio = (new_lp - lp_b[j]).exp()

                    # Clipped surrogate
                    pg1 = -adv_b[j] * ratio
                    pg2 = -adv_b[j] * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef,
                    )
                    batch_pg_loss = batch_pg_loss + torch.max(pg1, pg2)
                    batch_v_loss = batch_v_loss + (new_val.view(-1) - ret_b[j]) ** 2
                    batch_ent = batch_ent + entropy

                    with torch.no_grad():
                        log_ratio = new_lp - lp_b[j]
                        batch_kl += ((ratio - 1) - log_ratio).item()
                        batch_clip += (abs(ratio.item() - 1.0) > self.config.clip_coef)

                # Average over batch
                pg_loss = batch_pg_loss / bs
                v_loss = 0.5 * batch_v_loss / bs
                ent_loss = batch_ent / bs
                loss = pg_loss - self.config.ent_coef * ent_loss + self.config.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Capture gradient norms BEFORE clipping
                last_grad_norms = self._compute_grad_norms()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(v_loss.item())
                all_ent.append(ent_loss.item())
                all_total.append(loss.item())
                all_clip_frac.append(batch_clip / bs)
                approx_kl_sum += batch_kl / bs
                n_batches += 1

            avg_kl = approx_kl_sum / max(n_batches, 1)
            all_kl.append(avg_kl)
            if self.config.target_kl is not None and avg_kl > self.config.target_kl:
                print(f"    Early stop at epoch {epoch+1}/{self.config.n_epochs} (KL={avg_kl:.4f})")
                break

        self.buffer.ptr = 0

        metrics = {
            "policy_loss": float(np.mean(all_pg_loss)) if all_pg_loss else 0.0,
            "value_loss": float(np.mean(all_v_loss)) if all_v_loss else 0.0,
            "entropy": float(np.mean(all_ent)) if all_ent else 0.0,
            "total_loss": float(np.mean(all_total)) if all_total else 0.0,
            "approx_kl": float(np.mean(all_kl)) if all_kl else 0.0,
            "clip_fraction": float(np.mean(all_clip_frac)) if all_clip_frac else 0.0,
            "epochs_used": epochs_used,
            **last_grad_norms,
        }
        return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(config: PPOConfig, total_timesteps: int, datafile: str):
    print("═══ Railway RL Training (PPO + LGNN + RK4) ═══")
    print(f"Device: {config.device}")
    print(f"Timesteps: {total_timesteps}")
    print()

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger = TrainingLogger()

    # ── Initialise environment ───────────────────────────────────────────
    env = RailwayGymEnv(config, datafile)
    obs_np, _ = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=config.device)

    print(f"  Graph: {env.num_nodes} nodes, {env.num_edges} edges")
    print(f"  Observation: ({env.num_nodes}, {config.input_dim}) raw features per step")
    print(f"  Action: MultiBinary({env.num_edges}) — per-edge signal control")
    print()

    # ── Build actor-critic WITH LGNN backbone ────────────────────────────
    ac = RailwayActorCritic(config, env.num_edges).to(config.device)
    ac.set_edge_index(env.graph_edge_index)

    agent = PPOAgent(config, ac, env.num_nodes, env.num_edges)
    print()

    # ── Training loop ────────────────────────────────────────────────────
    global_step = 0
    start_time = time.time()
    episode_count = 0
    episode_reward = 0.0
    ep_trains_finished = 0
    ep_spads = 0
    best_reward = float('-inf')

    num_updates = max(1, total_timesteps // config.n_steps)

    for update in range(1, num_updates + 1):
        # Anneal learning rate linearly
        frac = 1.0 - (update - 1) / num_updates
        lr_now = frac * config.learning_rate
        for pg in agent.optimizer.param_groups:
            pg["lr"] = lr_now

        done_flag = False
        for step in range(config.n_steps):
            global_step += 1

            with torch.no_grad():
                action, logprob, _, value = ac.get_action_and_value(obs)

            action_np = action.cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done_flag = terminated or truncated
            episode_reward += reward
            logger.log_step_reward(reward)

            # Track sim events for this episode
            ep_trains_finished += info.get("trains_finished", 0)
            ep_spads += info.get("spads", 0)

            agent.buffer.add(obs, action, logprob, reward, float(done_flag), value)
            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=config.device)

            if done_flag:
                episode_count += 1
                elapsed = time.time() - start_time
                sps = global_step / max(elapsed, 1e-6)
                ep_len = getattr(env, '_step_count', 0)

                marker = ""
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    marker = " ★ best"

                logger.log_episode(
                    reward=episode_reward,
                    length=ep_len,
                    global_step=global_step,
                    trains_finished=ep_trains_finished,
                    spads=ep_spads,
                    wall_time=elapsed,
                )

                print(
                    f"  Ep {episode_count:4d} | "
                    f"R={episode_reward:8.2f}{marker} | "
                    f"Steps={ep_len:3d} | "
                    f"Global={global_step:6d} | "
                    f"SPS={sps:.0f} | "
                    f"LR={lr_now:.2e}"
                )
                episode_reward = 0.0
                ep_trains_finished = 0
                ep_spads = 0
                obs_np, _ = env.reset()
                obs = torch.tensor(obs_np, dtype=torch.float32, device=config.device)
                ac.set_edge_index(env.graph_edge_index)

        # PPO update (backprop through LGNN!)
        print(f"  ── PPO Update {update}/{num_updates} ──")
        update_metrics = agent.learn(obs, float(done_flag))
        logger.log_update(update_metrics, global_step, lr_now)

        print(
            f"    pg={update_metrics['policy_loss']:.4f} "
            f"vf={update_metrics['value_loss']:.4f} "
            f"ent={update_metrics['entropy']:.2f} "
            f"kl={update_metrics['approx_kl']:.4f} "
            f"clip={update_metrics['clip_fraction']:.2f} "
            f"‖∇LGNN‖={update_metrics['grad_norm_lgnn']:.3f} "
            f"‖∇heads‖={update_metrics['grad_norm_heads']:.3f}"
        )

    elapsed = time.time() - start_time
    print(f"\n═══ Training Complete ({elapsed:.0f}s) ═══")
    print(f"  Episodes: {episode_count}  |  Best reward: {best_reward:.2f}")

    # Save model
    save_path = os.path.join(_root, "models", "trained_ppo.pt")
    torch.save({
        "actor_critic": ac.state_dict(),
        "config": config,
        "num_edges": env.num_edges,
        "num_nodes": env.num_nodes,
    }, save_path)
    print(f"  Model saved → {save_path}")

    # Save metrics
    metrics_path = os.path.join(_root, "dataset", "training_metrics.json")
    logger.save(metrics_path)

    # Auto-generate plots
    try:
        from visualization.training_plots import generate_training_dashboard
        plot_path = os.path.join(_root, "dataset", "training_dashboard.png")
        generate_training_dashboard(metrics_path, plot_path)
    except Exception as e:
        print(f"  ⚠ Could not generate plots: {e}")

    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO + LGNN Railway Signal Control Training")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default = os.path.join(_root, "data", "processed", "godot_station_graph_with_signals.json")

    parser.add_argument("--datafile", type=str, default=_default)
    parser.add_argument("--timesteps", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--trains", type=int, default=30)
    parser.add_argument("--decision-interval", type=float, default=30.0)
    args = parser.parse_args()

    cfg = PPOConfig(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        num_trains_per_episode=args.trains,
        decision_interval_s=args.decision_interval,
    )
    train(cfg, total_timesteps=args.timesteps, datafile=args.datafile)
