import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torchdiffeq import odeint
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


# Enums for system states
class SignalState(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2


class TrainPriority(Enum):
    FREIGHT = 0
    PASSENGER = 1
    EXPRESS = 2
    EMERGENCY = 3


@dataclass
class TrainProfile:
    train_id: str
    length: float
    weight: float
    max_acceleration: float
    max_braking: float
    priority: TrainPriority
    current_speed: float
    position_block: str
    # --- ADDED FIELDS ---
    status: str
    destination_block: str


@dataclass
class NetworkState:
    current_time: str
    trains: List[TrainProfile]
    block_occupancy: Dict[str, bool]
    signal_states: Dict[str, SignalState]
    scheduled_arrivals: Dict[str, float]  # train_id -> scheduled_time


class LGNN(nn.Module):
    """
    Liquid Graph Neural Network (LGNN) module.

    This class defines the dynamics of a continuous-time graph neural network,
    suitable for use with a Neural Ordinary Differential Equation (NODE) solver.
    It computes the derivative of node features with respect to time (dx/dt),
    which can then be integrated by an external ODE solver.

    The model combines Graph Attention (GAT) and Graph Convolutional (GCN)
    layers to capture graph structure. It uses a gating mechanism to balance
    between a proposed update and a decay term, modulated by a learned
    time constant, inspired by Liquid Time-Constant Networks.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 kappa: float = 0.1,
                 epsilon: float = 1e-3
                 ):
        """
        Initializes the LGNN module.

        Args:
            hidden_dim (int): The dimensionality of the node features.
            num_heads (int, optional): The number of multi-head attentions for
                the GAT layer. Defaults to 4.
            kappa (float, optional): The decay rate constant. Defaults to 0.1.
            epsilon (float, optional): A small constant to ensure the learned
                time constant `tau` is always positive. Defaults to 1e-3.
        """
        super().__init__()

        self.hidden_dims = hidden_dim
        self.num_heads = num_heads
        self.kappa = float(kappa)
        self.epsilon = epsilon
        self.edge_index: Optional[torch.Tensor] = None

        self.gat = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gcn = GCNConv(hidden_dim, hidden_dim, cached=False)

        self.proposal_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.tau_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, x):
        """
        Computes the derivative of the node features with respect to time (dx/dt).

        This function defines the core dynamics of the model. It is intended to be
        called by an ODE solver, which will integrate this derivative over time.

        Args:
            t (torch.Tensor): The current time step. Although not used directly
                in the calculations, it is a required argument for compatibility
                with standard ODE solver interfaces.
            x (torch.Tensor): The node features at time `t`, with shape
                (num_nodes, hidden_dim).

        Returns:
            torch.Tensor: The computed derivative dx/dt, with the same shape as x.

        Raises:
            ValueError: If `self.edge_index` has not been set before this
                function is called.
        """
        if self.edge_index is None:
            raise ValueError("edge_index must be set before calling forward")

        s_gat = self.gat(x, self.edge_index)
        s_gcn = self.gcn(x, self.edge_index)
        s = s_gat + s_gcn

        xs = torch.cat([x, s], dim=-1)

        proposed = self.proposal_net(xs)

        g_raw = self.gate_net(xs)
        g = torch.sigmoid(g_raw)

        decay = -self.kappa * x

        tau_raw = self.tau_net(xs)
        tau = F.softplus(tau_raw) + self.epsilon

        dx_dt = (g * proposed + (1.0 - g) * decay) / tau
        return dx_dt


# 2. Multi-Objective Reward Calculator
class RewardCalculator(nn.Module):
    """
    Calculates the multi-objective reward based on system performance.
    """

    def __init__(self, weights=None):
        super(RewardCalculator, self).__init__()

        # Default weights for the multi-objective function
        if weights is None:
            weights = {
                'punctuality': 1.0,
                'delay': 1.0,
                'throughput': 0.8,
                'utilization': 0.6,
                'journey_time': 0.9
            }

        self.register_buffer('w_punctuality', torch.tensor(weights['punctuality']))
        self.register_buffer('w_delay', torch.tensor(weights['delay']))
        self.register_buffer('w_throughput', torch.tensor(weights['throughput']))
        self.register_buffer('w_utilization', torch.tensor(weights['utilization']))
        self.register_buffer('w_journey_time', torch.tensor(weights['journey_time']))

    def forward(self, metrics):
        """
        Calculate composite reward: R(t) = w_p * Punctuality - w_d * Delay + ...
        """
        reward = (
                self.w_punctuality * metrics['punctuality'] -
                self.w_delay * metrics['delay'] +
                self.w_throughput * metrics['throughput'] +
                self.w_utilization * metrics['utilization'] -
                self.w_journey_time * metrics['journey_time']
        )
        return reward


# 3. Safety Guard Module
class SafetyGuard(nn.Module):
    """
    Ensures all actions satisfy safety constraints before execution.
    """

    def __init__(self):
        super(SafetyGuard, self).__init__()

    def validate_signal_commands(self, signal_commands, block_occupancy):
        """
        Validate that signal commands don't create unsafe conditions.
        """
        safe_commands = {}

        for signal_id, command in signal_commands.items():
            # Rule: Don't set signal to GREEN if next block is occupied
            if command['color'] == SignalState.GREEN:
                # Check if this would create a conflict
                # This is a simplified check - real implementation would be more complex
                safe_commands[signal_id] = command
            else:
                safe_commands[signal_id] = command

        return safe_commands


@dataclass
class RailwayPPOConfig:
    """Hyperparameters for the Railway PPO agent."""
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps to collect per environment before an update
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RailwayActorCritic(nn.Module):
    """
    A multi-headed Actor-Critic network for the railway environment.
    It takes an aggregated graph state and outputs decisions for all four policy areas.
    """

    def __init__(self, hidden_dim, num_signal_states, max_path_length):
        super().__init__()

        # Shared body to process the aggregated graph state
        self.shared_body = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Critic head: outputs a single state value
        self.critic_head = nn.Linear(256, 1)

        # --- Actor Heads (one for each action type) ---
        # 1. Signal Color (Discrete)
        self.signal_head = nn.Linear(256, num_signal_states)

        # 2. Signal Timing (Continuous) - outputs mean of a distribution
        self.timing_head = nn.Linear(256, 1)

        # 3. Path Assignment (Discrete)
        self.path_head = nn.Linear(256, max_path_length)

        # 4. Trajectory Control (2 Continuous actions) - outputs means for accel/braking
        self.trajectory_head = nn.Linear(256, 2)

        # Learnable standard deviations for continuous actions
        self.timing_log_std = nn.Parameter(torch.zeros(1))
        self.trajectory_log_std = nn.Parameter(torch.zeros(2))

    def get_value(self, graph_state):
        """Returns the state value V(s) from the critic."""
        return self.critic_head(self.shared_body(graph_state))

    def get_actions_and_value(self, graph_state, actions=None):
        """
        Returns actions, their log probabilities, entropy, and the state value.
        If actions are not provided, they are sampled from the policies.
        """
        shared_features = self.shared_body(graph_state)
        value = self.critic_head(shared_features)

        # --- Sample or evaluate actions for each head ---

        # 1. Signal Color
        signal_logits = self.signal_head(shared_features)
        signal_probs = Categorical(logits=signal_logits)
        if actions is None or 'signal' not in actions:
            signal_action = signal_probs.sample()
        else:
            signal_action = actions['signal']

        # 2. Signal Timing
        timing_mean = self.timing_head(shared_features)
        timing_std = torch.exp(self.timing_log_std)
        timing_probs = Normal(timing_mean, timing_std)
        if actions is None or 'timing' not in actions:
            timing_action = timing_probs.sample()
        else:
            timing_action = actions['timing']

        # 3. Path Assignment
        path_logits = self.path_head(shared_features)
        path_probs = Categorical(logits=path_logits)
        if actions is None or 'path' not in actions:
            path_action = path_probs.sample()
        else:
            path_action = actions['path']

        # 4. Trajectory Control
        trajectory_mean = self.trajectory_head(shared_features)
        trajectory_std = torch.exp(self.trajectory_log_std)
        trajectory_probs = Normal(trajectory_mean, trajectory_std)
        if actions is None or 'trajectory' not in actions:
            trajectory_action = trajectory_probs.sample()
        else:
            trajectory_action = actions['trajectory']

        # --- Collect results ---
        log_probs = {
            'signal': signal_probs.log_prob(signal_action),
            'timing': timing_probs.log_prob(timing_action).sum(1),
            'path': path_probs.log_prob(path_action),
            'trajectory': trajectory_probs.log_prob(trajectory_action).sum(1),
        }

        entropies = {
            'signal': signal_probs.entropy(),
            'timing': timing_probs.entropy().sum(1),
            'path': path_probs.entropy(),
            'trajectory': trajectory_probs.entropy().sum(1),
        }

        sampled_actions = {
            'signal': signal_action, 'timing': timing_action, 'path': path_action, 'trajectory': trajectory_action
        }

        return sampled_actions, log_probs, entropies, value


class RailwayRolloutBuffer:
    """Stores trajectories and calculates advantages for the multi-action railway agent."""

    def __init__(self, n_steps, hidden_dim, device):
        self.n_steps = n_steps
        self.device = device
        self.hidden_dim = hidden_dim
        self.ptr = 0

        # State and general RL data
        self.obs = torch.zeros((n_steps, hidden_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_steps,), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps,), dtype=torch.float32, device=device)

        # Action-specific data
        self.signal_actions = torch.zeros((n_steps,), dtype=torch.int64, device=device)
        self.timing_actions = torch.zeros((n_steps, 1), dtype=torch.float32, device=device)
        self.path_actions = torch.zeros((n_steps,), dtype=torch.int64, device=device)
        self.trajectory_actions = torch.zeros((n_steps, 2), dtype=torch.float32, device=device)

        self.logprobs = {
            'signal': torch.zeros((n_steps,), dtype=torch.float32, device=device),
            'timing': torch.zeros((n_steps,), dtype=torch.float32, device=device),
            'path': torch.zeros((n_steps,), dtype=torch.float32, device=device),
            'trajectory': torch.zeros((n_steps,), dtype=torch.float32, device=device),
        }

    def add(self, obs, actions, logprobs, reward, done, value):
        """Add a new transition to the buffer."""
        self.obs[self.ptr] = obs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.flatten()

        self.signal_actions[self.ptr] = actions['signal']
        self.timing_actions[self.ptr] = actions['timing']
        self.path_actions[self.ptr] = actions['path']
        self.trajectory_actions[self.ptr] = actions['trajectory']

        for key in self.logprobs:
            self.logprobs[key][self.ptr] = logprobs[key]

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, done, gamma, gae_lambda):
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            nextnonterminal = 1.0 - self.dones[t + 1] if t < self.n_steps - 1 else 1.0 - done
            nextvalues = self.values[t + 1] if t < self.n_steps - 1 else last_value
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for start in range(0, self.n_steps, batch_size):
            batch_indices = indices[start: start + batch_size]
            yield (
                self.obs[batch_indices],
                {
                    'signal': self.signal_actions[batch_indices],
                    'timing': self.timing_actions[batch_indices],
                    'path': self.path_actions[batch_indices],
                    'trajectory': self.trajectory_actions[batch_indices],
                },
                {key: logprob[batch_indices] for key, logprob in self.logprobs.items()},
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )


class RailwayPPOAgent:
    """The PPO Agent that orchestrates the training of the RailwayActorCritic."""

    def __init__(self, config, hidden_dim, num_signal_states, max_path_length):
        self.config = config
        self.device = config.device
        self.agent = RailwayActorCritic(hidden_dim, num_signal_states, max_path_length).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RailwayRolloutBuffer(config.n_steps, hidden_dim, self.device)

    def learn(self, next_obs, done):
        """Perform the PPO update step for the multi-action agent."""
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            self.buffer.compute_returns_and_advantages(next_value, done, self.config.gamma, self.config.gae_lambda)

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs_batch, actions_batch, old_logprobs_batch, advantages_batch, returns_batch = batch

                _, new_logprobs, entropies, new_value = self.agent.get_actions_and_value(obs_batch, actions_batch)

                total_pg_loss = 0
                total_entropy_loss = 0

                # Calculate losses for each action head
                for key in new_logprobs:
                    logratio = new_logprobs[key] - old_logprobs_batch[key]
                    ratio = logratio.exp()

                    # Policy loss
                    pg_loss1 = -advantages_batch * ratio
                    pg_loss2 = -advantages_batch * torch.clamp(ratio, 1 - self.config.clip_coef,
                                                               1 + self.config.clip_coef)
                    total_pg_loss += torch.max(pg_loss1, pg_loss2).mean()

                    # Entropy loss
                    total_entropy_loss += entropies[key].mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - returns_batch) ** 2).mean()

                # Total loss
                loss = total_pg_loss - self.config.ent_coef * total_entropy_loss + self.config.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

        self.buffer.ptr = 0  # Clear buffer


# --- Main Liquid Railway Control System - NOW WITH PPO ---
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import time
import json
import argparse
import numpy as np
from pathlib import Path


# --- NOTE: Unchanged classes are omitted for brevity ---
# (SignalState, TrainPriority, TrainProfile, NetworkState, LGNN, RewardCalculator,
#  SafetyGuard, RailwayPPOConfig, RailwayActorCritic, RailwayRolloutBuffer, RailwayPPOAgent)
#  Please include them from the previous response in the final file.

# --- NEW: SCADA-like Simulation Environment ---

class LiquidNetworkEnv(gym.Env):
    """
    A Gymnasium environment that wraps the LiquidRailwayControlSystem.
    It simulates the railway network dynamics in response to the agent's actions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # The environment contains the 'brain' or 'controller'
        self.model = LiquidRailwayControlSystem().to(config.device)

        # The agent's observation is the aggregated state vector of the railway graph
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.model.hidden_dim,), dtype=np.float32
        )

        # The action is a trigger; the complex multi-part action is handled internally
        self.action_space = spaces.Discrete(1)

        self.current_step = 0
        self.max_episode_steps = 500  # Episode truncates after this many steps

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None): # type: ignore
        super().reset(seed=seed, options=options)
        self.current_step = 0

        # Create a new, randomized initial state for the railway network
        self.network_state, self.edge_index = self._create_initial_state()

        # Get the initial observation vector from the model
        obs = self.model.get_aggregated_state(self.network_state, self.edge_index)

        # --- FIX: Use .detach() before converting to numpy ---
        return obs.detach().cpu().numpy(), self._get_info()

    def step(self, action, mode='train'):
        """
        Run one timestep of the environment's dynamics using the agent's action.
        """
        # 1. Get the current state observation vector
        current_obs_tensor = self.model.get_aggregated_state(self.network_state, self.edge_index)

        # 2. Get the multi-part action from the internal PPO agent
        actions, log_probs, value = self.model.get_actions(current_obs_tensor, mode=mode)

        # 3. Simulate the effect of the action on the network state
        self.network_state = self._simulate_step(self.network_state, actions)

        # 4. Calculate the reward for the new state
        reward_tensor = self.model.calculate_reward(self.network_state, self.edge_index)
        reward = reward_tensor.item()

        # 5. Check for termination or truncation
        self.current_step += 1
        terminated = all(t.status == "arrived" for t in self.network_state.trains)
        truncated = self.current_step >= self.max_episode_steps
        done = terminated or truncated

        # 6. If in training mode, store the experience
        if mode == 'train':
            # Detach tensors to prevent gradients from flowing back during storage
            self.model.store_experience(
                current_obs_tensor.detach(), actions, log_probs, reward, done, value.detach()
            )
            # Trigger the PPO learning phase if the buffer is full
            self.model.learn_if_ready(current_obs_tensor, done)

        # 7. Get the next observation
        next_obs_tensor = self.model.get_aggregated_state(self.network_state, self.edge_index)

        # --- FIX: Use .detach() before converting to numpy ---
        return next_obs_tensor.detach().cpu().numpy(), reward, terminated, truncated, self._get_info()

    def _simulate_step(self, state, actions):
        """Placeholder for railway physics simulation."""
        # This is where you would update train speeds and positions based on
        # the agent's acceleration/braking commands.
        for train in state.trains:
            if train.status != "arrived":
                # Apply acceleration/braking (simplified)
                accel, brake = torch.tanh(actions['trajectory']).cpu().numpy()
                train.current_speed += (accel * train.max_acceleration) - (brake * train.max_braking)
                train.current_speed = max(0, train.current_speed)
        return state

    def _create_initial_state(self):
        """Creates a sample initial NetworkState and topology."""
        trains = [
            TrainProfile("T101", 120, 800, 1.2, 2.5, TrainPriority.PASSENGER, 0, "B2", "arriving", "P10"),
            TrainProfile("T202", 200, 1200, 0.8, 1.8, TrainPriority.FREIGHT, 0, "B5", "arriving", "P12"),
        ]
        block_occupancy = {"B2": True, "B5": True}
        network_state = NetworkState("t0", trains, block_occupancy, {}, {})

        num_nodes = 20
        edges = [[i, i + 1] for i in range(num_nodes - 1)] + [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.config.device)

        return network_state, edge_index

    def _get_info(self):
        return {"current_step": self.current_step, "trains": self.network_state.trains}

    def close(self):
        print("Environment closed.")


# --- MODIFIED: Main Liquid Railway Control System ---
# Now refactored to support the Gym environment and external training loops

class LiquidRailwayControlSystem(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_signal_states=3, max_path_length=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.state_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.liquid_ode = LGNN(hidden_dim)

        self.ppo_config = RailwayPPOConfig()
        self.ppo_agent = RailwayPPOAgent(self.ppo_config, hidden_dim, num_signal_states, max_path_length)

        self.reward_calculator = RewardCalculator()
        self.safety_guard = SafetyGuard()
        self.global_step = 0

    def get_aggregated_state(self, network_state: NetworkState, edge_index: torch.Tensor):
        """Encodes the network and returns the single aggregated state vector."""
        graph_data = self._encode_network_state(network_state, edge_index)
        h0 = self.state_encoder(graph_data.x)
        self.liquid_ode.edge_index = edge_index
        t = torch.tensor([0.0, 30.0], device=h0.device)  # 30s time horizon
        h_final = odeint(self.liquid_ode, h0, t, method='dopri5')[1]
        return h_final.mean(dim=0) # type: ignore

    def get_actions(self, aggregated_state, mode='train'):
        """Gets actions from the PPO agent's actor-critic network."""
        with torch.no_grad():
            actions, log_probs, _, value = self.ppo_agent.agent.get_actions_and_value(
                aggregated_state.unsqueeze(0)
            )

        # Remove batch dimension for single-environment processing
        for k in actions: actions[k] = actions[k].squeeze(0)
        for k in log_probs: log_probs[k] = log_probs[k].squeeze(0)
        value = value.squeeze(0)
        return actions, log_probs, value

    def store_experience(self, obs, actions, log_probs, reward, done, value):
        self.global_step += 1
        self.ppo_agent.buffer.add(obs, actions, log_probs, reward, done, value)

    def learn_if_ready(self, next_obs, done):
        """Checks if the buffer is full and triggers a PPO update."""
        if self.global_step > 0 and self.global_step % self.ppo_config.n_steps == 0:
            print(f"--- Performing PPO update at global step {self.global_step} ---")
            self.ppo_agent.learn(next_obs.unsqueeze(0), done)
            return True
        return False

    def calculate_reward(self, network_state, edge_index):
        h_final = self.get_aggregated_state(network_state, edge_index)
        metrics = self._calculate_metrics_from_state(h_final)
        return self.reward_calculator(metrics)

    # Helper methods _encode_network_state and _calculate_metrics_from_state remain the same
    def _encode_network_state(self, network_state: NetworkState, edge_index: torch.Tensor):
        num_nodes = int(edge_index.max().item()) + 1
        node_features = torch.zeros(num_nodes, int(self.input_dim), device=self.ppo_config.device)
        for i in range(num_nodes):
            block_id = f"B{i}"
            node_features[i, 0] = 1.0 if network_state.block_occupancy.get(block_id, False) else 0.0
        return Data(x=node_features, edge_index=edge_index)

    def _calculate_metrics_from_state(self, hidden_state):
        # FIX: The 'hidden_state' argument is already the aggregated state vector.
        # We should index it directly instead of calculating its mean again.
        return {
            'punctuality': torch.sigmoid(hidden_state[0]), 'delay': torch.relu(hidden_state[1]),
            'throughput': torch.sigmoid(hidden_state[2]), 'utilization': torch.sigmoid(hidden_state[3]),
            'journey_time': torch.relu(hidden_state[4]),
        }


# --- NEW: Main Training and Testing Functions ---

def train(config, total_timesteps=100_000):
    """Main training loop for the PPO agent with improved feedback."""
    print("--- Starting Training ---")
    env = LiquidNetworkEnv(config)
    agent_model = env.model  # The underlying LiquidRailwayControlSystem

    # --- NEW: Variables for improved logging ---
    episode_rewards = []
    current_episode_reward = 0.0

    obs, info = env.reset()
    start_time = time.time()

    print(f"Starting training for {total_timesteps} timesteps...")

    for global_step in range(total_timesteps):
        # The environment step handles action selection, simulation, and learning triggers
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), mode='train')
        print(global_step)

        current_episode_reward += reward

        if terminated or truncated:
            # --- NEW: Log episode completion ---
            print(f"Global Step: {global_step + 1} | Episode Finished | Reward: {current_episode_reward:.2f}")
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0  # Reset for next episode

            # Also log mean reward periodically
            if len(episode_rewards) % 10 == 0:
                mean_reward = np.mean(episode_rewards[-10:])
                print(f"-------------------------------------------------------")
                print(f"Mean reward of last 10 episodes: {mean_reward:.2f}")
                print(f"-------------------------------------------------------")

            obs, info = env.reset()

        # --- MODIFIED: More informative periodic log ---
        # This log now complements the PPO update log
        if (global_step + 1) % config.n_steps == 0:
            sps = int((global_step + 1) / (time.time() - start_time + 1e-8))
            mean_reward_str = f"{np.mean(episode_rewards[-10:]):.2f}" if episode_rewards else "N/A"
            print(
                f"Global Step: {global_step + 1}/{total_timesteps} | SPS: {sps} | Mean Reward (last 10): {mean_reward_str}")

    # Save the final trained model
    model_path = "railway_ppo_agent.pth"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent_model.ppo_agent.agent.state_dict(), model_path)
    print(f"--- Training Finished. Model saved to {model_path} ---")
    env.close()


def test(config, num_test_episodes=10):
    """Main testing loop for the trained PPO agent."""
    print("--- Starting Testing ---")
    model_path = "railway_ppo_agent.pth"

    env = LiquidNetworkEnv(config)
    agent_model = env.model

    try:
        agent_model.ppo_agent.agent.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Cannot run test.")
        env.close()
        return

    agent_model.ppo_agent.agent.eval()  # Set agent to evaluation mode
    total_rewards = []

    for episode in range(num_test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Pass 'inference' mode to select deterministic actions
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), mode='inference')
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_test_episodes}, Reward: {episode_reward:.2f}")

    print(f"\nTesting finished. Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Agent for Railway Control")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Set the agent to training or testing mode.')
    args = parser.parse_args()

    config = RailwayPPOConfig()

    if args.mode == 'train':
        train(config, total_timesteps=10000)  # Run a shorter training for demonstration
    else:
        test(config)