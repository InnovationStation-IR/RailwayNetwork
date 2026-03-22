#!/usr/bin/env python3
"""
Enhanced Railway Simulation System

A comprehensive, production-ready railway simulation system with advanced features:
- Modular architecture with clear separation of concerns
- Advanced train scheduling and collision detection
- Real-time metrics and analytics
- Enhanced visualization with debugging tools
- AI integration capabilities
- Robust error handling and validation
- Comprehensive testing framework

Requirements:
    pip install simpy networkx pygame pandas pyarrow h5py numpy matplotlib seaborn
    Optional: torch torch-geometric gymnasium stable-baselines3
"""

import argparse
import json
import os
import random
import time
import datetime
import multiprocessing as mp
import logging
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Core simulation libs
import simpy
import networkx as nx
from simpy.resources.resource import Request

# Import canonical signal types from single source of truth
from simulation.signals import (
    SignalAspect as IRSignalAspect,
    SignalType as IRSignalType,
    Signal as IRSignal,
    BlockSection,
    CascadeSignalController,
    SPADDetector,
    build_signals_and_blocks_from_graph,
)
from simulation.disruptions import DisruptionManager, DisruptionConfig

# Type Aliases for clarity
SimPyEnv = simpy.Environment
NetworkXGraph = nx.Graph

# Optional libraries with graceful fallbacks
try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    warnings.warn("Pygame not available. Visualization disabled.")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Advanced analytics disabled.")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some optimizations disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting disabled.")

# AI/ML libraries (optional)
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gymnasium as gym

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


# ---------------------------
# Part 1: Enhanced Configuration & Constants
# ---------------------------

class SimulationMode(Enum):
    """Simulation execution modes"""
    HEADLESS = "headless"
    VISUAL = "visual"
    HYBRID = "hybrid"  # Visual with headless fallback


class TrainType(Enum):
    """Train types with different characteristics"""
    PASSENGER = "passenger"
    FREIGHT = "freight"
    EXPRESS = "express"
    LOCAL = "local"
    MAINTENANCE = "maintenance"


class SignalAspect(Enum):
    """Signal aspects"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"


class Priority(Enum):
    """Priority levels for trains"""
    EMERGENCY = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# Enhanced configuration with validation
@dataclass
class SimulationConfig:
    """Comprehensive simulation configuration with validation"""
    # Core simulation settings
    simulation_speed_factor: float = 150.0
    train_speed_pps: float = 12.0
    initial_trains: int = 20
    snapshot_interval: float = 1.0
    episode_duration: float = 3600.0
    seed: Optional[int] = None

    # Train settings
    max_trains: int = 100
    train_spawn_rate: float = 0.1  # trains per second
    default_route_length: int = 8

    # Safety settings
    collision_detection: bool = False
    safety_margin: float = 100.0  # meters (increased)
    emergency_brake_distance: float = 200.0  # meters
    # REFACTOR: Added magic numbers from code as config options
    train_deceleration: float = 2.0  # m/s^2, used for safety calcs
    emergency_stop_time: float = 2.0  # s, time train is stopped after emergency
    collision_critical_distance: float = 5.0  # meters, for collision detection
    collision_warning_distance: float = 20.0  # meters, for collision risk

    # Disruption settings
    disruption_rate: float = 0.05
    mean_disruption_duration: float = 120.0

    # Performance settings
    max_memory_usage_mb: int = 1024
    enable_multiprocessing: bool = True
    max_workers: int = 4

    # Visualization settings
    screen_width: int = 1280
    screen_height: int = 720
    fps: int = 60
    enable_animations: bool = True
    show_debug_info: bool = False
    # REFACTOR: Added magic number as config option
    visual_update_step: float = 0.1  # s, time step for visual train movement

    # Analytics settings
    enable_metrics: bool = True
    metrics_update_interval: float = 5.0
    export_format: str = "jsonl"  # jsonl, parquet, hdf5

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.simulation_speed_factor <= 0:
            raise ValueError("simulation_speed_factor must be positive")
        if self.train_speed_pps <= 0:
            raise ValueError("train_speed_pps must be positive")
        if self.max_trains <= 0:
            raise ValueError("max_trains must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        # REFACTOR: Added validation for new config options
        if self.train_deceleration <= 0:
            raise ValueError("train_deceleration must be positive")
        if self.visual_update_step <= 0:
            raise ValueError("visual_update_step must be positive")
        if self.collision_critical_distance < 0:
            raise ValueError("collision_critical_distance must be non-negative")


# Enhanced color scheme with better contrast and accessibility
class Colors:
    """Enhanced color palette with better contrast and accessibility"""
    # Background colors
    BACKGROUND = (12, 24, 35)
    BACKGROUND_LIGHT = (20, 35, 50)

    # Track colors
    TRACK_MAIN = (85, 170, 85)
    TRACK_SECONDARY = (120, 120, 120)
    TRACK_DISRUPTED = (255, 190, 0)
    TRACK_OCCUPIED = (255, 100, 100)

    # Station colors
    STATION_FILL = (255, 245, 220)
    STATION_BORDER = (70, 130, 180)
    STATION_SELECTED = (255, 200, 0)

    # Junction colors
    JUNCTION_FILL = (200, 200, 200)
    JUNCTION_BORDER = (100, 100, 100)

    # Train colors by type
    TRAIN_PASSENGER = (255, 100, 80)
    TRAIN_FREIGHT = (150, 150, 150)
    TRAIN_EXPRESS = (255, 200, 0)
    TRAIN_LOCAL = (100, 200, 255)
    TRAIN_MAINTENANCE = (255, 150, 0)

    # Signal colors
    SIGNAL_RED = (255, 60, 60)
    SIGNAL_YELLOW = (255, 220, 0)
    SIGNAL_GREEN = (60, 255, 60)
    SIGNAL_OFF = (60, 60, 60)

    # UI colors
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (200, 200, 200)
    TEXT_WARNING = (255, 200, 0)
    TEXT_ERROR = (255, 100, 100)

    # Grid and debug
    GRID = (40, 60, 80)
    DEBUG_LINE = (255, 255, 0)

    @classmethod
    def get_train_color(cls, train_type: TrainType) -> Tuple[int, int, int]:
        """Get color for train type"""
        color_map = {
            TrainType.PASSENGER: cls.TRAIN_PASSENGER,
            TrainType.FREIGHT: cls.TRAIN_FREIGHT,
            TrainType.EXPRESS: cls.TRAIN_EXPRESS,
            TrainType.LOCAL: cls.TRAIN_LOCAL,
            TrainType.MAINTENANCE: cls.TRAIN_MAINTENANCE,
        }
        return color_map.get(train_type, cls.TRAIN_PASSENGER)


# ---------------------------
# Part 2: Enhanced Data Structures
# ---------------------------

@dataclass
class ODPair:
    """Origin-Destination pair with enhanced metadata"""
    origin: str
    destination: str
    depart_time: float
    priority: Priority = Priority.NORMAL
    train_type: TrainType = TrainType.PASSENGER
    passenger_count: int = 0
    cargo_weight: float = 0.0
    max_speed: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainMetrics:
    """Comprehensive train performance metrics"""
    total_distance: float = 0.0
    total_time: float = 0.0
    wait_time: float = 0.0
    delay_time: float = 0.0
    stops_count: int = 0
    collisions_count: int = 0
    energy_consumed: float = 0.0
    passenger_satisfaction: float = 1.0

    @property
    def average_speed(self) -> float:
        # REFACTOR: Replaced max(..., 0.001) with a robust check
        if self.total_time == 0:
            return 0.0
        return self.total_distance / self.total_time

    @property
    def efficiency(self) -> float:
        """Efficiency metric combining speed and satisfaction"""
        # REFACTOR: Replaced max(..., 1) with a robust check
        wait_component = self.wait_time + self.delay_time
        if wait_component == 0:
            return self.average_speed * self.passenger_satisfaction
        return (self.average_speed * self.passenger_satisfaction) / wait_component


@dataclass
class NetworkMetrics:
    """Network-wide performance metrics"""
    total_throughput: float = 0.0
    average_delay: float = 0.0
    utilization_rate: float = 0.0
    collision_rate: float = 0.0
    energy_efficiency: float = 0.0
    passenger_satisfaction: float = 1.0
    timestamp: float = 0.0


# ---------------------------
# Part 3: Enhanced Data Recorder
# ---------------------------

class EnhancedDataRecorder:
    """Advanced data recording with multiple export formats and real-time analytics"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.snapshot_interval = config.snapshot_interval
        self.snapshots: List[Tuple[float, Dict[str, Any]]] = []
        self.events: List[Dict[str, Any]] = []
        self.metrics_history: List[NetworkMetrics] = []
        self._last_snapshot_time: Optional[float] = None
        self._last_metrics_time: Optional[float] = None

        # Performance optimization
        self._snapshot_buffer = deque(maxlen=1000)
        self._event_buffer = deque(maxlen=10000)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('RailwaySimulation')

    def record_event(self, sim_time: float, kind: str, payload: Dict[str, Any]):
        """Record simulation event with enhanced metadata"""
        event = {
            "time": float(sim_time),
            "kind": kind,
            "payload": deepcopy(payload),
            "timestamp": datetime.datetime.now().isoformat(),
            "event_id": f"{kind}_{int(sim_time * 1000)}_{len(self.events)}"
        }
        self._event_buffer.append(event)
        self.events.append(event)

        # Log important events
        if kind in ['collision', 'disruption_start', 'emergency_brake']:
            self.logger.warning(f"Critical event: {kind} at {sim_time:.2f}s")

    def maybe_snapshot(self, sim_time: float, simulation: 'EnhancedSimulation'):
        """Take snapshot if interval has passed"""
        if self._last_snapshot_time is None:
            self._last_snapshot_time = sim_time - self.snapshot_interval

        if sim_time >= self._last_snapshot_time + self.snapshot_interval - 1e-9:
            snap = self._make_snapshot(sim_time, simulation)
            self._snapshot_buffer.append((float(sim_time), snap))
            self.snapshots.append((float(sim_time), snap))
            self._last_snapshot_time = sim_time

    def maybe_update_metrics(self, sim_time: float, simulation: 'EnhancedSimulation'):
        """Update metrics if interval has passed"""
        if not self.config.enable_metrics:
            return

        if self._last_metrics_time is None:
            self._last_metrics_time = sim_time - self.config.metrics_update_interval

        if sim_time >= self._last_metrics_time + self.config.metrics_update_interval - 1e-9:
            metrics = self._calculate_network_metrics(sim_time, simulation)
            self.metrics_history.append(metrics)
            self._last_metrics_time = sim_time

    def _make_snapshot(self, sim_time: float, simulation: 'EnhancedSimulation') -> Dict[str, Any]:
        """Create comprehensive snapshot with enhanced data"""
        G = simulation.graph

        # Enhanced node data
        nodes = {}
        for nid, data in G.nodes(data=True):
            nodes[nid] = {
                "pos": data.get("pos"),
                "label": data.get("label"),
                "is_station": bool(data.get("label")),
                "capacity": data.get("capacity", 0),
                "utilization": simulation.get_node_utilization(nid),
                "trains_at_node": len([t for t in simulation.trains if t.is_at_node(nid)])
            }

        # Enhanced edge data
        edges = {}
        for u, v, d in G.edges(data=True):
            key = tuple(sorted((u, v)))
            block = simulation.track_blocks.get(key)
            edges[str(key)] = {
                "u": u, "v": v,
                "distance": d.get("distance", 0.0),
                # REFACTOR: Uses the new is_occupied property
                "occupied": bool(block.is_occupied) if block else False,
                "resource_count": getattr(block.resource, "count", 0) if block else 0,
                "disrupted": getattr(block, "_disrupted", False),
                "utilization": simulation.get_edge_utilization(key),
                # REFACTOR: Uses the new helper function for efficiency
                "trains_on_edge": len(simulation.get_trains_on_edge(key))
            }

        # Enhanced signal data
        signals = {}
        for s in simulation.signals:
            signals[s.id] = {
                "aspect": s.aspect.value if hasattr(s.aspect, 'value') else s.aspect,
                "pos": s.pos,
                "protected_block": s.protected_block_key,
                "last_change": getattr(s, 'last_change_time', sim_time),
                "change_count": getattr(s, 'change_count', 0)
            }

        # Enhanced train data
        trains = []
        for t in simulation.trains:
            train_data = {
                "id": t.id,
                "pos": t.visual_pos,
                "route": list(t.route),
                "is_visible": t.is_visible,
                "current_index": getattr(t, "current_index", None),
                "train_type": t.train_type.value if hasattr(t.train_type, 'value') else t.train_type,
                "priority": t.priority.value if hasattr(t.priority, 'value') else t.priority,
                "speed": getattr(t, 'current_speed', 0),
                "metrics": t.metrics.__dict__ if hasattr(t, 'metrics') else {}
            }
            trains.append(train_data)

        # Network metadata
        meta = {
            "sim_time": float(sim_time),
            "active_trains": sum(1 for t in simulation.trains if t.is_visible),
            "total_trains": len(simulation.trains),
            "network_utilization": simulation.get_network_utilization(),
            "collision_count": simulation.collision_count,
            "disruption_count": simulation.disruption_count,
            "energy_consumed": simulation.total_energy_consumed
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "signals": signals,
            "trains": trains,
            "meta": meta
        }

    def _calculate_network_metrics(self, sim_time: float, simulation: 'EnhancedSimulation') -> NetworkMetrics:
        """Calculate comprehensive network metrics"""
        active_trains = [t for t in simulation.trains if t.is_visible]
        all_trains = simulation.trains

        if not all_trains:
            return NetworkMetrics(timestamp=sim_time)

        # Calculate various metrics
        total_delay_time = sum(t.metrics.delay_time for t in all_trains)
        total_passengers = sum(getattr(t.od_pair, 'passenger_count', 0) for t in all_trains)

        # REFACTOR: Use robust division
        avg_delay = total_delay_time / len(all_trains) if all_trains else 0
        collision_rate = simulation.collision_count / sim_time if sim_time > 0 else 0

        # Throughput = passengers * km
        total_passenger_km = sum(
            t.metrics.total_distance * getattr(t.od_pair, 'passenger_count', 0) for t in all_trains)
        throughput = total_passenger_km / sim_time if sim_time > 0 else 0

        return NetworkMetrics(
            total_throughput=throughput,
            average_delay=avg_delay,
            utilization_rate=simulation.get_network_utilization(),
            collision_rate=collision_rate,
            energy_efficiency=simulation.get_energy_efficiency(),
            passenger_satisfaction=simulation.get_passenger_satisfaction(),
            timestamp=sim_time
        )

    def export_data(self, out_dir: str = "dataset", episode_name: str = "ep") -> Dict[str, str]:
        """Export data in multiple formats"""
        os.makedirs(out_dir, exist_ok=True)
        exported_files = {}

        # Export snapshots
        snaps_f = os.path.join(out_dir, f"{episode_name}_snapshots.jsonl")
        with open(snaps_f, 'w') as sf:
            for t, s in self.snapshots:
                sf.write(json.dumps({"time": t, "snapshot": s}) + "\n")
        exported_files['snapshots'] = snaps_f

        # Export events
        events_f = os.path.join(out_dir, f"{episode_name}_events.jsonl")
        with open(events_f, 'w') as ef:
            for e in self.events:
                ef.write(json.dumps(e))
        exported_files['events'] = events_f

        # Export metrics
        if self.metrics_history:
            metrics_f = os.path.join(out_dir, f"{episode_name}_metrics.json")
            with open(metrics_f, 'w') as mf:
                json.dump([m.__dict__ for m in self.metrics_history], mf, indent=2)
            exported_files['metrics'] = metrics_f

        # Export in other formats if available
        if PANDAS_AVAILABLE and self.config.export_format == "parquet":
            self._export_parquet(out_dir, episode_name, exported_files)

        return exported_files

    def _export_parquet(self, out_dir: str, episode_name: str, exported_files: Dict[str, str]):
        """Export data in Parquet format for better performance"""
        try:
            # Convert snapshots to DataFrame
            snapshot_data = []
            for t, s in self.snapshots:
                snapshot_data.append({
                    'time': t,
                    'active_trains': s['meta']['active_trains'],
                    'network_utilization': s['meta']['network_utilization'],
                    'collision_count': s['meta']['collision_count']
                })

            if snapshot_data:
                df_snapshots = pd.DataFrame(snapshot_data)
                parquet_f = os.path.join(out_dir, f"{episode_name}_snapshots.parquet")
                df_snapshots.to_parquet(parquet_f, engine='pyarrow')
                exported_files['snapshots_parquet'] = parquet_f

            # Convert events to DataFrame
            if self.events:
                df_events = pd.DataFrame(self.events)
                parquet_f = os.path.join(out_dir, f"{episode_name}_events.parquet")
                df_events.to_parquet(parquet_f, engine='pyarrow')
                exported_files['events_parquet'] = parquet_f

        except Exception as e:
            self.logger.warning(f"Failed to export Parquet format: {e}")


# ---------------------------
# Part 4: Enhanced Simulation Entities
# ---------------------------

class EnhancedSignal:
    """Enhanced signal with better state management and timing"""

    def __init__(self, data: Dict[str, Any]):
        self.id: str = data['id']
        self.pos: Tuple[float, float] = (data['position']['x'], data['position']['y'])
        self.aspect: SignalAspect = SignalAspect(data.get('initial_aspect', 'green'))
        self.last_change_time: float = 0.0
        self.change_count: int = 0

        # Parse protected block from signal ID
        parts = self.id.split('_')
        self.protected_block_key: Optional[Tuple[str, str]] = (
            tuple(sorted((parts[1], parts[3]))) if len(parts) >= 4 else None
        )

        # Signal timing constraints
        self.min_green_time: float = 5.0  # Minimum green time
        self.min_red_time: float = 3.0  # Minimum red time
        self.last_aspect_change: float = 0.0

    def change_aspect(self, new_aspect: SignalAspect, sim_time: float) -> bool:
        """Change signal aspect with timing constraints"""
        if self.aspect == new_aspect:
            return False

        # Check timing constraints
        time_since_last_change = sim_time - self.last_aspect_change
        if self.aspect == SignalAspect.GREEN and time_since_last_change < self.min_green_time:
            return False
        if self.aspect == SignalAspect.RED and time_since_last_change < self.min_red_time:
            return False

        self.aspect = new_aspect
        self.last_change_time = sim_time
        self.last_aspect_change = sim_time
        self.change_count += 1
        return True


class EnhancedTrackBlock:
    """Enhanced track block with better resource management and safety features"""

    def __init__(self, env: SimPyEnv, id_tuple: Tuple[str, str], config: SimulationConfig):
        self.id = id_tuple
        self.env = env
        self.config = config

        # Use PriorityResource for train precedence
        self.resource = simpy.PriorityResource(env, capacity=1)
        # REFACTOR: Removed self.is_occupied, it's redundant.
        self._disrupted: bool = False

        # Enhanced safety features
        self.safety_margin: float = config.safety_margin
        self.emergency_brake_distance: float = config.emergency_brake_distance

        # Performance tracking
        self.utilization_history: deque = deque(maxlen=1000)
        self.last_occupancy_time: float = 0.0
        self.total_occupancy_time: float = 0.0

    # REFACTOR: New property to get occupation status directly from the resource.
    @property
    def is_occupied(self) -> bool:
        """The block is occupied if its resource is in use."""
        return self.resource.count > 0

    def request_access(self, train_id: str, priority: Priority) -> Request:
        """Request access to track block with priority"""
        return self.resource.request(priority=priority.value)

    def release_access(self, request: Request):
        """Release access to track block"""
        self.resource.release(request)
        # REFACTOR: No need to manually set self.is_occupied = False

    def is_safe_for_entry(self, train_speed: float, distance_to_block: float) -> bool:
        """Check if it's safe for a train to enter the block"""
        if self.is_occupied or self._disrupted:
            return False

        # Check if train can stop safely
        # REFACTOR: Use deceleration from config
        deceleration = self.config.train_deceleration
        required_stopping_distance = (train_speed ** 2) / (2 * deceleration)
        return distance_to_block >= required_stopping_distance + self.safety_margin


class EnhancedTrain:
    """Enhanced train with comprehensive features and safety systems"""

    def __init__(self, env: SimPyEnv, train_id: str, route: List[str],
                 graph: NetworkXGraph, simulation: 'EnhancedSimulation',
                 od_pair: Optional[ODPair] = None):
        self.env = env
        self.id = train_id
        self.route = route
        self.graph = graph
        self.simulation = simulation
        self.od_pair = od_pair or ODPair("unknown", "unknown", 0.0)

        # Position and movement
        self.visual_pos: Tuple[float, float] = self.graph.nodes[self.route[0]]['pos']
        self.is_visible: bool = True
        self.current_index: int = 0
        self.current_speed: float = 0.0
        self.target_speed: float = simulation.config.train_speed_pps

        # Train characteristics
        self.train_type: TrainType = TrainType(od_pair.train_type) if od_pair else TrainType.PASSENGER
        self.priority: Priority = Priority(od_pair.priority) if od_pair else Priority.NORMAL
        self.max_speed: float = od_pair.max_speed or simulation.config.train_speed_pps

        # Safety and performance
        self.metrics = TrainMetrics()
        self.emergency_brake_active: bool = False
        self.collision_risk_level: float = 0.0
        self._last_collision_check: float = 0.0  # REFACTOR: Cooldown for collision checks

        # Process management
        self.process = env.process(self.run())

        # Performance tracking
        self.start_time: float = env.now
        self.last_position_update: float = env.now
        self.position_history: deque = deque(maxlen=100)

    def run(self):
        """Main train process with enhanced safety and performance features"""
        env, sim, rec = self.env, self.simulation, self.simulation.recorder

        if rec:
            rec.record_event(env.now, "spawn", {
                "train_id": self.id,
                "route": list(self.route),
                "train_type": self.train_type.value,
                "priority": self.priority.value
            })

        try:
            for i in range(len(self.route) - 1):
                self.current_index = i
                current_node, next_node = self.route[i], self.route[i + 1]

                # Signal handling with enhanced logic
                yield from self._handle_signal(current_node, next_node)

                # Track block handling with safety checks
                yield from self._handle_track_block(current_node, next_node)

                # Update metrics
                self._update_metrics()

        except simpy.Interrupt:
            self.logger.warning(f"Train {self.id} was interrupted.")
            if rec:
                rec.record_event(env.now, "interrupt", {"train_id": self.id})
        except Exception as e:
            sim.logger.error(f"Train {self.id} encountered error: {e}")
            if rec:
                rec.record_event(env.now, "error", {
                    "train_id": self.id,
                    "error": str(e)
                })
        finally:
            self.is_visible = False
            self.metrics.total_time = env.now - self.start_time
            if rec:
                rec.record_event(env.now, "finish", {
                    "train_id": self.id,
                    "metrics": self.metrics.__dict__
                })

    def _handle_signal(self, current_node: str, next_node: str):
        """Enhanced signal handling with timing and safety"""
        sig_id = f"sig_{current_node}_to_{next_node}_starter"
        sig = self.simulation.signal_map.get(sig_id)

        if not sig:
            return

        wait_started, wait_start_time = False, None

        while sig.aspect != SignalAspect.GREEN:
            if not wait_started:
                wait_started, wait_start_time = True, self.env.now
                print(f"⏳ WAITING  | T:{self.env.now:7.2f} | Train {self.id} waiting at RED signal {sig.id}")
                self.simulation.logger.debug(f"Train {self.id} waiting at signal {sig.id}")

            if self.simulation.recorder:
                self.simulation.recorder.record_event(self.env.now, "wait_start", {
                    "train_id": self.id,
                    "signal": sig.id,
                    "aspect": sig.aspect.value
                })

        yield self.env.timeout(1.0)

        if wait_started and self.simulation.recorder:
            wait_duration = self.env.now - wait_start_time
            self.metrics.wait_time += wait_duration
            self.simulation.recorder.record_event(self.env.now, "wait_end", {
                "train_id": self.id,
                "duration": float(wait_duration)
            })

    def _handle_track_block(self, current_node: str, next_node: str):
        """Enhanced track block handling with collision detection"""
        key = tuple(sorted((current_node, next_node)))
        if key not in self.simulation.track_blocks:
            self.simulation.logger.error(f"Missing track block for {key}")
            return

        block = self.simulation.track_blocks[key]

        # Safety check before entering
        if self.simulation.config.collision_detection and not block.is_safe_for_entry(self.current_speed, 0):
            yield from self._handle_emergency()
            return

        # DEBUG LOG: Announce the request for a track block, including priority
        print(
            f"➡️ REQUEST  | T:{self.env.now:7.2f} | Train {self.id} (Prio:{self.priority.value}) requests track {key}")

        # Request access with priority
        with block.request_access(self.id, self.priority) as req:
            yield req

            # DEBUG LOG: Announce when the request is granted
            print(f"✅ GRANTED  | T:{self.env.now:7.2f} | Train {self.id} (Prio:{self.priority.value}) gets track {key}")

            # REFACTOR: Removed block.is_occupied = True (now handled by property)
            block.last_occupancy_time = self.env.now

            if self.simulation.recorder:
                self.simulation.recorder.record_event(self.env.now, "enter_edge", {
                    "train_id": self.id,
                    "edge": key,
                    "priority": self.priority.value
                })

            # Movement with enhanced physics
            yield from self._move_along_edge(current_node, next_node)

            # REFACTOR: Removed block.is_occupied = False (now handled by property)
            block.total_occupancy_time += self.env.now - block.last_occupancy_time

            if self.simulation.recorder:
                self.simulation.recorder.record_event(self.env.now, "leave_edge", {
                    "train_id": self.id,
                    "edge": key
                })

    def _move_along_edge(self, start_node: str, end_node: str):
        """
        Enhanced movement with realistic physics and collision detection.
        REFACTOR: This function is now optimized for headless mode.
        """
        start_pos = self.graph.nodes[start_node]['pos']
        end_pos = self.graph.nodes[end_node]['pos']
        distance = self.graph.edges[start_node, end_node]['distance']

        # Calculate travel time with speed variations
        base_speed = self.target_speed
        if self.train_type == TrainType.FREIGHT:
            base_speed *= 0.6
        elif self.train_type == TrainType.EXPRESS:
            base_speed *= 1.2

        base_speed = min(base_speed, self.max_speed)
        travel_time = max(0.1, distance / base_speed if base_speed > 0 else 99999)

        # --- REFACTOR: Performance Optimization for Headless Mode ---
        if self.simulation.mode == SimulationMode.HEADLESS:
            # In headless mode, we don't need step-by-step visualization.
            # Just yield for the total time. The block resource handles
            # the primary collision avoidance (preventing entry).
            yield self.env.timeout(travel_time)
            self.metrics.total_distance += distance

        else:
            # --- Visual Mode: Step-by-step movement ---
            start_time = self.env.now
            step_duration = self.simulation.config.visual_update_step

            while self.env.now < start_time + travel_time:

                # REFACTOR: Check emergency flag *before* moving
                if self.emergency_brake_active:
                    yield from self._handle_emergency()  # Wait for brake to release
                    return  # Stop this movement

                # Calculate time to yield (don't overshoot)
                time_remaining = (start_time + travel_time) - self.env.now
                current_step_time = min(step_duration, time_remaining, 1.0)  # Cap at 1.0s

                if current_step_time <= 1e-9:  # Avoid tiny steps
                    break

                progress = (self.env.now - start_time + current_step_time) / travel_time
                self.visual_pos = (
                    start_pos[0] + (end_pos[0] - start_pos[0]) * progress,
                    start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                )
                self.current_speed = base_speed

                # REFACTOR: O(k) collision detection
                if self.simulation.config.collision_detection:
                    # This function is now a generator.
                    # If it detects a collision, it will yield to _handle_emergency
                    # and self.emergency_brake_active will be set.
                    yield from self._check_collisions()

                    # REFACTOR: If the check triggered an emergency, stop.
                    if self.emergency_brake_active:
                        return  # Exit movement loop

                self.metrics.total_distance += base_speed * current_step_time
                yield self.env.timeout(current_step_time)

        # Final position
        self.visual_pos = end_pos
        self.current_speed = 0.0

    # REFACTOR: This is now a generator and is O(k) instead of O(N^2)
    def _check_collisions(self):
        """
        Check for potential collisions with other trains ON THE SAME BLOCK.
        This is an efficient O(k) check.
        This function is a generator: it yields if an emergency brake is triggered.
        """
        if not NUMPY_AVAILABLE or not self.simulation.config.collision_detection:
            return

        # 1. Identify the current block
        if self.current_index >= len(self.route) - 1:
            return  # Train is at/near its destination node

        current_node = self.route[self.current_index]
        next_node = self.route[self.current_index + 1]
        edge_key = tuple(sorted((current_node, next_node)))

        # 2. Get trains on this edge
        trains_on_this_edge = self.simulation.get_trains_on_edge(edge_key)

        if len(trains_on_this_edge) <= 1:
            self.collision_risk_level = 0.0
            return  # No other trains on this block

        for other_train in trains_on_this_edge:
            if other_train.id == self.id:
                continue

            distance = np.sqrt(
                (self.visual_pos[0] - other_train.visual_pos[0]) ** 2 +
                (self.visual_pos[1] - other_train.visual_pos[1]) ** 2
            )

            # REFACTOR: Use collision distance from config
            critical_dist = self.config.collision_critical_distance
            if distance < critical_dist:
                # Use a cooldown to avoid spamming
                if self.env.now - self._last_collision_check > 1.0:
                    self.simulation.handle_collision(self, other_train)
                    self._last_collision_check = self.env.now

                    # REFACTOR: BUG FIX - Yield to the emergency handler
                    # This pauses this train's process.
                    yield from self._handle_emergency()

                    # REFACTOR: BUG FIX - Stop checking
                    return

            # REFACTOR: Use collision distance from config
            warning_dist = self.config.collision_warning_distance
            self.collision_risk_level = max(self.collision_risk_level,
                                            1.0 - (distance / warning_dist))

        # If no critical collision, reset risk (it's re-calculated next step)
        if self.collision_risk_level > 0:
            self.collision_risk_level = 0.0

        # This makes it a generator, but the yield from _handle_emergency
        # already does. We'll add a 'pass' to be explicit this function
        # is complete.
        pass

    def _check_emergency_conditions(self) -> bool:
        """Check if emergency conditions are met"""
        if self.emergency_brake_active:
            return True

        # REFACTOR: Use config value
        if self.collision_risk_level > (
                self.config.collision_warning_distance - self.config.collision_critical_distance) / self.config.collision_warning_distance:  # e.g., > 0.75
            return True

        return False

    def _handle_emergency(self):
        """Handle emergency situations"""
        if self.emergency_brake_active:
            # Already handling an emergency, just wait
            yield self.env.timeout(self.config.emergency_stop_time)
            return

        self.emergency_brake_active = True
        self.target_speed = 0.0
        self.current_speed = 0.0  # Full stop

        if self.simulation.recorder:
            self.simulation.recorder.record_event(self.env.now, "emergency_brake", {
                "train_id": self.id,
                "reason": "emergency_conditions"
            })

        # Emergency stop
        # REFACTOR: Use emergency stop time from config
        yield self.env.timeout(self.config.emergency_stop_time)

        # Reset flags after stop
        self.emergency_brake_active = False
        self.collision_risk_level = 0.0
        self.target_speed = self.simulation.config.train_speed_pps  # Resume speed

    def _update_metrics(self):
        """Update train performance metrics"""
        current_time = self.env.now
        time_delta = current_time - self.last_position_update

        if time_delta > 0:
            self.metrics.total_time += time_delta
            self.last_position_update = current_time

    def is_at_node(self, node_id: str) -> bool:
        """Check if train is at a specific node"""
        if self.current_index >= len(self.route):
            return False
        # Train is "at node" if it's at the start of its *next* edge
        return self.route[self.current_index] == node_id

    def is_on_edge(self, edge_key: Tuple[str, str]) -> bool:
        """Check if train is on a specific edge"""
        if self.current_index >= len(self.route) - 1:
            return False
        current_edge = tuple(sorted((self.route[self.current_index], self.route[self.current_index + 1])))
        return current_edge == edge_key


# ---------------------------
# Part 5: Enhanced Graph Loader
# ---------------------------

def load_enhanced_railway_graph(filepath: str) -> Tuple[Optional[NetworkXGraph], List[Dict[str, Any]]]:
    """Load railway graph with enhanced validation and preprocessing"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The data file '{filepath}' was not found.")
        return None, []
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Invalid JSON in file '{filepath}': {e}")
        return None, []

    G = nx.Graph()
    signals_data = []

    # Enhanced node processing
    for node in data.get('nodes', []):
        nid = str(node['id'])
        pos = (float(node['pos_x']), float(node['pos_y']))

        # Enhanced node attributes
        node_attrs = {
            'pos': pos,
            'label': node.get('label'),
            'capacity': node.get('capacity', 100),  # Default capacity
            'type': 'station' if node.get('label') else 'junction',
            'platforms': node.get('platforms', 1),
            'facilities': node.get('facilities', [])
        }

        G.add_node(nid, **node_attrs)

    # Enhanced edge processing
    for edge in data.get('edges', []):
        s, t = str(edge['source']), str(edge['target'])

        if not G.has_node(s) or not G.has_node(t):
            print(f"Warning: Edge {s}-{t} references non-existent nodes")
            continue

        # Calculate distance if not provided
        pos_s = G.nodes[s]['pos']
        pos_t = G.nodes[t]['pos']
        distance = 1.0  # Default
        if NUMPY_AVAILABLE:
            distance = np.sqrt((pos_s[0] - pos_t[0]) ** 2 + (pos_s[1] - pos_t[1]) ** 2)

        # Enhanced edge attributes
        edge_attrs = {
            'distance': float(edge.get('distance', distance)),
            'max_speed': edge.get('max_speed', 50.0),
            'track_type': edge.get('track_type', 'main'),
            'electrified': edge.get('electrified', True),
            'signals': edge.get('signals', [])
        }

        G.add_edge(s, t, **edge_attrs)

        # Process signals
        if edge.get('signals'):
            signals_data.extend(edge.get('signals'))

    # Validate graph connectivity
    if not nx.is_connected(G):
        print("Warning: Graph is not fully connected")
        # Find largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Using largest connected component with {len(G.nodes)} nodes")

    return G, signals_data


# ---------------------------
# Part 6: Enhanced Simulation Controller
# ---------------------------

class EnhancedSimulation:
    """Enhanced simulation controller with advanced features and better architecture"""

    def __init__(self, datafile: str = None,
                 config: Optional[SimulationConfig] = None,
                 mode: SimulationMode = SimulationMode.HEADLESS):

        # Configuration
        self.config = config or SimulationConfig()
        self.mode = mode

        # Setup logging
        self.logger = logging.getLogger('EnhancedSimulation')

        # Resolve default data path
        if datafile is None:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            datafile = os.path.join(_root, "data", "processed", "godot_station_graph.json")

        # Initialize random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            if NUMPY_AVAILABLE:
                np.random.seed(self.config.seed)

        # Core simulation components
        self.env: SimPyEnv = simpy.Environment()
        self.graph, signals_data = load_enhanced_railway_graph(datafile)

        if self.graph is None or len(self.graph.nodes) == 0:
            self.is_running = False
            self.logger.error("Failed to load railway graph or graph is empty")
            return

        self.is_running = True
        self.current_sim_time = 0.0

        # Enhanced data recording
        self.recorder = EnhancedDataRecorder(self.config)

        # Network topology
        self.stations = [nid for nid, data in self.graph.nodes(data=True) if data.get("label")]
        if not self.stations:
            self.logger.warning("No stations found in graph. Spawning will fail.")

        self.track_blocks: Dict[Tuple[str, str], EnhancedTrackBlock] = {
            tuple(sorted((u, v))): EnhancedTrackBlock(self.env, tuple(sorted((u, v))), self.config)
            for u, v in self.graph.edges()
        }

        # Enhanced signals
        self.signals = [EnhancedSignal(s) for s in signals_data]
        self.signal_map = {s.id: s for s in self.signals}

        # Train management
        self.trains: List[EnhancedTrain] = []
        self.train_counter = 0
        # REFACTOR: This lookup is for the O(k) collision check.
        # It's faster to maintain it than to compute it on the fly.
        # This is a trade-off: slightly slower train movement for much
        # faster collision checking.
        # On second thought, maintaining this dict is complex.
        # A helper function is cleaner, even if slightly slower.
        # self._edge_train_lookup: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # Performance tracking
        self.collision_count = 0
        self.disruption_count = 0
        self.total_energy_consumed = 0.0

        # Visualization setup
        self.visual = (mode != SimulationMode.HEADLESS) and PYGAME_AVAILABLE
        if self.visual:
            self._setup_visualization()

        # Start background processes
        self.env.process(self._signal_controller_process())
        self.env.process(self._metrics_updater_process())

        self.logger.info(
            f"Enhanced simulation initialized with {len(self.stations)} stations and {len(self.track_blocks)} track blocks")

    def _setup_visualization(self):
        """Setup enhanced visualization system"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.screen_width, self.config.screen_height))
        pygame.display.set_caption("Enhanced Railway Simulation")

        # Enhanced fonts
        try:
            self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
            self.font_medium = pygame.font.SysFont('Arial', 18)
            self.font_small = pygame.font.SysFont('Arial', 14)
            self.font_tiny = pygame.font.SysFont('Arial', 12)
        except Exception:
            self.font_large = pygame.font.Font(None, 30)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)
            self.font_tiny = pygame.font.Font(None, 18)

        self.clock = pygame.time.Clock()

        # Camera system
        self.camera = Camera(self.config.screen_width, self.config.screen_height)
        self.camera.fit_to_graph(self.graph)

        # UI system
        self.ui_manager = UIManager(self.config)

        # Input handling
        self.input_handler = InputHandler()

        # Animation system
        if self.config.enable_animations:
            self.animation_manager = AnimationManager()

    def _signal_controller_process(self):
        """Enhanced signal controller using MACLS 4-aspect cascade logic.
        Delegates to CascadeSignalController if available, else falls back to flat logic."""
        # Try to use the IR cascade controller
        if hasattr(self, '_cascade_controller') and self._cascade_controller is not None:
            while True:
                changed = self._cascade_controller.tick(self.env.now)
                if changed:
                    self.recorder.record_event(self.env.now, "signal_change", {
                        "changed_signals": changed,
                        "count": len(changed),
                    })
                yield self.env.timeout(1.0)
        else:
            # Fallback: flat red/green logic (backward compatibility)
            while True:
                for signal in self.signals:
                    protected = self.track_blocks.get(signal.protected_block_key)
                    if protected is None:
                        signal.change_aspect(SignalAspect.OFF, self.env.now)
                        continue
                    if protected.is_occupied:
                        signal.change_aspect(SignalAspect.RED, self.env.now)
                    elif protected._disrupted:
                        signal.change_aspect(SignalAspect.YELLOW, self.env.now)
                    else:
                        signal.change_aspect(SignalAspect.GREEN, self.env.now)
                yield self.env.timeout(1.0)

    def _metrics_updater_process(self):
        """Background process for updating metrics"""
        while True:
            self.recorder.maybe_update_metrics(self.env.now, self)
            yield self.env.timeout(self.config.metrics_update_interval)

    # REFACTOR: New helper function for O(k) collision detection.
    def get_trains_on_edge(self, edge_key: Tuple[str, str]) -> List[EnhancedTrain]:
        """Helper to find all visible trains on a specific edge."""
        # This is still O(N), but it's only called by trains on that edge,
        # so the *total* complexity is much lower than O(N^2).
        # To make this O(k), we would need to maintain a separate
        # lookup dictionary, which adds complexity to train movement.
        # This O(N) lookup inside an O(k) loop is acceptable.
        trains_found = []
        for t in self.trains:
            if t.is_visible and t.is_on_edge(edge_key):
                trains_found.append(t)
        return trains_found

    def spawn_train_from_od(self, od_pair: ODPair) -> Optional[EnhancedTrain]:
        """Spawn a single train from OD pair with enhanced validation"""
        if len(self.trains) >= self.config.max_trains:
            self.logger.warning(f"Maximum train limit ({self.config.max_trains}) reached")
            return None

        # Validate OD pair
        if not self.stations:
            self.logger.error("Cannot spawn trains, no stations defined in graph.")
            return None
        if od_pair.origin not in self.stations or od_pair.destination not in self.stations:
            self.logger.error(f"Invalid OD pair: {od_pair.origin} -> {od_pair.destination}")
            return None

        # Generate route
        try:
            route = nx.shortest_path(self.graph, od_pair.origin, od_pair.destination, weight='distance')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.logger.warning(f"No path found for {od_pair.origin} -> {od_pair.destination}")
            return None

        if len(route) < 2:
            self.logger.warning(f"Invalid route length for {od_pair.origin} -> {od_pair.destination}")
            return None

        # Create train
        train_id = f"T{self.train_counter:04d}"
        train = EnhancedTrain(self.env, train_id, route, self.graph, self, od_pair)

        self.trains.append(train)
        self.train_counter += 1

        self.logger.info(f"Spawned train {train_id}: {od_pair.origin} -> {od_pair.destination}")
        return train

    def spawn_trains_from_ods(self, od_pairs: List[ODPair], add_noise: bool = True):
        """Spawn multiple trains from OD pairs with enhanced scheduling"""

        def _spawn_proc(env: SimPyEnv, sim: 'EnhancedSimulation', od_list: List[ODPair]):
            for od in od_list:
                # Wait until departure time
                yield env.timeout(max(0.0, od.depart_time - env.now))

                # Add speed noise if requested
                if add_noise and NUMPY_AVAILABLE:
                    speed_noise = np.random.normal(1.0, 0.05)
                    od.max_speed = (od.max_speed or sim.config.train_speed_pps) * speed_noise

                # Spawn train
                train = sim.spawn_train_from_od(od)
                if train:
                    self.logger.debug(f"Spawned train {train.id} at {env.now:.2f}s")

        self.env.process(_spawn_proc(self.env, self, od_pairs))

    def schedule_random_disruptions(self):
        """Schedule random disruptions with enhanced effects"""

        def _disrupt_proc(env: SimPyEnv, sim: 'EnhancedSimulation'):
            edges = list(sim.track_blocks.keys())
            if not edges:
                return

            while True:
                # Exponential inter-arrival time
                next_in = random.expovariate(max(1e-6, sim.config.disruption_rate))
                yield env.timeout(next_in)

                # Select random edge
                edge = random.choice(edges)
                block = sim.track_blocks.get(edge)

                if block is None or block._disrupted:
                    continue

                # Generate disruption duration
                duration = max(1.0, random.expovariate(1.0 / sim.config.mean_disruption_duration))

                # Apply disruption
                block._disrupted = True
                sim.disruption_count += 1

                # DEBUG LOG: Announce disruption start
                print(f"🔴 DISRUPT  | T:{env.now:7.2f} | Track {edge} disrupted for {duration:.2f}s")
                sim.logger.warning(f"Disruption on track {edge} for {duration:.2f}s")
                sim.recorder.record_event(env.now, "disruption_start", {
                    "edge": edge,
                    "duration": duration
                })

                yield env.timeout(duration)

                # End disruption
                block._disrupted = False
                # DEBUG LOG: Announce disruption end
                print(f"🟢 REPAIRED | T:{env.now:7.2f} | Track {edge} is now clear")
                sim.logger.info(f"Track {edge} repaired")
                sim.recorder.record_event(env.now, "disruption_end", {"edge": edge})

        self.env.process(_disrupt_proc(self.env, self))

    def handle_collision(self, train1: EnhancedTrain, train2: EnhancedTrain):
        """Handle collision between two trains"""
        self.collision_count += 1

        self.logger.error(f"COLLISION between {train1.id} and {train2.id}")

        if self.recorder:
            self.recorder.record_event(self.env.now, "collision", {
                "train1": train1.id,
                "train2": train2.id,
                "position1": train1.visual_pos,
                "position2": train2.visual_pos
            })

        # Emergency response
        # The trains' own _check_collisions and _handle_emergency
        # processes will take care of stopping them.
        pass

    def has_network_emergency(self) -> bool:
        """Check if there's a network-wide emergency"""
        return self.collision_count > 0 or any(block._disrupted for block in self.track_blocks.values())

    def get_node_utilization(self, node_id: str) -> float:
        """Get utilization of a specific node"""
        trains_at_node = len([t for t in self.trains if t.is_at_node(node_id)])
        capacity = self.graph.nodes[node_id].get('capacity', 1)  # Avoid div by zero
        return min(trains_at_node / max(capacity, 1), 1.0)

    def get_edge_utilization(self, edge_key: Tuple[str, str]) -> float:
        """Get utilization of a specific edge"""
        block = self.track_blocks.get(edge_key)
        if not block:
            return 0.0

        # Calculate utilization based on occupancy time
        # REFACTOR: Robust division
        if self.current_sim_time > 0:
            return min(block.total_occupancy_time / self.current_sim_time, 1.0)
        return 0.0

    def get_network_utilization(self) -> float:
        """Get overall network utilization"""
        if not self.track_blocks:
            return 0.0

        total_utilization = sum(self.get_edge_utilization(key) for key in self.track_blocks.keys())
        # REFACTOR: Robust division
        return total_utilization / len(self.track_blocks) if self.track_blocks else 0.0

    def get_energy_efficiency(self) -> float:
        """Calculate energy efficiency metric"""
        if not self.trains:
            return 0.0

        total_distance = sum(getattr(t, 'metrics', TrainMetrics()).total_distance for t in self.trains)
        # REFACTOR: Robust division
        return total_distance / self.total_energy_consumed if self.total_energy_consumed > 0 else 0.0

    def get_passenger_satisfaction(self) -> float:
        """Calculate passenger satisfaction metric"""
        if not self.trains:
            return 1.0

        satisfaction_scores = []
        for train in self.trains:
            if hasattr(train, 'metrics'):
                satisfaction_scores.append(train.metrics.passenger_satisfaction)

        if not satisfaction_scores:
            return 1.0

        # REFACTOR: Use robust numpy mean
        return np.mean(satisfaction_scores) if NUMPY_AVAILABLE else (
                sum(satisfaction_scores) / len(satisfaction_scores))

    def advance(self, sim_seconds: float):
        """Advance simulation by specified time"""
        target = self.current_sim_time + sim_seconds
        self.env.run(until=target)
        self.current_sim_time = target

        # Update recorder
        self.recorder.maybe_snapshot(self.current_sim_time, self)

    def run_headless_episode(self, duration: float):
        """Run headless episode with enhanced monitoring"""
        self.logger.info(f"Starting headless episode for {duration:.2f} seconds")

        start_time = time.time()

        # REFACTOR: Use a single run command for better performance
        # The background processes (spawner, metrics) will still run.
        self.env.run(until=duration)
        self.current_sim_time = self.env.now

        real_time = time.time() - start_time
        self.logger.info(f"Episode completed in {real_time:.2f} real seconds")

    def run_visual_simulation(self):
        """Run visual simulation with enhanced UI"""
        if not self.visual:
            raise RuntimeError("Visual simulation requires pygame")

        self.logger.info("Starting visual simulation")

        running = True
        while running and self.is_running:
            delta_time = self.clock.tick(self.config.fps) / 1000.0

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.input_handler.handle_event(event)
                    # Pass event to camera for zoom
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 4:  # Scroll up
                            self.camera.handle_zoom(pygame.mouse.get_pos(), 1.1)
                        elif event.button == 5:  # Scroll down
                            self.camera.handle_zoom(pygame.mouse.get_pos(), 0.9)

            # Update systems
            self.input_handler.update(delta_time)
            self.camera.update(delta_time, self.input_handler)

            if self.config.enable_animations and hasattr(self, 'animation_manager'):
                self.animation_manager.update(delta_time)

            # Advance simulation
            sim_delta = delta_time * self.config.simulation_speed_factor
            self.advance(sim_delta)

            # Render
            self._render()

        pygame.quit()
        self.logger.info("Visual simulation ended")

    def _render(self):
        """Enhanced rendering with better performance and features"""
        self.screen.fill(Colors.BACKGROUND)

        # Render grid
        self._render_grid()

        # Render tracks
        self._render_tracks()

        # Render nodes
        self._render_nodes()

        # Render signals
        self._render_signals()

        # Render trains
        self._render_trains()

        # Render UI
        self._render_ui()

        # Render debug info if enabled
        if self.config.show_debug_info:
            self._render_debug_info()

        pygame.display.flip()

    def _render_grid(self):
        """Render grid with performance optimization"""
        if self.camera.scale < 0.3:
            return

        grid_size = 50 * self.camera.scale
        if grid_size < 20:
            return

        start_x = int(-self.camera.offset.x % grid_size)
        start_y = int(-self.camera.offset.y % grid_size)

        for x in range(start_x, self.config.screen_width + int(grid_size), int(grid_size)):
            pygame.draw.line(self.screen, Colors.GRID, (x, 0), (x, self.config.screen_height), 1)

        for y in range(start_y, self.config.screen_height + int(grid_size), int(grid_size)):
            pygame.draw.line(self.screen, Colors.GRID, (0, y), (self.config.screen_width, y), 1)

    def _render_tracks(self):
        """Render tracks with enhanced visual feedback"""
        track_width = max(2, int(3 * self.camera.scale))

        for u, v, data in self.graph.edges(data=True):
            sp = self.camera.world_to_screen(self.graph.nodes[u]['pos'])
            ep = self.camera.world_to_screen(self.graph.nodes[v]['pos'])

            # Performance culling
            if (max(sp[0], ep[0]) < 0 or min(sp[0], ep[0]) > self.config.screen_width or
                    max(sp[1], ep[1]) < 0 or min(sp[1], ep[1]) > self.config.screen_height):
                continue

            # Determine track color
            key = tuple(sorted((u, v)))
            block = self.track_blocks.get(key)

            color = Colors.TRACK_MAIN
            if block:
                if block._disrupted:
                    # Flashing effect for disruptions
                    if int(self.current_sim_time * 2) % 2 == 0:
                        color = Colors.TRACK_DISRUPTED
                # REFACTOR: Use property
                elif block.is_occupied:
                    color = Colors.TRACK_OCCUPIED

            # Draw track with shadow
            shadow_offset = max(1, int(self.camera.scale))
            try:
                pygame.draw.line(self.screen, (40, 40, 40),
                                 (sp[0] + shadow_offset, sp[1] + shadow_offset),
                                 (ep[0] + shadow_offset, ep[1] + shadow_offset), track_width)
                pygame.draw.line(self.screen, color, sp, ep, track_width)
            except TypeError:
                # Can happen if coords are too large
                pass

    def _render_nodes(self):
        """Render nodes with enhanced styling"""
        station_radius = max(4, int(8 * self.camera.scale))
        junction_radius = max(2, int(5 * self.camera.scale))

        for nid, d in self.graph.nodes(data=True):
            pos = self.camera.world_to_screen(d['pos'])

            # Performance culling
            if (pos[0] < -station_radius or pos[0] > self.config.screen_width + station_radius or
                    pos[1] < -station_radius or pos[1] > self.config.screen_height + station_radius):
                continue

            if d.get('label'):  # Station
                # Station body
                pygame.draw.circle(self.screen, Colors.STATION_FILL, pos, station_radius)
                pygame.draw.circle(self.screen, Colors.STATION_BORDER, pos, station_radius,
                                   max(1, int(self.camera.scale)))

                # Station label
                if self.camera.scale > 0.3:
                    label_surf = self.font_small.render(d['label'], True, Colors.TEXT_PRIMARY)
                    label_rect = label_surf.get_rect(midleft=(pos[0] + station_radius + 5, pos[1]))
                    self.screen.blit(label_surf, label_rect)

                # Utilization indicator
                utilization = self.get_node_utilization(nid)
                if utilization > 0.8:  # High utilization warning
                    warning_color = Colors.TEXT_WARNING
                    pygame.draw.circle(self.screen, warning_color, pos, station_radius + 2, 2)
            else:  # Junction
                pygame.draw.circle(self.screen, Colors.JUNCTION_FILL, pos, junction_radius)
                pygame.draw.circle(self.screen, Colors.JUNCTION_BORDER, pos, junction_radius,
                                   max(1, int(self.camera.scale)))

    def _render_signals(self):
        """Render signals with enhanced visual feedback"""
        if self.camera.scale < 0.2:  # Cull signals when zoomed out
            return

        signal_radius = max(3, int(6 * self.camera.scale))
        post_height = signal_radius * 2.5

        for s in self.signals:
            pos = self.camera.world_to_screen(s.pos)

            # Performance culling
            if (pos[0] < -signal_radius or pos[0] > self.config.screen_width + signal_radius or
                    pos[1] < -post_height or pos[1] > self.config.screen_height + post_height):
                continue

            # Signal post
            post_rect = pygame.Rect(pos[0] - 2, pos[1], 4, -post_height)
            pygame.draw.rect(self.screen, (80, 80, 80), post_rect)

            # Signal light
            light_pos = (pos[0], int(pos[1] - post_height + signal_radius))

            # Color based on aspect
            aspect_colors = {
                SignalAspect.RED: Colors.SIGNAL_RED,
                SignalAspect.YELLOW: Colors.SIGNAL_YELLOW,
                SignalAspect.GREEN: Colors.SIGNAL_GREEN,
                SignalAspect.OFF: Colors.SIGNAL_OFF
            }

            color = aspect_colors.get(s.aspect, Colors.SIGNAL_OFF)
            pygame.draw.circle(self.screen, color, light_pos, signal_radius)
            pygame.draw.circle(self.screen, (200, 200, 200), light_pos, signal_radius, 1)

    def _render_trains(self):
        """Render trains with enhanced styling and animations"""
        train_size = max(6, int(12 * self.camera.scale))

        for t in self.trains:
            if not t.is_visible:
                continue

            pos = self.camera.world_to_screen(t.visual_pos)

            # Performance culling
            if (pos[0] < -train_size or pos[0] > self.config.screen_width + train_size or
                    pos[1] < -train_size or pos[1] > self.config.screen_height + train_size):
                continue

            # Train color based on type
            color = Colors.get_train_color(t.train_type)

            # Emergency brake indicator
            if t.emergency_brake_active:
                # Flashing effect
                if int(self.current_sim_time * 5) % 2 == 0:
                    color = Colors.TEXT_ERROR

            # Train body
            rect = pygame.Rect(pos[0] - train_size // 2, pos[1] - train_size // 2,
                               train_size, train_size)

            # Shadow
            shadow_rect = rect.move(2, 2)
            pygame.draw.rect(self.screen, (40, 40, 40), shadow_rect)

            # Main body
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (40, 40, 40), rect, max(1, int(self.camera.scale)))

            # Train ID
            if self.camera.scale > 0.5:
                id_surf = self.font_tiny.render(t.id, True, Colors.TEXT_PRIMARY)
                id_rect = id_surf.get_rect(center=(pos[0], pos[1] - train_size - 5))
                self.screen.blit(id_surf, id_rect)

            # Speed indicator
            if self.camera.scale > 0.3 and t.current_speed > 0:
                speed_text = f"{t.current_speed:.1f}"
                speed_surf = self.font_tiny.render(speed_text, True, Colors.TEXT_SECONDARY)
                speed_rect = speed_surf.get_rect(center=(pos[0], pos[1] + train_size + 5))
                self.screen.blit(speed_surf, speed_rect)

    def _render_ui(self):
        """Render enhanced UI with comprehensive information"""
        # Main UI panel
        ui_width, ui_height = 400, 200
        ui_surface = pygame.Surface((ui_width, ui_height))
        ui_surface.set_alpha(220)
        ui_surface.fill((0, 0, 0))
        self.screen.blit(ui_surface, (10, 10))

        # Time display
        hours = int(self.current_sim_time // 3600)
        minutes = int((self.current_sim_time % 3600) // 60)
        seconds = int(self.current_sim_time % 60)
        time_text = f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
        time_surf = self.font_medium.render(time_text, True, Colors.TEXT_PRIMARY)
        self.screen.blit(time_surf, (20, 20))

        # Train information
        active_trains = sum(1 for t in self.trains if t.is_visible)
        total_trains = len(self.trains)
        train_text = f"Trains: {active_trains}/{total_trains}"
        train_surf = self.font_small.render(train_text, True, Colors.TEXT_PRIMARY)
        self.screen.blit(train_surf, (20, 50))

        # Network metrics
        utilization = self.get_network_utilization()
        util_text = f"Utilization: {utilization:.1%}"
        util_surf = self.font_small.render(util_text, True, Colors.TEXT_PRIMARY)
        self.screen.blit(util_surf, (20, 70))

        # Collision count
        if self.collision_count > 0:
            collision_text = f"Collisions: {self.collision_count}"
            collision_surf = self.font_small.render(collision_text, True, Colors.TEXT_ERROR)
            self.screen.blit(collision_surf, (20, 90))
        else:
            collision_text = "Collisions: 0"
            collision_surf = self.font_small.render(collision_text, True, Colors.SIGNAL_GREEN)
            self.screen.blit(collision_surf, (20, 90))

        # Disruption count
        if self.disruption_count > 0:
            disruption_text = f"Disruptions: {self.disruption_count}"
            disruption_surf = self.font_small.render(disruption_text, True, Colors.TEXT_WARNING)
            self.screen.blit(disruption_surf, (20, 110))

        # Controls
        controls_text = "Controls: WASD pan | Mouse wheel zoom | R reset | ESC quit"
        controls_surf = self.font_tiny.render(controls_text, True, Colors.TEXT_SECONDARY)
        self.screen.blit(controls_surf, (20, 150))

        # Camera info
        camera_text = f"Zoom: {self.camera.scale:.2f}x"
        camera_surf = self.font_tiny.render(camera_text, True, Colors.TEXT_SECONDARY)
        self.screen.blit(camera_surf, (20, 170))

    def _render_debug_info(self):
        """Render debug information overlay"""
        debug_y = 220
        debug_surface = pygame.Surface((400, 300))
        debug_surface.set_alpha(180)
        debug_surface.fill((0, 0, 0))
        self.screen.blit(debug_surface, (10, debug_y))

        # Debug information
        debug_info = [
            f"FPS: {self.clock.get_fps():.1f}",
            f"Sim Speed: {self.config.simulation_speed_factor:.1f}x",
            f"Memory: {self._get_memory_usage():.1f} MB",
            f"Events: {len(self.recorder.events)}",
            f"Snapshots: {len(self.recorder.snapshots)}",
            f"Metrics: {len(self.recorder.metrics_history)}"
        ]

        for i, info in enumerate(debug_info):
            info_surf = self.font_tiny.render(info, True, Colors.TEXT_SECONDARY)
            self.screen.blit(info_surf, (20, debug_y + 20 + i * 20))

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


# ---------------------------
# Part 7: Enhanced Support Classes
# ---------------------------

class Camera:
    """Enhanced camera system for visualization"""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = 0.5
        self.offset = pygame.Vector2(0, 0)
        self.target_scale = self.scale
        self.target_offset = pygame.Vector2(0, 0)
        self.bounds = (0, 1, 0, 1)
        # REFACTOR: Store the graph for resetting view
        self.graph: Optional[NetworkXGraph] = None

    def fit_to_graph(self, graph: NetworkXGraph):
        """Fit camera to show entire graph"""
        # REFACTOR: Store graph
        self.graph = graph

        valid_nodes = [d['pos'] for _, d in graph.nodes(data=True) if 'pos' in d]
        if not valid_nodes:
            return

        min_x = min(p[0] for p in valid_nodes)
        max_x = max(p[0] for p in valid_nodes)
        min_y = min(p[1] for p in valid_nodes)
        max_y = max(p[1] for p in valid_nodes)

        self.bounds = (min_x, max_x, min_y, max_y)

        # Calculate scale to fit graph
        graph_width = max_x - min_x
        graph_height = max_y - min_y

        if graph_width == 0 or graph_height == 0:
            return

        padding = 0.8
        scale_x = (self.screen_width * padding) / graph_width
        scale_y = (self.screen_height * padding) / graph_height

        self.scale = min(scale_x, scale_y, 1.0)
        self.target_scale = self.scale

        # Center the view
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        self.offset.x = self.screen_width / 2 - center_x * self.scale
        self.offset.y = self.screen_height / 2 - center_y * self.scale
        self.target_offset = self.offset.copy()

    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(world_pos[0] * self.scale + self.offset.x)
        screen_y = int(world_pos[1] * self.scale + self.offset.y)
        return screen_x, screen_y

    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        try:
            world_x = (screen_pos[0] - self.offset.x) / self.scale
            world_y = (screen_pos[1] - self.offset.y) / self.scale
            return world_x, world_y
        except ZeroDivisionError:
            return 0.0, 0.0

    def update(self, delta_time: float, input_handler: 'InputHandler'):
        """Update camera based on input"""
        # Handle panning
        pan_speed = 500 * delta_time / max(self.scale, 0.1)

        if input_handler.is_key_pressed(pygame.K_a) or input_handler.is_key_pressed(pygame.K_LEFT):
            self.target_offset.x += pan_speed
        if input_handler.is_key_pressed(pygame.K_d) or input_handler.is_key_pressed(pygame.K_RIGHT):
            self.target_offset.x -= pan_speed
        if input_handler.is_key_pressed(pygame.K_w) or input_handler.is_key_pressed(pygame.K_UP):
            self.target_offset.y += pan_speed
        if input_handler.is_key_pressed(pygame.K_s) or input_handler.is_key_pressed(pygame.K_DOWN):
            self.target_offset.y -= pan_speed

        # Handle reset
        # REFACTOR: Fixed reset logic
        if input_handler.is_key_pressed(pygame.K_r):
            if self.graph:
                self.fit_to_graph(self.graph)

        # Handle mouse dragging
        if input_handler.is_dragging:
            dx, dy = input_handler.drag_delta
            self.target_offset.x += dx
            self.target_offset.y += dy

        # Smooth interpolation
        lerp_factor = 8.0 * delta_time
        self.scale += (self.target_scale - self.scale) * lerp_factor
        self.offset += (self.target_offset - self.offset) * lerp_factor

    def handle_zoom(self, zoom_center: Tuple[int, int], zoom_factor: float):
        """Handle zoom with center point"""
        old_scale = self.target_scale
        self.target_scale = max(0.1, min(5.0, self.target_scale * zoom_factor))

        if self.target_scale != old_scale:
            world_coords = self.screen_to_world(zoom_center)
            self.target_offset.x = zoom_center[0] - world_coords[0] * self.target_scale
            self.target_offset.y = zoom_center[1] - world_coords[1] * self.target_scale


class InputHandler:
    """Enhanced input handling system"""

    def __init__(self):
        self.keys_pressed = set()
        self.is_dragging = False
        self.drag_start = None
        self.drag_delta = (0, 0)
        self.last_mouse_pos = (0, 0)

    def handle_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.KEYDOWN:
            self.keys_pressed.add(event.key)
        elif event.type == pygame.KEYUP:
            self.keys_pressed.discard(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.is_dragging = True
                self.drag_start = event.pos
                self.last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.is_dragging = False
                self.drag_start = None
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                self.drag_delta = (event.pos[0] - self.last_mouse_pos[0],
                                   event.pos[1] - self.last_mouse_pos[1])
                self.last_mouse_pos = event.pos

    def update(self, delta_time: float):
        """Update input handler state"""
        self.drag_delta = (0, 0)  # Reset drag delta each frame

    def is_key_pressed(self, key):
        """Check if key is currently pressed"""
        return key in self.keys_pressed


class UIManager:
    """Enhanced UI management system"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.panels = {}
        self.widgets = {}

    def create_panel(self, name: str, rect: pygame.Rect, alpha: int = 200):
        """Create a UI panel"""
        surface = pygame.Surface((rect.width, rect.height))
        surface.set_alpha(alpha)
        surface.fill((0, 0, 0))
        self.panels[name] = {'surface': surface, 'rect': rect}
        return surface

    def add_widget(self, panel_name: str, widget_name: str, widget):
        """Add widget to panel"""
        if panel_name not in self.widgets:
            self.widgets[panel_name] = {}
        self.widgets[panel_name][widget_name] = widget


class AnimationManager:
    """Animation management system"""

    def __init__(self):
        self.animations = []
        self.active_animations = []

    def add_animation(self, animation):
        """Add animation to manager"""
        self.animations.append(animation)

    def update(self, delta_time: float):
        """Update all active animations"""
        for animation in self.active_animations[:]:
            animation.update(delta_time)
            if animation.is_finished():
                self.active_animations.remove(animation)


# ---------------------------
# Part 8: Enhanced CLI and Main Functions
# ---------------------------

def create_sample_od_pairs(stations: List[str], num_pairs: int,
                           max_depart_time: float = 1800.0) -> List[ODPair]:
    """Create sample OD pairs with realistic characteristics"""
    if not stations:
        return []  # Can't create pairs without stations

    od_pairs = []
    for i in range(num_pairs):
        origin = random.choice(stations)
        destination_pool = [s for s in stations if s != origin]

        if not destination_pool:
            # Handle case with only one station
            destination = origin
        else:
            destination = random.choice(destination_pool)

        # Realistic departure time distribution
        depart_time = random.uniform(0, max_depart_time)

        # Priority distribution (more normal priority trains)
        priority_weights = [0.05, 0.15, 0.70, 0.10]  # Emergency, High, Normal, Low
        priority = random.choices(list(Priority), weights=priority_weights)[0]

        # Train type distribution
        type_weights = [0.60, 0.25, 0.10, 0.04, 0.01]  # Passenger, Freight, Express, Local, Maintenance
        train_type = random.choices(list(TrainType), weights=type_weights)[0]

        # Additional characteristics
        passenger_count = random.randint(50, 500) if train_type == TrainType.PASSENGER else 0
        cargo_weight = random.uniform(1000, 5000) if train_type == TrainType.FREIGHT else 0

        od_pair = ODPair(
            origin=origin,
            destination=destination,
            depart_time=depart_time,
            priority=priority,
            train_type=train_type,
            passenger_count=passenger_count,
            cargo_weight=cargo_weight,
            max_speed=random.uniform(40, 80) if train_type == TrainType.EXPRESS else None
        )

        od_pairs.append(od_pair)

    # Sort by departure time
    od_pairs.sort(key=lambda x: x.depart_time)
    return od_pairs


def run_enhanced_simulation(datafile: str, config: SimulationConfig,
                            mode: SimulationMode = SimulationMode.HEADLESS) -> Dict[str, Any]:
    """Run enhanced simulation and return results"""

    # Create simulation
    sim = EnhancedSimulation(datafile=datafile, config=config, mode=mode)

    if not sim.is_running:
        return {"error": "Failed to initialize simulation"}

    # Generate OD pairs
    od_pairs = create_sample_od_pairs(
        stations=sim.stations,
        num_pairs=config.initial_trains,
        max_depart_time=config.episode_duration / 2
    )

    # Spawn trains
    sim.spawn_trains_from_ods(od_pairs)

    # Enable disruptions if configured
    if config.disruption_rate > 0:
        sim.schedule_random_disruptions()

    # Run simulation
    if mode == SimulationMode.VISUAL:
        sim.run_visual_simulation()
    else:
        sim.run_headless_episode(config.episode_duration)

    # Export results
    episode_name = f"ep_{config.seed if config.seed else int(time.time())}"
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exported_files = sim.recorder.export_data(out_dir=os.path.join(_root, "data", "episodes"), episode_name=episode_name)

    # Calculate final metrics
    final_metrics = {
        "episode_duration": sim.current_sim_time,
        "total_trains": len(sim.trains),
        "completed_trains": sum(1 for t in sim.trains if not t.is_visible),
        "collision_count": sim.collision_count,
        "disruption_count": sim.disruption_count,
        "network_utilization": sim.get_network_utilization(),
        "energy_efficiency": sim.get_energy_efficiency(),
        "passenger_satisfaction": sim.get_passenger_satisfaction(),
        "exported_files": exported_files
    }

    return final_metrics


def main_cli():
    """Enhanced CLI with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="Enhanced Railway Simulation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True, help="Available commands")

    # Visualize command
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_data = os.path.join(_root, "data", "processed", "godot_station_graph.json")
    vis_parser = subparsers.add_parser("visualize", help="Run simulation with enhanced visualization")
    vis_parser.add_argument("--datafile", default=_default_data,
                            help="Path to railway graph JSON file")
    vis_parser.add_argument("--initial-trains", type=int, default=20,
                            help="Number of trains to generate")
    vis_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    vis_parser.add_argument("--enable-disruptions", action="store_true",
                            help="Enable random track disruptions")
    vis_parser.add_argument("--enable-collisions", action="store_true",
                            help="Enable train-train collision detection")
    vis_parser.add_argument("--debug", action="store_true", help="Show debug information")
    vis_parser.add_argument("--no-animations", dest="animations", action="store_false",
                            help="Disable animations")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run headless simulation")
    sim_parser.add_argument("--datafile", default=_default_data,
                            help="Path to railway graph JSON file")
    sim_parser.add_argument("--duration", type=float, default=3600.0,
                            help="Episode duration in seconds")
    sim_parser.add_argument("--num-trains", type=int, default=100,
                            help="Number of trains to generate")
    sim_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    sim_parser.add_argument("--out-dir", default=os.path.join(_root, "data", "episodes"), help="Output directory")
    sim_parser.add_argument("--enable-disruptions", action="store_true",
                            help="Enable random track disruptions")
    sim_parser.add_argument("--export-format", choices=["jsonl", "parquet", "hdf5"],
                            default="jsonl", help="Export format")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run multiple simulations in batch")
    batch_parser.add_argument("--datafile", default=_default_data,
                              help="Path to railway graph JSON file")
    batch_parser.add_argument("--episodes", type=int, default=10,
                              help="Number of episodes to run")
    batch_parser.add_argument("--duration", type=float, default=3600.0,
                              help="Episode duration in seconds")
    batch_parser.add_argument("--num-trains", type=int, default=100,
                              help="Number of trains per episode")
    batch_parser.add_argument("--out-dir", default=os.path.join(_root, "data", "episodes"), help="Output directory")
    batch_parser.add_argument("--parallel", action="store_true",
                              help="Run episodes in parallel")
    batch_parser.add_argument("--max-workers", type=int, default=4,
                              help="Maximum parallel workers")

    args = parser.parse_args()

    if args.cmd == "visualize":
        if not PYGAME_AVAILABLE:
            print("Error: Pygame not available. Cannot run visualization.")
            return

        config = SimulationConfig(
            initial_trains=args.initial_trains,
            seed=args.seed,
            disruption_rate=0.05 if args.enable_disruptions else 0.0,
            collision_detection=args.enable_collisions,
            show_debug_info=args.debug,
            enable_animations=args.animations
        )

        print("Starting enhanced visual simulation...")
        run_enhanced_simulation(args.datafile, config, SimulationMode.VISUAL)

    elif args.cmd == "simulate":
        config = SimulationConfig(
            episode_duration=args.duration,
            initial_trains=args.num_trains,
            seed=args.seed,
            disruption_rate=0.05 if args.enable_disruptions else 0.0,
            export_format=args.export_format,
            collision_detection=False  # Collisions usually off for max speed
        )

        print("Starting enhanced headless simulation...")
        results = run_enhanced_simulation(args.datafile, config, SimulationMode.HEADLESS)

        print(f"\n--- Simulation Results ---")
        if "error" in results:
            print(f"Error: {results['error']}")
            return

        print(f"Duration: {results['episode_duration']:.2f} seconds")
        print(f"Total Trains: {results['total_trains']}")
        print(f"Completed Trains: {results['completed_trains']}")
        print(f"Collisions: {results['collision_count']}")
        print(f"Disruptions: {results['disruption_count']}")
        print(f"Network Utilization: {results['network_utilization']:.1%}")
        print(f"Energy Efficiency: {results['energy_efficiency']:.2f}")
        print(f"Passenger Satisfaction: {results['passenger_satisfaction']:.2f}")

        print(f"\nExported files:")
        for file_type, file_path in results['exported_files'].items():
            print(f"  {file_type}: {file_path}")

    elif args.cmd == "batch":
        print(f"Running batch simulation with {args.episodes} episodes...")

        # Create output directory
        os.makedirs(args.out_dir, exist_ok=True)

        if args.parallel and args.max_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                for i in range(args.episodes):
                    config = SimulationConfig(
                        episode_duration=args.duration,
                        initial_trains=args.num_trains,
                        seed=args.seed + i if args.seed else None,
                        disruption_rate=0.05
                    )
                    future = executor.submit(run_enhanced_simulation, args.datafile, config, SimulationMode.HEADLESS)
                    futures.append(future)

                # Collect results
                results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"Episode {i + 1}/{args.episodes} completed")
                    except Exception as e:
                        print(f"Episode {i + 1} failed: {e}")
        else:
            # Sequential execution
            results = []
            for i in range(args.episodes):
                config = SimulationConfig(
                    episode_duration=args.duration,
                    initial_trains=args.num_trains,
                    seed=args.seed + i if args.seed else None,
                    disruption_rate=0.05
                )
                result = run_enhanced_simulation(args.datafile, config, SimulationMode.HEADLESS)
                results.append(result)
                print(f"Episode {i + 1}/{args.episodes} completed")

        # Summary statistics
        if results and NUMPY_AVAILABLE:
            # Filter out error results
            valid_results = [r for r in results if "error" not in r]
            if not valid_results:
                print("\n--- Batch Results Summary ---")
                print("All episodes failed.")
                return

            print(f"\n--- Batch Results Summary ({len(valid_results)} successful episodes) ---")
            print(f"Average Duration: {np.mean([r['episode_duration'] for r in valid_results]):.2f}s")
            print(f"Average Trains: {np.mean([r['total_trains'] for r in valid_results]):.1f}")
            print(f"Average Utilization: {np.mean([r['network_utilization'] for r in valid_results]):.1%}")
            print(f"Total Collisions: {sum(r['collision_count'] for r in valid_results)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
