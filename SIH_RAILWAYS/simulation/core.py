"""
simulation/core.py — SimPy-based Railway Simulation with IR MACLS Signaling

This is the base simulation that:
  • Loads the graph JSON produced by graph/add_signals.py.
  • Uses Signal / BlockSection / CascadeSignalController from simulation/signals.py.
  • Optionally integrates DisruptionManager from simulation/disruptions.py.
  • Trains respect ALL signal types (not just starter), modulate speed by aspect,
    and trigger SPAD events when passing an absolute signal at RED.
  • Includes a DataRecorder for RL training data export.
  • Optional Pygame visualization (headless by default).

CLI:
    python -m simulation.core visualize --datafile PATH
    python -m simulation.core simulate --duration 3600
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import simpy

from simulation.signals import (
    BlockSection,
    CascadeSignalController,
    Signal,
    SignalAspect,
    SignalType,
    SPADDetector,
    StoppageRule,
    TrainType,
    build_signals_and_blocks_from_graph,
)

# Optional imports
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Type aliases
SimPyEnv = simpy.Environment
NetworkXGraph = nx.Graph


# ══════════════════════════════════════════════════════════════════════════════
# Configuration & Constants
# ══════════════════════════════════════════════════════════════════════════════

# Visualization
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 60
GRID_SIZE = 100
PAN_SPEED = 300
ZOOM_SPEED = 0.1
MIN_SCALE, MAX_SCALE = 0.05, 5.0

COLOR_BACKGROUND   = (12, 24, 35)
COLOR_GRID          = (40, 60, 80)
COLOR_TRACK_MAIN    = (85, 170, 85)
COLOR_DISRUPTION    = (255, 190, 0)
COLOR_STATION_FILL  = (255, 245, 220)
COLOR_STATION_BORDER = (70, 130, 180)
COLOR_JUNCTION_FILL  = (200, 200, 200)
COLOR_JUNCTION_BORDER = (100, 100, 100)
COLOR_TRAIN          = (255, 100, 80)
COLOR_TRAIN_SHADOW   = (40, 40, 40)
COLOR_SIGNAL_POST    = (160, 160, 160)
COLOR_TEXT           = (255, 255, 255)

ASPECT_COLORS = {
    SignalAspect.RED:           (255, 60, 60),
    SignalAspect.YELLOW:        (255, 220, 0),
    SignalAspect.DOUBLE_YELLOW: (255, 180, 0),
    SignalAspect.GREEN:         (60, 255, 60),
    SignalAspect.OFF:           (60, 60, 60),
}

# Simulation defaults
DEFAULTS: Dict[str, Any] = {
    "simulation_speed_factor": 150.0,
    "train_speed_pps": 12.0,       # base speed in graph-units per second
    "initial_trains": 20,
    "snapshot_interval": 1.0,
    "episode_duration": 3600.0,
    "seed": None,
    "headless": True,
    "visual": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ODPair:
    origin: str
    destination: str
    depart_time: float
    priority: int
    train_type: str


# ══════════════════════════════════════════════════════════════════════════════
# Data Recorder
# ══════════════════════════════════════════════════════════════════════════════

class DataRecorder:
    def __init__(self, snapshot_interval: float = 1.0):
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[Tuple[float, Dict[str, Any]]] = []
        self.events: List[Dict[str, Any]] = []
        self._last_snapshot_time: Optional[float] = None

    def record_event(self, sim_time: float, kind: str, payload: Dict[str, Any]):
        self.events.append({"time": float(sim_time), "kind": kind, "payload": deepcopy(payload)})

    def maybe_snapshot(self, sim_time: float, simulation: 'Simulation'):
        if self._last_snapshot_time is None:
            self._last_snapshot_time = sim_time - self.snapshot_interval
        if sim_time >= self._last_snapshot_time + self.snapshot_interval - 1e-9:
            snap = self._make_snapshot(sim_time, simulation)
            self.snapshots.append((float(sim_time), snap))
            self._last_snapshot_time = sim_time

    def _make_snapshot(self, sim_time: float, simulation: 'Simulation') -> Dict[str, Any]:
        G = simulation.graph
        nodes = {
            nid: {
                "pos": data.get("pos"),
                "label": data.get("label"),
                "is_station": bool(data.get("label")),
            }
            for nid, data in G.nodes(data=True)
        }

        edges = {}
        for u, v, d in G.edges(data=True):
            key = tuple(sorted((u, v)))
            block = simulation.blocks.get(key)
            edges[str(key)] = {
                "u": u, "v": v,
                "distance": d.get("distance", 0.0),
                "occupied": block.is_occupied if block else False,
                "disrupted": block.is_disrupted if block else False,
            }

        signals = {
            sid: {
                "aspect": s.aspect.value,
                "type": s.signal_type.value,
                "pos": s.position,
                "protected_block": list(s.controls_block),
                "direction": list(s.direction),
            }
            for sid, s in simulation.signals.items()
        }

        trains = [
            {
                "id": t.id,
                "pos": t.visual_pos,
                "route": list(t.route),
                "is_visible": t.is_visible,
                "current_index": t.current_index,
                "train_type": t.train_type,
                "speed": t.current_speed,
            }
            for t in simulation.trains
        ]

        meta = {
            "sim_time": float(sim_time),
            "active_trains": sum(1 for t in simulation.trains if t.is_visible),
        }
        return {"nodes": nodes, "edges": edges, "signals": signals, "trains": trains, "meta": meta}

    def get_and_clear_events(self) -> List[Dict[str, Any]]:
        events = list(self.events)
        self.events.clear()
        return events

    def export_jsonl(self, out_dir: str = "dataset", episode_name: str = "ep") -> Tuple[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        snaps_f = os.path.join(out_dir, f"{episode_name}_snapshots.jsonl")
        events_f = os.path.join(out_dir, f"{episode_name}_events.jsonl")
        with open(snaps_f, "w") as sf:
            for t, s in self.snapshots:
                sf.write(json.dumps({"time": t, "snapshot": s}) + "\n")
        with open(events_f, "w") as ef:
            for e in self.events:
                ef.write(json.dumps(e) + "\n")
        return snaps_f, events_f


# ══════════════════════════════════════════════════════════════════════════════
# Track Block  (SimPy Resource wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class TrackBlock:
    """
    Wraps a BlockSection with a SimPy PriorityResource to enforce
    the Absolute Block System: at most 1 train in the block at a time.
    """

    def __init__(self, env: SimPyEnv, block: BlockSection):
        self.block = block
        self.resource = simpy.PriorityResource(env, capacity=1)

    @property
    def is_occupied(self) -> bool:
        return self.block.is_occupied

    @is_occupied.setter
    def is_occupied(self, val: bool):
        self.block.is_occupied = val

    @property
    def is_disrupted(self) -> bool:
        return self.block.is_disrupted


# ══════════════════════════════════════════════════════════════════════════════
# Train Process
# ══════════════════════════════════════════════════════════════════════════════

class Train:
    """
    SimPy process representing a train moving along its route.

    The train checks EVERY signal it encounters (not just starters) and
    adjusts speed according to the signal aspect.
    """

    def __init__(
        self,
        env: SimPyEnv,
        train_id: str,
        route: List[str],
        graph: NetworkXGraph,
        simulation: 'Simulation',
        speed_pps: Optional[float] = None,
        train_type: str = "passenger",
        priority: int = 2,
    ):
        self.env = env
        self.id = train_id
        self.route = route
        self.graph = graph
        self.simulation = simulation
        self.speed_pps = speed_pps or simulation.base_speed
        self.train_type = train_type
        self.priority = priority

        self.visual_pos: Tuple[float, float] = graph.nodes[route[0]]["pos"]
        self.is_visible: bool = True
        self.current_index: int = 0
        self.current_speed: float = self.speed_pps

        self.process = env.process(self.run())

    def run(self):
        env = self.env
        sim = self.simulation
        rec = sim.recorder

        rec.record_event(env.now, "spawn", {"train_id": self.id, "route": list(self.route)})

        for i in range(len(self.route) - 1):
            self.current_index = i
            current_node = self.route[i]
            next_node = self.route[i + 1]
            direction = (current_node, next_node)
            block_key = tuple(sorted(direction))

            # ── Check all signals for this direction on this edge ────────
            edge_signals = self._get_signals_for_direction(direction)

            for sig in edge_signals:
                yield from self._handle_signal(sig)

            # ── Request the block ────────────────────────────────────────
            tb = sim.track_blocks.get(block_key)
            if tb is None:
                rec.record_event(env.now, "error", {
                    "train_id": self.id,
                    "error": f"missing_block_{block_key}",
                })
                break

            with tb.resource.request(priority=self.priority) as req:
                yield req
                tb.is_occupied = True
                tb.block.occupying_train = self.id
                rec.record_event(env.now, "enter_edge", {
                    "train_id": self.id,
                    "edge": block_key,
                })

                # ── Travel the edge with speed modulation ────────────────
                start_pos = self.graph.nodes[current_node]["pos"]
                end_pos = self.graph.nodes[next_node]["pos"]
                distance = self.graph.edges[current_node, next_node]["distance"]

                # Speed is modulated by the most restrictive signal on this edge
                speed_frac = self._compute_speed_fraction(edge_signals)
                effective_speed = max(0.1, self.speed_pps * speed_frac)
                self.current_speed = effective_speed

                travel_time = max(0.1, distance / effective_speed)
                start_time = env.now

                while env.now < start_time + travel_time:
                    progress = min(1.0, (env.now - start_time) / travel_time)
                    self.visual_pos = (
                        start_pos[0] + (end_pos[0] - start_pos[0]) * progress,
                        start_pos[1] + (end_pos[1] - start_pos[1]) * progress,
                    )
                    yield env.timeout(0.1)

                self.visual_pos = end_pos
                tb.is_occupied = False
                tb.block.occupying_train = None
                rec.record_event(env.now, "leave_edge", {
                    "train_id": self.id,
                    "edge": block_key,
                })

        self.is_visible = False
        self.current_speed = 0.0
        rec.record_event(env.now, "finish", {"train_id": self.id})

    # ── signal helpers ───────────────────────────────────────────────────

    def _get_signals_for_direction(self, direction: Tuple[str, str]) -> List[Signal]:
        """Get all signals governing this direction, ordered by approach distance."""
        result = []
        for sig in self.simulation.signals.values():
            if sig.direction == direction:
                result.append(sig)
        # Sort: Distant first, then Inner Distant, Home, Starter, Adv Starter
        type_order = {
            SignalType.INNER_DISTANT: 0,
            SignalType.DISTANT: 1,
            SignalType.HOME: 2,
            SignalType.STARTER: 3,
            SignalType.ADVANCED_STARTER: 4,
            SignalType.INTERMEDIATE_BLOCK: 2,
        }
        result.sort(key=lambda s: type_order.get(s.signal_type, 5))
        return result

    def _handle_signal(self, sig: Signal):
        """
        Handle a signal according to its stoppage rule.

        ABSOLUTE:   Wait until not RED.
        PERMISSIVE: Wait up to 60 s, then proceed at restricted speed.
        ADVISORY:   Record aspect, modulate speed (don't stop).
        """
        env = self.env
        rec = self.simulation.recorder

        if sig.stoppage_rule == StoppageRule.ABSOLUTE:
            # Must stop at RED
            if sig.aspect == SignalAspect.RED or sig.aspect == SignalAspect.OFF:
                # SPAD check — we're approaching, if it's RED we wait
                wait_start = env.now
                rec.record_event(env.now, "wait_start", {
                    "train_id": self.id,
                    "signal": sig.id,
                    "signal_type": sig.signal_type.value,
                })
                while sig.aspect == SignalAspect.RED or sig.aspect == SignalAspect.OFF:
                    yield env.timeout(1.0)
                wait_dur = env.now - wait_start
                rec.record_event(env.now, "wait_end", {
                    "train_id": self.id,
                    "signal": sig.id,
                    "duration": wait_dur,
                })

        elif sig.stoppage_rule == StoppageRule.PERMISSIVE:
            if sig.aspect == SignalAspect.RED:
                wait_start = env.now
                timeout = 60.0  # max wait at permissive signal
                while sig.aspect == SignalAspect.RED and (env.now - wait_start) < timeout:
                    yield env.timeout(1.0)
                if sig.aspect == SignalAspect.RED:
                    # Proceed at restricted speed (recorded as caution_proceed)
                    rec.record_event(env.now, "caution_proceed", {
                        "train_id": self.id,
                        "signal": sig.id,
                    })

        # ADVISORY: no stop, just note it (speed modulation handled elsewhere)

    def _compute_speed_fraction(self, edge_signals: List[Signal]) -> float:
        """
        Compute effective speed fraction from the most restrictive signal
        encountered on this edge.
        """
        if not edge_signals:
            return 1.0
        return min(sig.aspect.speed_fraction for sig in edge_signals) or 0.25  # floor at caution speed


# ══════════════════════════════════════════════════════════════════════════════
# Graph Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_railway_graph(filepath: str):
    """
    Load graph JSON and return (NetworkX graph, edges_data, signals_data).
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: '{filepath}' not found.")
        return None, [], []

    G = nx.Graph()
    all_signals = []
    edges_data = data.get("edges", [])

    for node in data.get("nodes", []):
        nid = str(node["id"])
        G.add_node(nid, pos=(float(node["pos_x"]), float(node["pos_y"])), label=node.get("label"))

    for edge in edges_data:
        s, t = str(edge["source"]), str(edge["target"])
        if G.has_node(s) and G.has_node(t):
            G.add_edge(s, t, distance=float(edge.get("distance", 1.0)))
            if edge.get("signals"):
                all_signals.extend(edge["signals"])

    return G, edges_data, all_signals


# ══════════════════════════════════════════════════════════════════════════════
# Simulation Controller
# ══════════════════════════════════════════════════════════════════════════════

class Simulation:
    def __init__(
        self,
        datafile: str = None,
        headless: bool = True,
        recorder: Optional[DataRecorder] = None,
        cfg: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg or DEFAULTS.copy()
        self.seed = seed if seed is not None else self.cfg.get("seed")
        if self.seed is not None:
            random.seed(self.seed)

        # Resolve default data path
        if datafile is None:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            datafile = os.path.join(_root, "data", "processed", "godot_station_graph_with_signals.json")

        self.recorder = recorder or DataRecorder(snapshot_interval=self.cfg.get("snapshot_interval", 1.0))
        self.base_speed = self.cfg.get("train_speed_pps", DEFAULTS["train_speed_pps"])
        self.simulation_speed_factor = self.cfg.get("simulation_speed_factor", DEFAULTS["simulation_speed_factor"])

        self.env: SimPyEnv = simpy.Environment()

        # Load graph + signals
        result = load_railway_graph(datafile)
        if result[0] is None:
            self.is_running = False
            self.graph = None
            return

        self.graph, edges_data, signals_data = result
        self.is_running = True
        self.current_sim_time = 0.0

        self.stations = [nid for nid, data in self.graph.nodes(data=True) if data.get("label")]

        # Build signal & block objects from the new data model
        self.signals, blocks = build_signals_and_blocks_from_graph(edges_data, signals_data)
        self.blocks = blocks

        # Wrap BlockSections into SimPy TrackBlocks
        self.track_blocks: Dict[Tuple[str, str], TrackBlock] = {
            bk: TrackBlock(self.env, bs) for bk, bs in self.blocks.items()
        }

        # Create cascade signal controller
        self.signal_controller = CascadeSignalController(
            signals=self.signals,
            blocks=self.blocks,
            tick_interval=1.0,
        )

        # SPAD detector
        self.spad_detector = SPADDetector(self.signals)

        # Train list
        self.trains: List[Train] = []

        # Start the signal controller process
        self.env.process(self._signal_controller_process())

        # Visualization
        self.visual = (not headless) and PYGAME_AVAILABLE
        if self.visual:
            self._init_visual()
        else:
            self.screen = self.clock = self.offset = self.target_offset = None

    def _signal_controller_process(self):
        """SimPy process: tick the cascade controller every second."""
        while True:
            changed = self.signal_controller.tick(self.env.now)
            if changed:
                self.recorder.record_event(self.env.now, "signal_change", {
                    "changed_signals": changed,
                    "count": len(changed),
                })
            yield self.env.timeout(1.0)

    # ── OD pair generation & train spawning ──────────────────────────────

    def sample_od_pairs(
        self,
        stations: List[str],
        num_pairs: int,
        max_depart_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not stations:
            return []
        if max_depart_time is None:
            max_depart_time = max(1.0, self.cfg.get("episode_duration", 3600.0) / 2.0)

        ods: List[Dict[str, Any]] = []
        for _ in range(num_pairs):
            origin = random.choice(stations)
            possible_dest = [s for s in stations if s != origin]
            if not possible_dest:
                continue
            destination = random.choice(possible_dest)
            depart_time = random.uniform(0, max_depart_time)
            priority = random.choices([0, 1, 2], weights=[0.1, 0.2, 0.7])[0]
            train_type = random.choice(["passenger", "freight", "express"])
            ods.append({
                "origin": origin,
                "destination": destination,
                "depart_time": depart_time,
                "priority": priority,
                "type": train_type,
            })
        ods.sort(key=lambda x: x["depart_time"])
        return ods

    def spawn_trains_from_ods(self, ods: List[Dict[str, Any]]):
        def _spawn_proc(env, sim, od_list):
            id_counter = len(sim.trains)
            for od in od_list:
                yield env.timeout(max(0.0, od["depart_time"] - env.now))
                origin, dest = od["origin"], od["destination"]
                try:
                    route = nx.shortest_path(sim.graph, origin, dest, weight="distance")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(route) > 1:
                    speed = sim.base_speed * (1.0 + random.gauss(0, 0.05))
                    if od["type"] == "freight":
                        speed *= 0.5
                    elif od["type"] == "express":
                        speed *= 1.1
                    t = Train(
                        env, f"OD{id_counter}", route, sim.graph, sim,
                        speed_pps=speed, train_type=od["type"], priority=od["priority"],
                    )
                    sim.trains.append(t)
                    id_counter += 1

        self.env.process(_spawn_proc(self.env, self, ods))

    # ── Disruption scheduling (simple Poisson) ───────────────────────────

    def schedule_random_disruptions(self, disruption_rate: float = 0.001, mean_duration: float = 120.0):
        """Schedule disruptions using the DisruptionManager if available, else simple Poisson."""
        try:
            from simulation.disruptions import DisruptionManager, DisruptionConfig
            config = DisruptionConfig(
                enabled=True,
                base_rate=disruption_rate,
                seed=self.seed,
            )
            self._disruption_mgr = DisruptionManager(
                config=config,
                blocks=self.blocks,
                signals=self.signals,
                event_callback=lambda kind, payload: self.recorder.record_event(
                    self.env.now, kind, payload
                ),
            )
            self.env.process(self._disruption_mgr.run(self.env))
            self.env.process(self._disruption_mgr.auto_resolve(self.env))
        except ImportError:
            # Fallback: simple disruption process
            self.env.process(self._simple_disruption_process(disruption_rate, mean_duration))

    def _simple_disruption_process(self, rate, mean_dur):
        block_keys = list(self.blocks.keys())
        if not block_keys:
            return
        while True:
            yield self.env.timeout(random.expovariate(max(1e-9, rate)))
            bk = random.choice(block_keys)
            block = self.blocks[bk]
            if block.is_disrupted:
                continue
            duration = max(1.0, random.expovariate(1.0 / mean_dur))
            block.is_disrupted = True
            self.recorder.record_event(self.env.now, "disruption_start", {"block": bk, "duration": duration})
            yield self.env.timeout(duration)
            block.is_disrupted = False
            self.recorder.record_event(self.env.now, "disruption_end", {"block": bk})

    # ── Simulation advancement ───────────────────────────────────────────

    def advance(self, sim_seconds: float):
        target = self.current_sim_time + sim_seconds
        self.env.run(until=target)
        self.current_sim_time = target
        self.recorder.maybe_snapshot(self.current_sim_time, self)

    def run_headless_episode(self, duration: float, snapshot_interval: Optional[float] = None):
        snapshot_interval = snapshot_interval or self.recorder.snapshot_interval
        t = 0.0
        while t < duration:
            self.advance(snapshot_interval)
            t += snapshot_interval

    # ── RL agent interface ───────────────────────────────────────────────

    def apply_signal_actions(self, signal_green_mask: Dict[str, bool], sim_time: float):
        """
        Apply RL agent actions: for each edge, decide whether to set signals
        to GREEN (allow passage) or keep them RED.

        This is a per-edge action space:
            signal_green_mask = {block_key_str: True/False}
            True  = set all entry signals for that block to GREEN
            False = leave them controlled by the cascade controller
        """
        for block_key_str, allow_green in signal_green_mask.items():
            if allow_green:
                # Find signals protecting this block and set them green
                for sig in self.signals.values():
                    bk_str = str(sorted(sig.controls_block))
                    if bk_str == block_key_str:
                        if sig.signal_type in (SignalType.STARTER, SignalType.ADVANCED_STARTER, SignalType.HOME):
                            sig.set_aspect(SignalAspect.GREEN, sim_time)

    @property
    def is_complete(self) -> bool:
        return all(not t.is_visible for t in self.trains)

    # ══════════════════════════════════════════════════════════════════════
    # Visualization (Pygame)
    # ══════════════════════════════════════════════════════════════════════

    def _init_visual(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Railway Simulation — IR MACLS")
        self.font_station = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 12)
        self.clock = pygame.time.Clock()
        self.scale = 0.5
        self.offset = pygame.Vector2(0, 0)
        self.target_scale = self.scale
        self.target_offset = pygame.Vector2(0, 0)
        self.bounds = self._calculate_bounds()
        self.center_view()
        self.keys_pressed = set()
        self.mouse_dragging = False
        self.mouse_drag_start = None

    def _calculate_bounds(self):
        valid = [d["pos"] for _, d in self.graph.nodes(data=True) if "pos" in d]
        if not valid:
            return 0, 1, 0, 1
        return (
            min(p[0] for p in valid), max(p[0] for p in valid),
            min(p[1] for p in valid), max(p[1] for p in valid),
        )

    def center_view(self):
        if not self.visual:
            return
        min_x, max_x, min_y, max_y = self.bounds
        gw, gh = max_x - min_x, max_y - min_y
        if gw == 0 or gh == 0:
            return
        padding = 0.8
        sx = (SCREEN_WIDTH * padding) / gw
        sy = (SCREEN_HEIGHT * padding) / gh
        self.scale = min(sx, sy, 1.0)
        self.target_scale = self.scale
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        self.offset.x = SCREEN_WIDTH / 2 - cx * self.scale
        self.offset.y = SCREEN_HEIGHT / 2 - cy * self.scale
        self.target_offset = self.offset.copy()

    def _transform_pos(self, pos):
        if not self.visual:
            return int(pos[0]), int(pos[1])
        return int(pos[0] * self.scale + self.offset.x), int(pos[1] * self.scale + self.offset.y)

    def _screen_to_world(self, screen_pos):
        return (
            (screen_pos[0] - self.offset.x) / self.scale,
            (screen_pos[1] - self.offset.y) / self.scale,
        )

    def handle_input(self, delta_time):
        if not self.visual:
            return
        pan_delta = PAN_SPEED * delta_time / max(0.001, self.scale)
        if pygame.K_a in self.keys_pressed or pygame.K_LEFT in self.keys_pressed:
            self.target_offset.x += pan_delta
        if pygame.K_d in self.keys_pressed or pygame.K_RIGHT in self.keys_pressed:
            self.target_offset.x -= pan_delta
        if pygame.K_w in self.keys_pressed or pygame.K_UP in self.keys_pressed:
            self.target_offset.y += pan_delta
        if pygame.K_s in self.keys_pressed or pygame.K_DOWN in self.keys_pressed:
            self.target_offset.y -= pan_delta
        if pygame.K_r in self.keys_pressed or pygame.K_HOME in self.keys_pressed:
            self.center_view()
        if self.mouse_dragging and self.mouse_drag_start:
            cur = pygame.mouse.get_pos()
            dx, dy = cur[0] - self.mouse_drag_start[0], cur[1] - self.mouse_drag_start[1]
            self.target_offset.x += dx
            self.target_offset.y += dy
            self.mouse_drag_start = cur
        lerp = 8.0 * delta_time
        self.scale += (self.target_scale - self.scale) * lerp
        self.offset += (self.target_offset - self.offset) * lerp

    def handle_zoom(self, center, factor):
        if not self.visual:
            return
        old = self.target_scale
        self.target_scale = max(MIN_SCALE, min(MAX_SCALE, self.target_scale * factor))
        if self.target_scale != old:
            wc = self._screen_to_world(center)
            self.target_offset.x = center[0] - wc[0] * self.target_scale
            self.target_offset.y = center[1] - wc[1] * self.target_scale

    def draw(self):
        if not self.visual:
            return
        self.screen.fill(COLOR_BACKGROUND)
        self._draw_tracks()
        self._draw_nodes()
        self._draw_signals()
        self._draw_trains()
        self._draw_ui()
        pygame.display.flip()

    def _draw_tracks(self):
        tw = max(2, int(3 * self.scale))
        for u, v, d in self.graph.edges(data=True):
            sp = self._transform_pos(self.graph.nodes[u]["pos"])
            ep = self._transform_pos(self.graph.nodes[v]["pos"])
            if max(sp[0], ep[0]) < 0 or min(sp[0], ep[0]) > SCREEN_WIDTH:
                continue
            if max(sp[1], ep[1]) < 0 or min(sp[1], ep[1]) > SCREEN_HEIGHT:
                continue
            key = tuple(sorted((u, v)))
            block = self.blocks.get(key)
            color = COLOR_TRACK_MAIN
            if block and block.is_disrupted and int(self.current_sim_time * 2) % 2 == 0:
                color = COLOR_DISRUPTION
            shadow = max(1, int(self.scale))
            pygame.draw.line(self.screen, (40, 40, 40), (sp[0] + shadow, sp[1] + shadow), (ep[0] + shadow, ep[1] + shadow), tw)
            pygame.draw.line(self.screen, color, sp, ep, tw)

    def _draw_nodes(self):
        sr = max(4, int(8 * self.scale))
        jr = max(2, int(5 * self.scale))
        for nid, d in self.graph.nodes(data=True):
            pos = self._transform_pos(d["pos"])
            if pos[0] < -sr or pos[0] > SCREEN_WIDTH + sr or pos[1] < -sr or pos[1] > SCREEN_HEIGHT + sr:
                continue
            if d.get("label"):
                pygame.draw.circle(self.screen, COLOR_STATION_FILL, pos, sr)
                pygame.draw.circle(self.screen, COLOR_STATION_BORDER, pos, sr, max(1, int(self.scale)))
                if self.scale > 0.3:
                    lbl = self.font_station.render(d["label"], True, COLOR_TEXT)
                    self.screen.blit(lbl, lbl.get_rect(midleft=(pos[0] + sr + 5, pos[1])))
            else:
                pygame.draw.circle(self.screen, COLOR_JUNCTION_FILL, pos, jr)
                pygame.draw.circle(self.screen, COLOR_JUNCTION_BORDER, pos, jr, max(1, int(self.scale)))

    def _draw_signals(self):
        sig_r = max(3, int(6 * self.scale))
        post_h = sig_r * 2.5
        for sig in self.signals.values():
            pos = self._transform_pos(sig.position)
            if pos[0] < -sig_r or pos[0] > SCREEN_WIDTH + sig_r:
                continue
            if pos[1] < -post_h or pos[1] > SCREEN_HEIGHT + post_h:
                continue
            post_rect = pygame.Rect(pos[0] - 2, pos[1], 4, -int(post_h))
            pygame.draw.rect(self.screen, COLOR_SIGNAL_POST, post_rect)
            light_pos = (pos[0], int(pos[1] - post_h + sig_r))
            color = ASPECT_COLORS.get(sig.aspect, ASPECT_COLORS[SignalAspect.OFF])
            pygame.draw.circle(self.screen, color, light_pos, sig_r)
            pygame.draw.circle(self.screen, (200, 200, 200), light_pos, sig_r, 1)

    def _draw_trains(self):
        ts = max(6, int(12 * self.scale))
        for t in self.trains:
            if not t.is_visible:
                continue
            pos = self._transform_pos(t.visual_pos)
            if pos[0] < -ts or pos[0] > SCREEN_WIDTH + ts:
                continue
            if pos[1] < -ts or pos[1] > SCREEN_HEIGHT + ts:
                continue
            rect = pygame.Rect(pos[0] - ts // 2, pos[1] - ts // 2, ts, ts)
            pygame.draw.rect(self.screen, COLOR_TRAIN_SHADOW, rect.move(2, 2))
            pygame.draw.rect(self.screen, COLOR_TRAIN, rect)
            pygame.draw.rect(self.screen, (40, 40, 40), rect, max(1, int(self.scale)))

    def _draw_ui(self):
        ui_s = pygame.Surface((380, 120))
        ui_s.set_alpha(200)
        ui_s.fill((0, 0, 0))
        self.screen.blit(ui_s, (10, 10))
        h = int(self.current_sim_time // 3600)
        m = int((self.current_sim_time % 3600) // 60)
        s = int(self.current_sim_time % 60)
        tt = self.font_ui.render(f"Time: {h:02}:{m:02}:{s:02}", True, COLOR_TEXT)
        self.screen.blit(tt, (20, 20))
        active = sum(1 for t in self.trains if t.is_visible)
        at = self.font_small.render(f"Active Trains: {active}", True, COLOR_TEXT)
        self.screen.blit(at, (20, 50))
        ct = self.font_small.render("Controls: Wheel zoom | Drag pan | R reset | WASD", True, COLOR_TEXT)
        self.screen.blit(ct, (20, 80))
        zt = self.font_small.render(f"Zoom: {self.scale:.2f}x", True, COLOR_TEXT)
        self.screen.blit(zt, (20, 100))

    def run_simulation(self):
        if not self.visual:
            raise RuntimeError("Cannot run_simulation() in headless mode.")
        while self.is_running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    self.keys_pressed.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_pressed.discard(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mouse_dragging, self.mouse_drag_start = True, event.pos
                    elif event.button == 4:
                        self.handle_zoom(event.pos, 1 + ZOOM_SPEED)
                    elif event.button == 5:
                        self.handle_zoom(event.pos, 1 - ZOOM_SPEED)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_dragging = False
            self.handle_input(dt)
            sim_delta = dt * self.simulation_speed_factor
            self.advance(sim_delta)
            self.draw()
        pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
# Stubs for backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class LGNNEnv:
    pass


def run_episode_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    return {}


def batch_generate(datafile: str, out_dir: str, episodes: int = 10, **kwargs) -> List[Dict[str, Any]]:
    return []


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main_cli():
    parser = argparse.ArgumentParser(
        description="Railway Simulation with IR MACLS Signaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True, help="Available commands")

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_data = os.path.join(_root, "data", "processed", "godot_station_graph_with_signals.json")

    # ── visualize ────────────────────────────────────────────────────────
    p_vis = subparsers.add_parser("visualize", help="Run with pygame visualization")
    p_vis.add_argument("--datafile", default=_default_data)
    p_vis.add_argument("--initial-trains", type=int, default=DEFAULTS["initial_trains"])
    p_vis.add_argument("--seed", type=int, default=None)
    p_vis.add_argument("--enable-disruptions", action="store_true")

    # ── simulate ─────────────────────────────────────────────────────────
    p_sim = subparsers.add_parser("simulate", help="Run headless episode and export data")
    p_sim.add_argument("--datafile", default=_default_data)
    p_sim.add_argument("--duration", type=float, default=DEFAULTS["episode_duration"])
    p_sim.add_argument("--snapshot-interval", type=float, default=DEFAULTS["snapshot_interval"])
    p_sim.add_argument("--num-trains", type=int, default=100)
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.add_argument("--out-dir", default=os.path.join(_root, "data", "episodes"))
    p_sim.add_argument("--episode-name", default=None)
    p_sim.add_argument("--enable-disruptions", action="store_true")

    args = parser.parse_args()

    if args.cmd == "visualize":
        if not PYGAME_AVAILABLE:
            print("Pygame not available.")
            return
        cfg = DEFAULTS.copy()
        cfg["initial_trains"] = args.initial_trains
        sim = Simulation(datafile=args.datafile, headless=False, cfg=cfg, seed=args.seed)
        if sim.is_running:
            ods = sim.sample_od_pairs(sim.stations, args.initial_trains, max_depart_time=5.0)
            sim.spawn_trains_from_ods(ods)
            if args.enable_disruptions:
                sim.schedule_random_disruptions()
            sim.run_simulation()

    elif args.cmd == "simulate":
        ep_name = args.episode_name or f"ep_{args.seed or int(time.time())}"
        cfg = {"snapshot_interval": args.snapshot_interval, "episode_duration": args.duration}
        rec = DataRecorder(snapshot_interval=args.snapshot_interval)
        sim = Simulation(datafile=args.datafile, headless=True, recorder=rec, cfg=cfg, seed=args.seed)
        if sim.is_running:
            ods = sim.sample_od_pairs(sim.stations, args.num_trains)
            sim.spawn_trains_from_ods(ods)
            if args.enable_disruptions:
                sim.schedule_random_disruptions()
            sim.run_headless_episode(duration=args.duration)
            completed = sum(1 for e in rec.events if e["kind"] == "finish")
            print(f"\n--- Simulation Summary ---")
            print(f"Episode '{ep_name}' finished. Duration: {sim.current_sim_time:.2f}s")
            print(f"Completed: {completed}  |  Events: {len(rec.events)}")
            paths = rec.export_jsonl(out_dir=args.out_dir, episode_name=ep_name)
            print(f"Exported → {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    main_cli()
