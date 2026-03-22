"""
simulation/signals.py — Single Source of Truth for Indian Railway MACLS Signaling

Implements the Multiple Aspect Colour Light Signaling (MACLS) system as defined
by Indian Railway standards. This module provides:

  • SignalAspect   – 4-aspect + OFF enum (RED, YELLOW, DOUBLE_YELLOW, GREEN, OFF)
  • StoppageRule   – What a driver must do when encountering each signal type at RED
  • SignalType     – 6 IR signal types with is_permissive & stoppage_rule attrs
  • Signal         – Data model for a single signal instance on the graph
  • BlockSection   – Represents a block section between two stations (Absolute Block System)
  • CascadeSignalController – Implements 4-aspect cascading + interlocking logic
  • IR placement constants  – Canonical distances for signal placement

All other modules (simulation/core.py, simulation/enhanced.py, training/train_rl.py,
models/lgnn.py) MUST import signal definitions exclusively from this module.

Reference:  Indian Railway MACLS 4-aspect signaling, IRISET documentation,
            SRS for KAVACH (TCAS) Level-2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class SignalAspect(Enum):
    """
    MACLS 4-aspect colour light signals as per Indian Railway standards.

    Cascading semantics (read by the driver):
        RED           → STOP.  Do not pass the signal.
        YELLOW        → CAUTION.  Next signal is RED; be prepared to stop.
        DOUBLE_YELLOW → ATTENTION. Next signal is YELLOW; reduce speed.
        GREEN         → PROCEED.  Line clear up to the next signal.
        OFF           → Signal not in service / lamp failure.
    """
    RED           = "red"
    YELLOW        = "yellow"
    DOUBLE_YELLOW = "double_yellow"
    GREEN         = "green"
    OFF           = "off"

    # ── speed limits (fraction of track line-speed) ──────────────────────
    @property
    def speed_fraction(self) -> float:
        """Fraction of line-speed the driver should observe."""
        return {
            SignalAspect.RED:           0.0,     # full stop
            SignalAspect.YELLOW:        0.25,    # caution — typically ≤25 km/h approach
            SignalAspect.DOUBLE_YELLOW: 0.60,    # attention — reduced speed
            SignalAspect.GREEN:         1.0,     # full line-speed
            SignalAspect.OFF:           0.0,     # treat as most restrictive
        }[self]

    # ── numeric ordering for "most restrictive" comparisons ──────────────
    @property
    def restrictiveness(self) -> int:
        """Lower = more restrictive.  Useful for min() comparisons."""
        return {
            SignalAspect.RED: 0,
            SignalAspect.OFF: 0,
            SignalAspect.YELLOW: 1,
            SignalAspect.DOUBLE_YELLOW: 2,
            SignalAspect.GREEN: 3,
        }[self]

    def __lt__(self, other):
        if not isinstance(other, SignalAspect):
            return NotImplemented
        return self.restrictiveness < other.restrictiveness

    def __le__(self, other):
        if not isinstance(other, SignalAspect):
            return NotImplemented
        return self.restrictiveness <= other.restrictiveness


class StoppageRule(Enum):
    """
    What the driver/train-process must do when encountering this signal at RED.

    ABSOLUTE   – Must stop; cannot pass RED under any circumstance (Home, Starter, Adv Starter).
    PERMISSIVE – Stop, then proceed at restricted speed if no authority (Intermediate Block Signal).
    ADVISORY   – Reduce speed / prepare to stop but may proceed (Distant, Inner Distant).
    """
    ABSOLUTE   = auto()
    PERMISSIVE = auto()
    ADVISORY   = auto()


class SignalType(Enum):
    """
    Six Indian Railway signal types placed around a station.

    Layout along direction of travel  →  :

        DISTANT ── INNER_DISTANT ── HOME ── [STATION] ── STARTER ── ADV_STARTER
        (advisory)  (advisory)   (absolute)            (absolute)   (absolute)

    INTERMEDIATE_BLOCK sits between two stations on long block sections where
    intermediate block signaling is provided.
    """
    DISTANT             = "distant"
    INNER_DISTANT       = "inner_distant"
    HOME                = "home"
    STARTER             = "starter"
    ADVANCED_STARTER    = "advanced_starter"
    INTERMEDIATE_BLOCK  = "intermediate_block"

    # ── properties ───────────────────────────────────────────────────────
    @property
    def is_permissive(self) -> bool:
        """Can a train pass this signal at RED (under caution)?"""
        return self in (
            SignalType.DISTANT,
            SignalType.INNER_DISTANT,
            SignalType.INTERMEDIATE_BLOCK,
        )

    @property
    def stoppage_rule(self) -> StoppageRule:
        _map = {
            SignalType.DISTANT:            StoppageRule.ADVISORY,
            SignalType.INNER_DISTANT:      StoppageRule.ADVISORY,
            SignalType.HOME:               StoppageRule.ABSOLUTE,
            SignalType.STARTER:            StoppageRule.ABSOLUTE,
            SignalType.ADVANCED_STARTER:   StoppageRule.ABSOLUTE,
            SignalType.INTERMEDIATE_BLOCK: StoppageRule.PERMISSIVE,
        }
        return _map[self]

    @property
    def default_aspect(self) -> SignalAspect:
        """Default (most restrictive safe) aspect when the signal is first placed."""
        _map = {
            SignalType.DISTANT:            SignalAspect.DOUBLE_YELLOW,
            SignalType.INNER_DISTANT:      SignalAspect.GREEN,
            SignalType.HOME:               SignalAspect.RED,
            SignalType.STARTER:            SignalAspect.RED,
            SignalType.ADVANCED_STARTER:   SignalAspect.RED,
            SignalType.INTERMEDIATE_BLOCK: SignalAspect.RED,
        }
        return _map[self]


class TrainType(Enum):
    """Train types with IR priority ordering."""
    RAJDHANI    = "rajdhani"
    SHATABDI    = "shatabdi"
    EXPRESS     = "express"
    PASSENGER   = "passenger"
    FREIGHT     = "freight"
    MAINTENANCE = "maintenance"

    @property
    def priority(self) -> int:
        """Lower number = higher priority."""
        return {
            TrainType.RAJDHANI:    0,
            TrainType.SHATABDI:    0,
            TrainType.EXPRESS:     1,
            TrainType.PASSENGER:   2,
            TrainType.FREIGHT:     3,
            TrainType.MAINTENANCE: 4,
        }[self]

    @property
    def speed_multiplier(self) -> float:
        """Fraction of base line-speed for this train type."""
        return {
            TrainType.RAJDHANI:    1.0,
            TrainType.SHATABDI:    1.0,
            TrainType.EXPRESS:     0.85,
            TrainType.PASSENGER:   0.70,
            TrainType.FREIGHT:     0.50,
            TrainType.MAINTENANCE: 0.30,
        }[self]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  IR SIGNAL PLACEMENT CONSTANTS  (metres)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IRPlacementConstants:
    """
    Indian Railway standard distances for MACLS signal placement.

    Incoming signals (measured BACK from station entry / facing point):
        HOME:          180 – 400 m before station entry
        DISTANT:       1000 m before Home signal
        INNER_DISTANT: 1000 m before Distant signal (double-distant territory)

    Outgoing signals (measured FORWARD from station limit / starter post):
        STARTER:         80 m after the station platform end
        ADV_STARTER:    250 m after the Starter signal

    Overlap:
        SIGNAL_OVERLAP:  120 m minimum overlap past signal for MACLS
    """
    # Incoming (towards station)
    HOME_FROM_STATION:           float = 400.0    # Home placed 400 m before station entry
    DISTANT_FROM_HOME:           float = 1000.0   # Distant placed 1000 m before Home
    INNER_DISTANT_FROM_DISTANT:  float = 1000.0   # Inner Distant 1000 m before Distant

    # Outgoing (leaving station)
    STARTER_FROM_STATION:        float = 80.0     # Starter 80 m after platform end
    ADV_STARTER_FROM_STARTER:    float = 250.0    # Advanced Starter 250 m after Starter

    # Overlap
    SIGNAL_OVERLAP_M:            float = 120.0    # MACLS minimum overlap

    # Intermediate block signals (long sections)
    INTERMEDIATE_BLOCK_MIN_GAP:  float = 2000.0   # Minimum gap between IBS


IR_PLACEMENT = IRPlacementConstants()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SIGNAL DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """
    A single signal instance on the railway network.

    Attributes:
        id:               Unique signal identifier  e.g. "sig_42_to_87_home"
        signal_type:      One of the 6 IR signal types.
        aspect:           Current MACLS aspect (changes at runtime).
        position:         (x, y) coordinate on the map.
        direction:        Tuple (source_node, target_node) — direction of travel this signal governs.
        controls_block:   Tuple (node_A, node_B) — the block section this signal protects (sorted).
        cascade_parent_id: ID of the next signal in the cascade chain (towards station), or None.
        overlap_m:        Overlap distance beyond this signal (metres).
        last_change_time: Simulation time of the last aspect change.
        change_count:     Total number of aspect changes.
    """
    id:                str
    signal_type:       SignalType
    aspect:            SignalAspect
    position:          Tuple[float, float]
    direction:         Tuple[str, str]              # (from_node, to_node)
    controls_block:    Tuple[str, str]              # sorted edge key
    cascade_parent_id: Optional[str]    = None      # next-closer signal in cascade
    overlap_m:         float            = IR_PLACEMENT.SIGNAL_OVERLAP_M
    last_change_time:  float            = 0.0
    change_count:      int              = 0

    # ── convenience properties ───────────────────────────────────────────
    @property
    def stoppage_rule(self) -> StoppageRule:
        return self.signal_type.stoppage_rule

    @property
    def is_permissive(self) -> bool:
        return self.signal_type.is_permissive

    @property
    def speed_fraction(self) -> float:
        return self.aspect.speed_fraction

    @property
    def protected_block_key(self) -> Tuple[str, str]:
        """Alias kept for backward compatibility with existing code."""
        return self.controls_block

    def set_aspect(self, new_aspect: SignalAspect, sim_time: float) -> bool:
        """
        Set a new aspect.  Returns True if the aspect actually changed.
        """
        if new_aspect == self.aspect:
            return False
        self.aspect = new_aspect
        self.last_change_time = sim_time
        self.change_count += 1
        return True


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BLOCK SECTION  (Absolute Block System)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockSection:
    """
    Represents a block section between two stations (or station limits).
    Under the Absolute Block System, only ONE train may occupy a block section
    at any time.

    The CascadeSignalController uses block occupancy to decide signal aspects.
    """
    block_key:      Tuple[str, str]            # sorted (node_A, node_B)
    is_occupied:    bool          = False
    is_disrupted:   bool          = False
    occupying_train: Optional[str] = None       # train_id currently in the block
    distance_m:     float         = 0.0         # track length in metres

    # Signals that govern entry into this block (keyed by direction)
    entry_signals:  Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    # direction (from, to) → list of signal IDs in approach order


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CASCADE SIGNAL CONTROLLER  (MACLS 4-Aspect Logic)
# ══════════════════════════════════════════════════════════════════════════════

class CascadeSignalController:
    """
    Implements Indian Railway MACLS 4-aspect cascading logic.

    Cascade rules (applied every tick):
        1. If the block protected by a signal is OCCUPIED or DISRUPTED → signal = RED.
        2. If a signal is RED → its cascade parent (the next signal further back) = YELLOW.
        3. If a signal is YELLOW → its cascade parent = DOUBLE_YELLOW.
        4. If a signal is GREEN (block clear, next signal ≥ DOUBLE_YELLOW) → GREEN.

    Interlocking rules:
        - A route can only be set if ALL blocks in the route are clear.
        - Overlap track beyond the Home signal must also be clear.
        - Once a route is set and a train is approaching, the route is LOCKED
          until the train clears the overlap.

    This controller is ticked by the simulation's SimPy process every T seconds.
    """

    def __init__(
        self,
        signals: Dict[str, Signal],
        blocks: Dict[Tuple[str, str], BlockSection],
        tick_interval: float = 1.0,
    ):
        self.signals = signals                      # signal_id → Signal
        self.blocks = blocks                        # block_key → BlockSection
        self.tick_interval = tick_interval

        # Interlocking state
        self._locked_routes: Dict[str, _LockedRoute] = {}   # route_id → _LockedRoute

        # Build reverse index: signal_id → list of child signal_ids (those whose cascade_parent_id == this)
        self._cascade_children: Dict[str, List[str]] = {}
        for sig in self.signals.values():
            if sig.cascade_parent_id:
                self._cascade_children.setdefault(sig.cascade_parent_id, []).append(sig.id)

    # ── main tick ────────────────────────────────────────────────────────
    def tick(self, sim_time: float) -> List[str]:
        """
        Run one cascade cycle.  Returns list of signal IDs whose aspect changed.
        Call this every `tick_interval` simulation seconds.
        """
        changed: List[str] = []

        # Pass 1 — set absolute signals based on block occupancy
        for sig in self.signals.values():
            block = self.blocks.get(sig.controls_block)
            if block is None:
                if sig.set_aspect(SignalAspect.OFF, sim_time):
                    changed.append(sig.id)
                continue

            if sig.signal_type in (
                SignalType.HOME,
                SignalType.STARTER,
                SignalType.ADVANCED_STARTER,
                SignalType.INTERMEDIATE_BLOCK,
            ):
                if block.is_occupied or block.is_disrupted:
                    if sig.set_aspect(SignalAspect.RED, sim_time):
                        changed.append(sig.id)
                else:
                    # Block clear — tentatively GREEN; cascade pass may downgrade
                    if sig.set_aspect(SignalAspect.GREEN, sim_time):
                        changed.append(sig.id)

        # Pass 2 — cascade: propagate restrictive aspects backwards
        #   We iterate until stable (max 4 iterations for 4 aspects).
        for _ in range(4):
            stable = True
            for sig in self.signals.values():
                if sig.cascade_parent_id is None:
                    continue
                parent = self.signals.get(sig.cascade_parent_id)
                if parent is None:
                    continue

                desired = self._cascade_aspect(sig.aspect)
                # Parent should be at least as restrictive as the cascade demands
                if desired < parent.aspect:
                    # desired is MORE restrictive
                    if parent.set_aspect(desired, sim_time):
                        changed.append(parent.id)
                        stable = False
            if stable:
                break

        # Pass 3 — interlocking: enforce locked-route constraints
        self._enforce_interlocking(sim_time, changed)

        return changed

    # ── cascade logic ────────────────────────────────────────────────────
    @staticmethod
    def _cascade_aspect(child_aspect: SignalAspect) -> SignalAspect:
        """
        Given a child signal's current aspect, what should the parent
        (one step further from the station) show?

        Child RED           → Parent YELLOW
        Child YELLOW        → Parent DOUBLE_YELLOW
        Child DOUBLE_YELLOW → Parent GREEN  (or stays GREEN)
        Child GREEN         → Parent GREEN
        Child OFF           → Parent YELLOW  (fail-safe)
        """
        return {
            SignalAspect.RED:           SignalAspect.YELLOW,
            SignalAspect.YELLOW:        SignalAspect.DOUBLE_YELLOW,
            SignalAspect.DOUBLE_YELLOW: SignalAspect.GREEN,
            SignalAspect.GREEN:         SignalAspect.GREEN,
            SignalAspect.OFF:           SignalAspect.YELLOW,
        }[child_aspect]

    # ── interlocking ─────────────────────────────────────────────────────
    def request_route(
        self,
        route_id: str,
        block_keys: List[Tuple[str, str]],
        signal_ids: List[str],
        sim_time: float,
    ) -> bool:
        """
        Request a route for a train.  Returns True if the route is granted.

        A route is granted only if:
            - All blocks in the route are clear and not disrupted.
            - Overlap blocks are clear.
            - No conflicting route is already locked.
        """
        # Check all blocks clear
        for bk in block_keys:
            block = self.blocks.get(bk)
            if block is None or block.is_occupied or block.is_disrupted:
                return False

        # Check no conflicting locked route
        for locked in self._locked_routes.values():
            if locked.conflicts_with(block_keys):
                return False

        # Grant route — lock it
        self._locked_routes[route_id] = _LockedRoute(
            route_id=route_id,
            block_keys={(bk[0], bk[1]) for bk in block_keys},
            signal_ids=list(signal_ids),
            granted_time=sim_time,
        )
        return True

    def release_route(self, route_id: str):
        """Release a locked route after the train has cleared the overlap."""
        self._locked_routes.pop(route_id, None)

    def _enforce_interlocking(self, sim_time: float, changed: List[str]):
        """Keep signals RED for blocks that are part of a conflicting locked route."""
        expired = []
        for rid, lr in self._locked_routes.items():
            # Auto-expire routes after 600s (safety timeout)
            if sim_time - lr.granted_time > 600.0:
                expired.append(rid)
                continue
        for rid in expired:
            self._locked_routes.pop(rid, None)

    # ── observation helpers (for RL agent) ───────────────────────────────
    def get_signal_feature_vector(self, signal_id: str) -> List[float]:
        """
        Returns a fixed-length numeric vector for a signal, suitable for RL observation.

        Layout (8 floats):
          [0] aspect_red, [1] aspect_yellow, [2] aspect_double_yellow, [3] aspect_green,
          [4] is_absolute, [5] block_occupied, [6] block_disrupted, [7] time_since_change
        """
        sig = self.signals.get(signal_id)
        if sig is None:
            return [0.0] * 8

        aspect_one_hot = [0.0, 0.0, 0.0, 0.0]
        idx_map = {
            SignalAspect.RED: 0, SignalAspect.YELLOW: 1,
            SignalAspect.DOUBLE_YELLOW: 2, SignalAspect.GREEN: 3,
        }
        aspect_one_hot[idx_map.get(sig.aspect, 0)] = 1.0

        block = self.blocks.get(sig.controls_block)
        block_occ = float(block.is_occupied) if block else 0.0
        block_dis = float(block.is_disrupted) if block else 0.0

        return aspect_one_hot + [
            float(not sig.is_permissive),       # is_absolute
            block_occ,
            block_dis,
            sig.last_change_time,               # caller can normalise
        ]

    def get_all_aspects_tensor(self) -> Dict[str, int]:
        """Return {signal_id: aspect_index} for all signals. Useful for RL."""
        idx_map = {
            SignalAspect.RED: 0, SignalAspect.YELLOW: 1,
            SignalAspect.DOUBLE_YELLOW: 2, SignalAspect.GREEN: 3,
            SignalAspect.OFF: 0,
        }
        return {sid: idx_map[s.aspect] for sid, s in self.signals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LOCKED ROUTE (internal dataclass)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _LockedRoute:
    route_id:     str
    block_keys:   Set[Tuple[str, str]]
    signal_ids:   List[str]
    granted_time: float

    def conflicts_with(self, candidate_blocks: List[Tuple[str, str]]) -> bool:
        """Does this locked route share any block with the candidate?"""
        return bool(self.block_keys & set(tuple(bk) for bk in candidate_blocks))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SPAD DETECTOR  (Signal Passed At Danger)
# ══════════════════════════════════════════════════════════════════════════════

class SPADDetector:
    """
    Detects when a train passes a signal at RED (Signal Passed At Danger).
    In real IR systems this triggers an automatic brake application (KAVACH/TPWS).
    """

    def __init__(self, signals: Dict[str, Signal]):
        self.signals = signals
        self.spad_events: List[Dict] = []

    def check(
        self,
        train_id: str,
        signal_id: str,
        sim_time: float,
    ) -> bool:
        """
        Call this when a train is about to pass a signal.
        Returns True if a SPAD has occurred (signal is RED and non-permissive).
        """
        sig = self.signals.get(signal_id)
        if sig is None:
            return False

        if sig.aspect == SignalAspect.RED and not sig.is_permissive:
            self.spad_events.append({
                "train_id": train_id,
                "signal_id": signal_id,
                "signal_type": sig.signal_type.value,
                "time": sim_time,
            })
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 8.  UTILITY:  Build signals / blocks from graph JSON edge data
# ══════════════════════════════════════════════════════════════════════════════

def build_signals_and_blocks_from_graph(
    edges_data: List[Dict],
    signals_data: List[Dict],
) -> Tuple[Dict[str, Signal], Dict[Tuple[str, str], BlockSection]]:
    """
    Construct Signal and BlockSection dicts from the JSON produced by
    graph/add_signals.py.

    Parameters:
        edges_data:   list of edge dicts from the graph JSON (source, target, distance, signals)
        signals_data: flat list of signal dicts (already extracted from edges)

    Returns:
        (signals_dict, blocks_dict)
    """
    signals_dict: Dict[str, Signal] = {}
    blocks_dict:  Dict[Tuple[str, str], BlockSection] = {}

    # Build block sections from edges
    for edge in edges_data:
        src = str(edge["source"])
        tgt = str(edge["target"])
        a, b = sorted((src, tgt))
        key: Tuple[str, str] = (a, b)
        dist = float(edge.get("distance", 0.0))
        if key not in blocks_dict:
            blocks_dict[key] = BlockSection(block_key=key, distance_m=dist)

    # Build Signal objects from signal dicts
    for sd in signals_data:
        sig_id = sd["id"]
        parts = sig_id.split("_")

        # Parse direction and block from ID:  sig_{from}_{to}_{to}_{type}
        #   format: sig_{source}_to_{target}_{type}
        from_node = parts[1] if len(parts) >= 5 else ""
        to_node   = parts[3] if len(parts) >= 5 else ""
        sig_type_str = parts[4] if len(parts) >= 5 else (parts[-1] if parts else "home")

        # Map type string to enum
        type_map = {
            "home":              SignalType.HOME,
            "distant":           SignalType.DISTANT,
            "inner_distant":     SignalType.INNER_DISTANT,
            "starter":           SignalType.STARTER,
            "adv_starter":       SignalType.ADVANCED_STARTER,
            "advanced_starter":  SignalType.ADVANCED_STARTER,
            "intermediate":      SignalType.INTERMEDIATE_BLOCK,
        }
        sig_type = type_map.get(sig_type_str, SignalType.HOME)

        bk_a, bk_b = sorted((from_node, to_node))
        block_key: Tuple[str, str] = (bk_a, bk_b)

        # Parse aspect — use 'initial_aspect' first (what the graph loader writes),
        # fall back to 'aspect', then to the signal type's default.
        raw_aspect = sd.get("initial_aspect", sd.get("aspect", None))
        aspect_map: Dict[str, SignalAspect] = {
            "red":           SignalAspect.RED,
            "yellow":        SignalAspect.YELLOW,
            "double_yellow": SignalAspect.DOUBLE_YELLOW,
            "green":         SignalAspect.GREEN,
            "off":           SignalAspect.OFF,
        }
        aspect = aspect_map.get(str(raw_aspect), sig_type.default_aspect)

        position = (
            float(sd.get("position", {}).get("x", 0.0)),
            float(sd.get("position", {}).get("y", 0.0)),
        )

        cascade_parent_id = sd.get("cascade_parent_id", None)

        sig = Signal(
            id=sig_id,
            signal_type=sig_type,
            aspect=aspect,
            position=position,
            direction=(from_node, to_node),
            controls_block=block_key,
            cascade_parent_id=cascade_parent_id,
        )
        signals_dict[sig_id] = sig

        # Register this signal as an entry signal for its block
        block = blocks_dict.get(block_key)
        if block is not None:
            direction: Tuple[str, str] = (from_node, to_node)
            block.entry_signals.setdefault(direction, []).append(sig_id)

    return signals_dict, blocks_dict
