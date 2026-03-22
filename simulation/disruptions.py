"""
simulation/disruptions.py — Seeded Natural Disruption System

Provides 7 realistic disruption types that can affect block sections and signals.
The DisruptionManager schedules disruptions via Poisson processes within a SimPy
environment, and applies / resolves their effects on the network.

Disruption Types:
    1. TRACK_FAULT          – Rail fracture, buckle, or subsidence
    2. SIGNAL_FAILURE       – Lamp / relay / cable fault on a signal
    3. OVERHEAD_WIRE_BREAK  – OHE failure (electrified sections)
    4. WEATHER              – Fog, flooding, cyclone → speed restriction
    5. ANIMAL_INTRUSION     – Cattle / elephant on track
    6. LANDSLIDE            – Hill-section earth / rock fall
    7. EQUIPMENT_FAILURE    – Point machine, level crossing gate, etc.

Each disruption has:
    - Severity (MINOR, MODERATE, SEVERE)
    - Duration distribution (exponential with type-specific mean)
    - Effect on blocks (capacity reduction, full closure, speed restriction)
    - Effect on signals (force RED, force OFF, or leave alone)

Usage:
    from simulation.disruptions import DisruptionManager, DisruptionConfig

    mgr = DisruptionManager(config=DisruptionConfig(), blocks=blocks, signals=signals)
    env.process(mgr.run(env))   # starts the Poisson scheduler
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from simulation.signals import (
    BlockSection,
    Signal,
    SignalAspect,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class DisruptionType(Enum):
    TRACK_FAULT         = "track_fault"
    SIGNAL_FAILURE      = "signal_failure"
    OVERHEAD_WIRE_BREAK = "overhead_wire_break"
    WEATHER             = "weather"
    ANIMAL_INTRUSION    = "animal_intrusion"
    LANDSLIDE           = "landslide"
    EQUIPMENT_FAILURE   = "equipment_failure"


class Severity(Enum):
    MINOR    = auto()   # speed restriction only
    MODERATE = auto()   # partial block, reduced capacity
    SEVERE   = auto()   # full block closure


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DISRUPTION PROFILES  (type → default parameters)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _DisruptionProfile:
    """Internal: default parameters for each disruption type."""
    mean_duration_s: float          # mean duration (exponential distribution)
    severity_weights: Tuple[float, float, float]  # (MINOR, MODERATE, SEVERE) probabilities
    speed_fraction: float           # when active, max speed as fraction of line-speed  (MINOR)
    affects_signal: bool            # does it also knock out the protecting signal?
    probability_weight: float       # relative frequency compared to others


_PROFILES: Dict[DisruptionType, _DisruptionProfile] = {
    DisruptionType.TRACK_FAULT: _DisruptionProfile(
        mean_duration_s=300.0, severity_weights=(0.3, 0.5, 0.2),
        speed_fraction=0.0, affects_signal=False, probability_weight=1.0,
    ),
    DisruptionType.SIGNAL_FAILURE: _DisruptionProfile(
        mean_duration_s=180.0, severity_weights=(0.4, 0.4, 0.2),
        speed_fraction=0.25, affects_signal=True, probability_weight=1.2,
    ),
    DisruptionType.OVERHEAD_WIRE_BREAK: _DisruptionProfile(
        mean_duration_s=600.0, severity_weights=(0.1, 0.3, 0.6),
        speed_fraction=0.0, affects_signal=False, probability_weight=0.4,
    ),
    DisruptionType.WEATHER: _DisruptionProfile(
        mean_duration_s=900.0, severity_weights=(0.5, 0.3, 0.2),
        speed_fraction=0.30, affects_signal=False, probability_weight=1.5,
    ),
    DisruptionType.ANIMAL_INTRUSION: _DisruptionProfile(
        mean_duration_s=120.0, severity_weights=(0.6, 0.3, 0.1),
        speed_fraction=0.10, affects_signal=False, probability_weight=0.8,
    ),
    DisruptionType.LANDSLIDE: _DisruptionProfile(
        mean_duration_s=1800.0, severity_weights=(0.05, 0.25, 0.7),
        speed_fraction=0.0, affects_signal=True, probability_weight=0.3,
    ),
    DisruptionType.EQUIPMENT_FAILURE: _DisruptionProfile(
        mean_duration_s=240.0, severity_weights=(0.3, 0.5, 0.2),
        speed_fraction=0.15, affects_signal=False, probability_weight=1.0,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ACTIVE DISRUPTION RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActiveDisruption:
    """Represents a disruption currently affecting the network."""
    disruption_id: str
    disruption_type: DisruptionType
    severity: Severity
    block_key: Tuple[str, str]
    affected_signals: List[str]
    start_time: float
    duration: float
    speed_restriction: float       # 0.0 = full closure, >0 = fractional speed

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DISRUPTION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisruptionConfig:
    """
    Configuration for the DisruptionManager.

    Attributes:
        enabled:              Master switch for disruptions.
        base_rate:            Base Poisson rate (disruptions per sim-second).
                              e.g. 0.001 → ~1 disruption every 1000 s.
        seed:                 Random seed for reproducibility.
        type_weights:         Override probability weights for each type.
        max_concurrent:       Maximum disruptions active at the same time.
        cooldown_per_block_s: Minimum seconds between disruptions on the same block.
    """
    enabled:              bool  = True
    base_rate:            float = 0.001
    seed:                 Optional[int] = None
    type_weights:         Optional[Dict[DisruptionType, float]] = None
    max_concurrent:       int   = 5
    cooldown_per_block_s: float = 300.0


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DISRUPTION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class DisruptionManager:
    """
    Manages the lifecycle of disruptions within the SimPy simulation.

    Usage:
        mgr = DisruptionManager(config, blocks, signals)
        env.process(mgr.run(env))            # Poisson scheduler
        env.process(mgr.auto_resolve(env))   # periodic resolution check
    """

    def __init__(
        self,
        config: DisruptionConfig,
        blocks: Dict[Tuple[str, str], BlockSection],
        signals: Dict[str, Signal],
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.config = config
        self.blocks = blocks
        self.signals = signals
        self.event_callback = event_callback  # called on start/end of disruption

        self._rng = random.Random(config.seed)
        self._active: Dict[str, ActiveDisruption] = {}   # disruption_id → ActiveDisruption
        self._block_last_disrupted: Dict[Tuple[str, str], float] = {}  # cooldown tracking
        self._counter = 0

        # Build weighted type distribution
        self._type_list: List[DisruptionType] = []
        self._type_weights: List[float] = []
        for dt in DisruptionType:
            profile = _PROFILES[dt]
            w = (config.type_weights or {}).get(dt, profile.probability_weight)
            self._type_list.append(dt)
            self._type_weights.append(w)

    # ── SimPy processes ──────────────────────────────────────────────────

    def run(self, env) -> Any:
        """
        SimPy process: schedule disruptions via a Poisson process.
        Yields SimPy timeout events.
        """
        if not self.config.enabled:
            return

        while True:
            # Inter-arrival time (exponential)
            lam = max(1e-9, self.config.base_rate)
            inter_arrival = self._rng.expovariate(lam)
            yield env.timeout(inter_arrival)

            if len(self._active) >= self.config.max_concurrent:
                continue

            self._trigger_disruption(env.now)

    def auto_resolve(self, env, check_interval: float = 10.0) -> Any:
        """
        SimPy process: periodically check and resolve expired disruptions.
        """
        while True:
            yield env.timeout(check_interval)
            self._resolve_expired(env.now)

    # ── internal logic ───────────────────────────────────────────────────

    def _trigger_disruption(self, sim_time: float):
        """Pick a random block, type, severity and create a disruption."""
        block_keys = list(self.blocks.keys())
        if not block_keys:
            return

        # Pick a block (avoid recently disrupted ones)
        eligible = [
            bk for bk in block_keys
            if sim_time - self._block_last_disrupted.get(bk, -1e9) >= self.config.cooldown_per_block_s
               and not self.blocks[bk].is_disrupted
        ]
        if not eligible:
            return

        block_key = self._rng.choice(eligible)
        block = self.blocks[block_key]

        # Pick disruption type (weighted)
        dtype = self._rng.choices(self._type_list, weights=self._type_weights, k=1)[0]
        profile = _PROFILES[dtype]

        # Pick severity
        sev = self._rng.choices(
            [Severity.MINOR, Severity.MODERATE, Severity.SEVERE],
            weights=profile.severity_weights, k=1
        )[0]

        # Duration (exponential)
        mean_dur = profile.mean_duration_s
        if sev == Severity.SEVERE:
            mean_dur *= 2.0
        elif sev == Severity.MINOR:
            mean_dur *= 0.5
        duration = max(10.0, self._rng.expovariate(1.0 / mean_dur))

        # Speed restriction based on severity
        if sev == Severity.SEVERE:
            speed_restriction = 0.0
        elif sev == Severity.MODERATE:
            speed_restriction = profile.speed_fraction * 0.5
        else:
            speed_restriction = profile.speed_fraction

        # Identify affected signals
        affected_signals: List[str] = []
        if profile.affects_signal:
            for direction_sigs in block.entry_signals.values():
                affected_signals.extend(direction_sigs)

        # Create disruption record
        self._counter += 1
        d_id = f"disruption_{self._counter}_{dtype.value}"
        ad = ActiveDisruption(
            disruption_id=d_id,
            disruption_type=dtype,
            severity=sev,
            block_key=block_key,
            affected_signals=affected_signals,
            start_time=sim_time,
            duration=duration,
            speed_restriction=speed_restriction,
        )
        self._active[d_id] = ad
        self._block_last_disrupted[block_key] = sim_time

        # Apply effects
        self._apply_effects(ad, sim_time)

        # Callback
        if self.event_callback:
            self.event_callback("disruption_start", {
                "disruption_id": d_id,
                "type": dtype.value,
                "severity": sev.name,
                "block": block_key,
                "duration": duration,
                "speed_restriction": speed_restriction,
            })

    def _apply_effects(self, ad: ActiveDisruption, sim_time: float):
        """Apply the disruption's effects to blocks and signals."""
        block = self.blocks.get(ad.block_key)
        if block:
            block.is_disrupted = True

        # Force affected signals to RED or OFF
        for sig_id in ad.affected_signals:
            sig = self.signals.get(sig_id)
            if sig:
                if ad.severity == Severity.SEVERE:
                    sig.set_aspect(SignalAspect.OFF, sim_time)
                else:
                    sig.set_aspect(SignalAspect.RED, sim_time)

    def _resolve_expired(self, sim_time: float):
        """Resolve all disruptions whose duration has elapsed."""
        to_remove = [
            d_id for d_id, ad in self._active.items()
            if sim_time >= ad.end_time
        ]
        for d_id in to_remove:
            ad = self._active.pop(d_id)
            self._clear_effects(ad, sim_time)

            if self.event_callback:
                self.event_callback("disruption_end", {
                    "disruption_id": d_id,
                    "type": ad.disruption_type.value,
                    "block": ad.block_key,
                })

    def _clear_effects(self, ad: ActiveDisruption, sim_time: float):
        """Remove the disruption's effects from blocks and signals."""
        block = self.blocks.get(ad.block_key)
        if block:
            # Only clear disrupted flag if no OTHER active disruption affects this block
            other_active = any(
                other.block_key == ad.block_key
                for other in self._active.values()
            )
            if not other_active:
                block.is_disrupted = False

        # Signal aspects will be corrected by the CascadeSignalController on its next tick

    # ── query helpers ────────────────────────────────────────────────────

    @property
    def active_disruptions(self) -> List[ActiveDisruption]:
        return list(self._active.values())

    def get_speed_restriction(self, block_key: Tuple[str, str]) -> float:
        """
        Returns the effective speed fraction for a block.
        1.0 = no restriction.  0.0 = full closure.
        If multiple disruptions affect a block, the most restrictive wins.
        """
        restrictions = [
            ad.speed_restriction
            for ad in self._active.values()
            if ad.block_key == block_key
        ]
        if not restrictions:
            return 1.0
        return min(restrictions)

    def is_block_closed(self, block_key: Tuple[str, str]) -> bool:
        """Is the block fully closed (zero speed restriction)?"""
        return self.get_speed_restriction(block_key) <= 0.0

    def get_disruption_feature_vector(self, block_key: Tuple[str, str]) -> List[float]:
        """
        Returns a feature vector for a block's disruption state (for RL).

        Layout (4 floats):
          [0] is_disrupted,  [1] speed_restriction,
          [2] severity_encoded (0=none, 1=minor, 2=mod, 3=severe),
          [3] remaining_fraction (0..1)
        """
        active = [ad for ad in self._active.values() if ad.block_key == block_key]
        if not active:
            return [0.0, 1.0, 0.0, 0.0]

        worst = min(active, key=lambda a: a.speed_restriction)
        sev_map = {Severity.MINOR: 1.0, Severity.MODERATE: 2.0, Severity.SEVERE: 3.0}
        sev_encoded = sev_map.get(worst.severity, 0.0) / 3.0  # normalised

        # remaining fraction
        elapsed = max(0.0, self._get_sim_time_approx() - worst.start_time)
        remaining = max(0.0, 1.0 - elapsed / worst.duration) if worst.duration > 0 else 0.0

        return [1.0, worst.speed_restriction, sev_encoded, remaining]

    def _get_sim_time_approx(self) -> float:
        """Approximate current sim time from the latest disruption start."""
        if self._active:
            return max(ad.start_time for ad in self._active.values())
        return 0.0
