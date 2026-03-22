"""
graph/add_signals.py — IR-Correct Signal Placement on the Railway Graph

Places MACLS 4-aspect colour-light signals on every edge of the graph according
to Indian Railway signaling standards.

Changes from the previous version:
  • IR-correct distances:  Distant at 1000 m from Home (was 400 m),
    Home at 400 m before station (was 100 m), Inner Distant at 1000 m from
    Distant (was 500 m).
  • Writes BOTH  "aspect" AND "initial_aspect" keys  → eliminates the reader bug
    where loaders read "initial_aspect" but the old writer only wrote "aspect".
  • Each signal dict now includes: direction, controls_block, cascade_parent_id,
    overlap_m, signal_type (matching SignalType enum values).
  • Signals are placed in BOTH directions for every edge (directed signaling).

Usage:
    python -m graph.add_signals                    # default paths
    python -m graph.add_signals --input X --output Y
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# Import canonical constants from the single source of truth
from simulation.signals import IR_PLACEMENT, SignalType, SignalAspect

# ══════════════════════════════════════════════════════════════════════════════
# Constants  (derived from IR_PLACEMENT for readability)
# ══════════════════════════════════════════════════════════════════════════════

HOME_FROM_STATION       = IR_PLACEMENT.HOME_FROM_STATION           # 400 m
DISTANT_FROM_HOME       = IR_PLACEMENT.DISTANT_FROM_HOME           # 1000 m
INNER_DISTANT_FROM_DIST = IR_PLACEMENT.INNER_DISTANT_FROM_DISTANT  # 1000 m
STARTER_FROM_STATION    = IR_PLACEMENT.STARTER_FROM_STATION        # 80 m
ADV_STARTER_FROM_START  = IR_PLACEMENT.ADV_STARTER_FROM_STARTER    # 250 m
SIGNAL_OVERLAP          = IR_PLACEMENT.SIGNAL_OVERLAP_M            # 120 m

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_graph(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"✔️  Loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges from {filepath}")
    return data


def _save_graph(data: Dict[str, Any], filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅  Saved graph with signals → {filepath}")


def _point_on_line(
    start: Dict[str, float],
    end: Dict[str, float],
    distance_from_start: float,
) -> Tuple[float, float]:
    """Interpolate a point at `distance_from_start` along the start→end segment."""
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return start["x"], start["y"]
    ratio = distance_from_start / length
    return start["x"] + dx * ratio, start["y"] + dy * ratio


def _make_signal_dict(
    sig_id: str,
    sig_type: SignalType,
    aspect: SignalAspect,
    x: float,
    y: float,
    source_id: str,
    target_id: str,
    cascade_parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a signal dict compatible with the new Signal dataclass loader."""
    block_key = sorted([source_id, target_id])
    return {
        "id": sig_id,
        "type": sig_type.value,
        "signal_type": sig_type.value,
        # Write BOTH keys so any loader works
        "aspect": aspect.value,
        "initial_aspect": aspect.value,
        "position": {"x": x, "y": y},
        "direction": [source_id, target_id],
        "controls_block": block_key,
        "cascade_parent_id": cascade_parent_id,
        "overlap_m": SIGNAL_OVERLAP,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Core placement logic
# ══════════════════════════════════════════════════════════════════════════════

def place_signals_on_edge(
    source_id: str,
    target_id: str,
    source_pos: Dict[str, float],
    target_pos: Dict[str, float],
    track_length: float,
) -> List[Dict[str, Any]]:
    """
    Place signals on a directed edge  source → target.

    Incoming signals (approaching target station, measured BACK from target):
        Home          – HOME_FROM_STATION before target
        Distant       – DISTANT_FROM_HOME before Home
        Inner Distant – INNER_DISTANT_FROM_DIST before Distant

    Outgoing signals (leaving source station, measured FORWARD from source):
        Starter       – STARTER_FROM_STATION after source
        Adv Starter   – ADV_STARTER_FROM_START after Starter

    Signals that don't fit (track too short) are silently skipped.
    """
    signals: List[Dict[str, Any]] = []

    # ── Incoming signals (placed relative to TARGET, facing source) ──────

    # 1. Home
    home_id: Optional[str] = None
    if track_length > HOME_FROM_STATION:
        hx, hy = _point_on_line(target_pos, source_pos, HOME_FROM_STATION)
        home_id = f"sig_{source_id}_to_{target_id}_home"
        signals.append(_make_signal_dict(
            sig_id=home_id,
            sig_type=SignalType.HOME,
            aspect=SignalType.HOME.default_aspect,
            x=hx, y=hy,
            source_id=source_id,
            target_id=target_id,
        ))

    # 2. Distant
    distant_id: Optional[str] = None
    required_distant = HOME_FROM_STATION + DISTANT_FROM_HOME
    if track_length > required_distant:
        dx, dy = _point_on_line(target_pos, source_pos, required_distant)
        distant_id = f"sig_{source_id}_to_{target_id}_distant"
        signals.append(_make_signal_dict(
            sig_id=distant_id,
            sig_type=SignalType.DISTANT,
            aspect=SignalType.DISTANT.default_aspect,
            x=dx, y=dy,
            source_id=source_id,
            target_id=target_id,
            cascade_parent_id=None,
        ))
        # Link: Home's cascade_parent is Distant
        if home_id:
            for s in signals:
                if s["id"] == home_id:
                    s["cascade_parent_id"] = distant_id

    # 3. Inner Distant
    required_inner = required_distant + INNER_DISTANT_FROM_DIST
    if track_length > required_inner:
        idx, idy = _point_on_line(target_pos, source_pos, required_inner)
        inner_id = f"sig_{source_id}_to_{target_id}_inner_distant"
        signals.append(_make_signal_dict(
            sig_id=inner_id,
            sig_type=SignalType.INNER_DISTANT,
            aspect=SignalType.INNER_DISTANT.default_aspect,
            x=idx, y=idy,
            source_id=source_id,
            target_id=target_id,
        ))
        # Link: Distant's cascade_parent is Inner Distant
        if distant_id:
            for s in signals:
                if s["id"] == distant_id:
                    s["cascade_parent_id"] = inner_id

    # ── Outgoing signals (placed relative to SOURCE, facing target) ──────

    # 4. Starter
    starter_id: Optional[str] = None
    if track_length > STARTER_FROM_STATION:
        sx, sy = _point_on_line(source_pos, target_pos, STARTER_FROM_STATION)
        starter_id = f"sig_{source_id}_to_{target_id}_starter"
        signals.append(_make_signal_dict(
            sig_id=starter_id,
            sig_type=SignalType.STARTER,
            aspect=SignalType.STARTER.default_aspect,
            x=sx, y=sy,
            source_id=source_id,
            target_id=target_id,
        ))

    # 5. Advanced Starter
    required_adv = STARTER_FROM_STATION + ADV_STARTER_FROM_START
    if track_length > required_adv:
        ax, ay = _point_on_line(source_pos, target_pos, required_adv)
        adv_id = f"sig_{source_id}_to_{target_id}_adv_starter"
        signals.append(_make_signal_dict(
            sig_id=adv_id,
            sig_type=SignalType.ADVANCED_STARTER,
            aspect=SignalType.ADVANCED_STARTER.default_aspect,
            x=ax, y=ay,
            source_id=source_id,
            target_id=target_id,
        ))

    return signals


def add_signals_to_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process all edges and add directed signals in BOTH directions.

    For each undirected edge (A, B), signals are placed for:
        • Direction A → B  (signals governing travel from A towards B)
        • Direction B → A  (signals governing travel from B towards A)
    """
    nodes_map = {str(n["id"]): n for n in data["nodes"]}
    total_signals = 0

    for edge in data["edges"]:
        source_id = str(edge["source"])
        target_id = str(edge["target"])

        if source_id not in nodes_map or target_id not in nodes_map:
            continue

        src = nodes_map[source_id]
        tgt = nodes_map[target_id]
        source_pos = {"x": src["pos_x"], "y": src["pos_y"]}
        target_pos = {"x": tgt["pos_x"], "y": tgt["pos_y"]}

        track_length = math.hypot(
            target_pos["x"] - source_pos["x"],
            target_pos["y"] - source_pos["y"],
        )
        edge["distance"] = track_length

        # Place signals in BOTH directions
        sigs_fwd = place_signals_on_edge(source_id, target_id, source_pos, target_pos, track_length)
        sigs_rev = place_signals_on_edge(target_id, source_id, target_pos, source_pos, track_length)

        edge["signals"] = sigs_fwd + sigs_rev
        total_signals += len(edge["signals"])

    print(f"✨  Generated {total_signals} signals across {len(data['edges'])} edges.")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Add IR-correct MACLS signals to the railway graph.")
    default_input  = os.path.join(_PROJECT_ROOT, "data", "processed", "godot_station_graph.json")
    default_output = os.path.join(_PROJECT_ROOT, "data", "processed", "godot_station_graph_with_signals.json")

    parser.add_argument("--input",  default=default_input,  help="Input graph JSON file.")
    parser.add_argument("--output", default=default_output, help="Output graph JSON file with signals.")
    args = parser.parse_args()

    data = _load_graph(args.input)
    data = add_signals_to_graph(data)
    _save_graph(data, args.output)


if __name__ == "__main__":
    main()
