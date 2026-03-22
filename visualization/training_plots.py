#!/usr/bin/env python3
"""
visualization/training_plots.py — Training Dashboard for Railway RL

Generates a 7-panel visualization that answers:
  "Is the LGNN actually learning? Is the RL working?"

Panels:
  1. Episode Reward curve (raw + rolling avg) — is reward increasing?
  2. Policy Loss + Value Loss  — are losses converging?
  3. Entropy over updates  — is exploration decaying naturally?
  4. KL Divergence + Clip Fraction — is PPO stable?
  5. Gradient Norms: LGNN vs Encoder vs Heads — proof LGNN is being trained
  6. Episode Length over time — are trains finishing faster?
  7. Learning Rate schedule — confirms annealing

Usage:
    python -m visualization.training_plots                         # uses default path
    python -m visualization.training_plots --metrics PATH.json     # custom path
    python -m visualization.training_plots --metrics PATH --out dashboard.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Utility ──────────────────────────────────────────────────────────────────

def rolling_avg(data: List[float], window: int = 10) -> np.ndarray:
    """Compute a centered rolling average with edge handling."""
    if len(data) < 2:
        return np.array(data)
    arr = np.array(data, dtype=np.float64)
    kernel = np.ones(min(window, len(arr))) / min(window, len(arr))
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad to same length
    pad_left = (len(arr) - len(smoothed)) // 2
    pad_right = len(arr) - len(smoothed) - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode="edge")


# ── Main Dashboard Generator ────────────────────────────────────────────────

def generate_training_dashboard(metrics_path: str, output_path: str):
    """
    Read training_metrics.json and produce a comprehensive dashboard PNG.
    """
    with open(metrics_path, "r") as f:
        data = json.load(f)

    episodes = data["episodes"]
    updates = data["updates"]
    step_rewards = data.get("step_rewards", [])

    n_ep = len(episodes["rewards"])
    n_up = len(updates["policy_loss"])

    if n_ep < 2 and n_up < 1:
        print("  ⚠ Not enough data to plot (need ≥2 episodes)")
        return

    # ── Figure setup ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24), facecolor="#0d1117")
    fig.suptitle(
        "Railway RL Training Dashboard — PPO + LGNN + RK4",
        fontsize=22, fontweight="bold", color="white", y=0.98,
    )

    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.28,
                            left=0.07, right=0.96, top=0.94, bottom=0.04)

    # Common style
    DARK_BG = "#161b22"
    GRID_COLOR = "#30363d"
    TEXT_COLOR = "#c9d1d9"
    ACCENT = "#58a6ff"
    GREEN = "#3fb950"
    RED = "#f85149"
    ORANGE = "#d29922"
    PURPLE = "#bc8cff"
    CYAN = "#39d353"

    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=11)
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    ep_x = list(range(1, n_ep + 1))
    up_x = list(range(1, n_up + 1))

    # ── Panel 1: Episode Reward ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "① Episode Reward (is the agent improving?)", "Episode", "Reward")
    rewards = episodes["rewards"]
    ax1.plot(ep_x, rewards, alpha=0.3, color=ACCENT, linewidth=1, label="Raw")
    if n_ep >= 3:
        window = max(3, n_ep // 8)
        smooth = rolling_avg(rewards, window)
        ax1.plot(ep_x, smooth, color=ACCENT, linewidth=2.5,
                 label=f"Rolling avg (w={window})")
    # Best marker
    if rewards:
        best_idx = int(np.argmax(rewards))
        ax1.scatter([best_idx + 1], [rewards[best_idx]], color=GREEN, s=120,
                    zorder=5, marker="*", label=f"Best: {rewards[best_idx]:.2f}")
    ax1.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    # ── Panel 2: Policy Loss + Value Loss ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "② Loss Curves (are they converging?)", "PPO Update", "Loss")
    ax2.plot(up_x, updates["policy_loss"], color=RED, linewidth=2, label="Policy Loss", alpha=0.9)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(up_x, updates["value_loss"], color=ORANGE, linewidth=2, label="Value Loss", alpha=0.9)
    ax2_twin.tick_params(colors=ORANGE, labelsize=9)
    ax2_twin.set_ylabel("Value Loss", color=ORANGE, fontsize=11)
    ax2.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    ax2_twin.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, fontsize=9)
    for spine in ax2_twin.spines.values():
        spine.set_color(GRID_COLOR)

    # ── Panel 3: Entropy ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "③ Entropy (should decrease gradually — less random over time)",
             "PPO Update", "Entropy")
    ax3.plot(up_x, updates["entropy"], color=PURPLE, linewidth=2)
    ax3.fill_between(up_x, updates["entropy"], alpha=0.15, color=PURPLE)

    # ── Panel 4: KL Divergence + Clip Fraction ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "④ KL Divergence & Clip Fraction (PPO stability)",
             "PPO Update", "KL / Clip")
    ax4.plot(up_x, updates["approx_kl"], color=CYAN, linewidth=2, label="Approx KL")
    if "clip_fraction" in updates:
        ax4.plot(up_x, updates["clip_fraction"], color=ORANGE, linewidth=2,
                 label="Clip Fraction", alpha=0.8)
    if updates.get("approx_kl"):
        # Draw target KL line
        ax4.axhline(y=0.02, color=RED, linestyle="--", alpha=0.5, label="Target KL (0.02)")
    ax4.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    # ── Panel 5: Gradient Norms (PROOF the LGNN is being trained) ───────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, "⑤ Gradient Norms (proof LGNN is actively trained by PPO)",
             "PPO Update", "‖∇‖₂")
    ax5.plot(up_x, updates["grad_norm_lgnn"], color=ACCENT, linewidth=2.5,
             label="LGNN (backbone)")
    ax5.plot(up_x, updates["grad_norm_encoder"], color=GREEN, linewidth=2,
             label="Encoder", alpha=0.8)
    ax5.plot(up_x, updates["grad_norm_heads"], color=ORANGE, linewidth=2,
             label="Actor+Critic heads", alpha=0.8)
    ax5.plot(up_x, updates["grad_norm_total"], color=TEXT_COLOR, linewidth=1.5,
             label="Total", alpha=0.5, linestyle="--")
    ax5.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    # Add annotation explaining what this means
    if n_up >= 1:
        mean_lgnn = np.mean(updates["grad_norm_lgnn"])
        mean_heads = np.mean(updates["grad_norm_heads"])
        ratio = mean_lgnn / max(mean_heads, 1e-8)
        ax5.text(
            0.02, 0.95,
            f"Avg ‖∇LGNN‖={mean_lgnn:.3f}  Avg ‖∇heads‖={mean_heads:.3f}\n"
            f"Ratio: {ratio:.1f}× → LGNN receives {'strong' if ratio > 0.3 else 'weak'} gradients",
            transform=ax5.transAxes, fontsize=9, color=TEXT_COLOR,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=GRID_COLOR),
        )

    # ── Panel 6: Episode Length ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, "⑥ Episode Length (should stabilize or decrease = trains finish faster)",
             "Episode", "Steps")
    ax6.plot(ep_x, episodes["lengths"], color=ACCENT, alpha=0.4, linewidth=1)
    if n_ep >= 3:
        smooth_len = rolling_avg(episodes["lengths"], max(3, n_ep // 8))
        ax6.plot(ep_x, smooth_len, color=ACCENT, linewidth=2.5)

    # ── Panel 7: Learning Rate + PPO Epochs ──────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    style_ax(ax7, "⑦ Learning Rate Schedule + PPO Epochs Used",
             "PPO Update", "Learning Rate")
    ax7.plot(up_x, updates["learning_rate"], color=GREEN, linewidth=2, label="LR")
    ax7_twin = ax7.twinx()
    ax7_twin.bar(up_x, updates.get("epochs_used", [0]*n_up),
                 color=PURPLE, alpha=0.3, label="PPO Epochs")
    ax7_twin.set_ylabel("PPO Epochs", color=PURPLE, fontsize=11)
    ax7_twin.tick_params(colors=PURPLE, labelsize=9)
    for spine in ax7_twin.spines.values():
        spine.set_color(GRID_COLOR)
    ax7.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    ax7_twin.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, fontsize=9)

    # ── Panel 8: Step-level reward (smoothed) ────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    style_ax(ax8, "⑧ Per-Step Reward (smoothed — overall learning signal)",
             "Global Step", "Reward")
    if len(step_rewards) > 10:
        step_x = list(range(1, len(step_rewards) + 1))
        window = max(10, len(step_rewards) // 30)
        smooth_sr = rolling_avg(step_rewards, window)
        ax8.plot(step_x, smooth_sr, color=GREEN, linewidth=2,
                 label=f"Smoothed (w={window})")
        ax8.plot(step_x, step_rewards, alpha=0.1, color=GREEN, linewidth=0.5)
        ax8.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    else:
        ax8.text(0.5, 0.5, "Not enough step data yet",
                 transform=ax8.transAxes, ha="center", va="center",
                 color=TEXT_COLOR, fontsize=12)

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Dashboard saved → {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RL training dashboard")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--metrics",
        type=str,
        default=os.path.join(_root, "dataset", "training_metrics.json"),
        help="Path to training_metrics.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_root, "dataset", "training_dashboard.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        print(f"Error: {args.metrics} not found. Run training first.")
        sys.exit(1)

    generate_training_dashboard(args.metrics, args.out)
