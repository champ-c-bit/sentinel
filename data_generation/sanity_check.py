"""
sanity_check.py — visual + statistical validation of the generated dataset.

What it does
------------
1.  For each of the 7 fault types, randomly samples 3 episodes and plots the
    affected channel over time with a vertical marker at fault onset.
    All 21 subplots are arranged in a 7×3 grid and saved to
    data/sanity_check.png.

2.  Prints a final summary table:
      - Total episode count per category (nominal + 7 fault types)
      - How many nominal episodes ended with a safe landing

3.  run_unknown_sanity_check() (separate function) samples a few unknown-anomaly
    episodes and saves a single-row plot to data/unknown_sanity_check.png.
    Running this script directly produces both plots.

Usage
-----
  cd sentinel
  python data_generation/sanity_check.py
  # or called programmatically via run_all.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from data_generation.config import (
    NOMINAL_DIR, FAULTED_DIR, MULTI_FAULTED_DIR, LABELS_CSV, DATA_ROOT,
    UNKNOWN_DIR, UNKNOWN_LABELS_CSV,
)
from data_generation.faults import FAULT_TYPES
from data_generation.utils import safe_landing



def _list_episodes(fault_type: str) -> list[str]:
    folder = os.path.join(FAULTED_DIR, fault_type)
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder) if f.endswith(".csv")
    ])


def _affected_channel(labels: pd.DataFrame, filename: str) -> str | None:
    """
    Return the affected channel for this episode from labels.csv fault_params.

    Return values:
      str  — the sensor channel name (e.g. "y_vel") for sensor-side faults.
      None — explicit skip sentinel for multi-fault rows (>1 fault in list).
      None — also returned for actuator faults (thrust_degradation,
             actuator_stuck) whose params dict has no "channel" key; the
             caller's  if channel not in df.columns  guard would silently fall
             through to "y_pos" if we returned that default, making those
             sanity-check subplots show altitude instead of the action column.
             Returning None lets the plot loop redirect to "action" explicitly.
    """
    row = labels[labels["filename"].str.endswith(os.path.basename(filename))]
    if row.empty or "fault_params" not in row.columns:
        return None   # row not found — let caller decide fallback
    try:
        params = json.loads(row.iloc[0]["fault_params"])
    except (ValueError, TypeError):
        return None
    if len(params) > 1:
        return None   # multi-fault row — skip sentinel
    # None when "channel" key absent (actuator faults); str otherwise
    ch = params[0].get("channel")
    return str(ch) if ch is not None else None


def _onset_step(labels: pd.DataFrame, filename: str) -> int:
    """
    Return the fault onset step for this episode from labels.csv fault_params.

    Returns -1 for multi-fault rows — callers must treat -1 as a skip sentinel.
    """
    row = labels[labels["filename"].str.endswith(os.path.basename(filename))]
    if row.empty or "fault_params" not in row.columns:
        return 0
    try:
        params = json.loads(row.iloc[0]["fault_params"])
    except (ValueError, TypeError):
        return 0
    if len(params) > 1:
        return -1   # multi-fault row — explicit skip sentinel
    return int(params[0]["onset"])


# ── main ──────────────────────────────────────────────────────────────────────

def run_sanity_check(n_examples: int = 3) -> None:
    """
    Generate the 7×3 sanity-check grid and print the episode summary.

    Parameters
    ----------
    n_examples : number of example episodes to plot per fault type (default 3).
    """
    if not os.path.exists(LABELS_CSV):
        print(f"[sanity_check] labels.csv not found at {LABELS_CSV}. "
              "Run generate_dataset.py first.")
        return

    labels = pd.read_csv(LABELS_CSV)

    n_cols = n_examples
    n_rows = len(FAULT_TYPES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        "Sanity check — affected channel per fault type\n"
        "(vertical dashed line = fault onset)",
        fontsize=13, y=1.01,
    )

    rng = np.random.default_rng(0)

    for row_idx, ft in enumerate(FAULT_TYPES):
        episodes = _list_episodes(ft)
        if not episodes:
            for col_idx in range(n_cols):
                axes[row_idx][col_idx].set_visible(False)
            continue

        sampled = rng.choice(episodes, size=min(n_examples, len(episodes)),
                             replace=False)

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]

            if col_idx >= len(sampled):
                ax.set_visible(False)
                continue

            fpath   = sampled[col_idx]
            df      = pd.read_csv(fpath)
            channel = _affected_channel(labels, fpath)
            onset   = _onset_step(labels, fpath)

            # onset < 0 means multi-fault row — no single onset to mark.
            if onset < 0:
                ax.set_visible(False)
                continue

            # channel is None for two distinct reasons:
            #   • multi-fault / parse failure → already handled above via onset
            #   • actuator faults (no "channel" key in params) → redirect to "action"
            # In either remaining case, fall through to the column-presence check.
            if channel is None or channel not in df.columns:
                channel = "action" if "action" in df.columns else df.columns[1]

            ax.plot(df["step"], df[channel], linewidth=0.8, color="#3b82d4")
            ax.axvline(onset, color="#e74c3c", linestyle="--", linewidth=1.2,
                       label=f"onset={onset}")

            fname = os.path.basename(fpath)
            ax.set_title(f"{ft}\n{fname}", fontsize=7, pad=3)
            ax.set_xlabel("step", fontsize=7)
            ax.set_ylabel(channel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(DATA_ROOT, "sanity_check.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSanity-check grid saved → {out_path}")

    # ── printed summary ───────────────────────────────────────────────────
    nominal_files = sorted([
        f for f in os.listdir(NOMINAL_DIR) if f.endswith(".csv")
    ]) if os.path.isdir(NOMINAL_DIR) else []

    safe_count = sum(
        safe_landing(pd.read_csv(os.path.join(NOMINAL_DIR, f)))
        for f in nominal_files
    )

    print("\n" + "=" * 52)
    print("EPISODE COUNTS")
    print("=" * 52)
    print(f"  {'nominal':<34}: {len(nominal_files)}")
    print(f"    └─ safe landings        : {safe_count} / {len(nominal_files)}")
    total = len(nominal_files)
    for ft in FAULT_TYPES:
        eps = _list_episodes(ft)
        print(f"  {ft:<34}: {len(eps)}")
        total += len(eps)
    if os.path.isdir(MULTI_FAULTED_DIR):
        multi_total = sum(
            len([f for f in os.listdir(os.path.join(MULTI_FAULTED_DIR, combo))
                 if f.endswith(".csv")])
            for combo in os.listdir(MULTI_FAULTED_DIR)
            if os.path.isdir(os.path.join(MULTI_FAULTED_DIR, combo))
        )
        print(f"  {'multi-fault (all combos)':<34}: {multi_total}")
        total += multi_total
    print(f"  {'TOTAL':<34}: {total}")
    print("=" * 52)


# ── Unknown-anomaly spot-check ────────────────────────────────────────────────

def run_unknown_sanity_check(n_examples: int = 3) -> None:
    """
    Plot a few unknown-anomaly episode trajectories for visual inspection.

    Produces data/unknown_sanity_check.png — a single row of n_examples
    subplots.  Each subplot shows:
      - y_pos (altitude) in blue on the left axis
      - the corrupted channel in orange on a twin right axis
      - a vertical dashed line at fault onset

    If data/unknown/ doesn't exist or is empty, prints a message and returns.
    """
    if not os.path.isdir(UNKNOWN_DIR):
        print("[unknown_sanity_check] data/unknown/ not found — "
              "run generate_unknown_episodes() first.")
        return

    csv_files = sorted([
        os.path.join(UNKNOWN_DIR, f)
        for f in os.listdir(UNKNOWN_DIR) if f.endswith(".csv")
    ])
    if not csv_files:
        print("[unknown_sanity_check] data/unknown/ is empty.")
        return

    # Load labels for onset / channel lookup (may not exist on first call)
    if os.path.exists(UNKNOWN_LABELS_CSV):
        ulabels = pd.read_csv(UNKNOWN_LABELS_CSV)
    else:
        ulabels = pd.DataFrame()

    rng = np.random.default_rng(1)
    sampled = rng.choice(csv_files, size=min(n_examples, len(csv_files)), replace=False)

    fig, axes = plt.subplots(
        1, len(sampled),
        figsize=(5 * len(sampled), 4),
        squeeze=False,
    )
    fig.suptitle(
        "Unknown-anomaly spot-check\n"
        "(blue = y_pos altitude, orange = corrupted channel, dashed = onset)",
        fontsize=11, y=1.02,
    )

    for col_idx, fpath in enumerate(sampled):
        ax = axes[0][col_idx]
        df = pd.read_csv(fpath)
        fname = os.path.basename(fpath)

        # Look up onset and channel from labels
        onset   = 0
        channel = "y_vel"   # sensible fallback
        if not ulabels.empty and "fault_params" in ulabels.columns:
            row = ulabels[ulabels["filename"] == fname]
            if not row.empty:
                try:
                    params  = json.loads(row.iloc[0]["fault_params"])
                    onset   = int(params[0]["onset"])
                    channel = str(params[0].get("channel", channel))
                except (ValueError, TypeError, KeyError):
                    pass

        # Left axis — altitude trajectory
        ax.plot(df["step"], df["y_pos"], linewidth=0.8, color="#3b82d4",
                label="y_pos")
        ax.set_ylabel("y_pos (altitude)", fontsize=7, color="#3b82d4")
        ax.tick_params(axis="y", labelcolor="#3b82d4", labelsize=6)

        # Right axis — corrupted channel
        ax2 = ax.twinx()
        plot_col = channel if channel in df.columns else "y_vel"
        ax2.plot(df["step"], df[plot_col], linewidth=0.8, color="#e67e22",
                 alpha=0.8, label=plot_col)
        ax2.set_ylabel(plot_col, fontsize=7, color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22", labelsize=6)

        # Onset marker
        ax.axvline(onset, color="#e74c3c", linestyle="--", linewidth=1.2,
                   label=f"onset={onset}")

        ax.set_title(f"unknown\n{fname}", fontsize=7, pad=3)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(DATA_ROOT, "unknown_sanity_check.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Unknown sanity-check plot saved → {out_path}")


if __name__ == "__main__":
    run_sanity_check()
    run_unknown_sanity_check()
