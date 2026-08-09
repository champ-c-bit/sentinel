"""
scenario_generator.py — batch dataset builder.

Output layout
-------------
  data/
    nominal/
      nominal_0000.csv … nominal_0299.csv
    faulted/
      sensor_dropout/
        sensor_dropout_0000.csv … _0049.csv
      thrust_degradation/
        …
      (one sub-folder per fault type)
    faulted_multi/
      <typeA>+<typeB>/
        multi_<typeA>+<typeB>_0000.csv … (combo folder created on demand)
    unknown/
      unknown_0000.csv … (written by generate_unknown_episodes(), not generate_dataset())
    labels.csv         ← ground-truth index for all faulted episodes
    unknown_labels.csv ← ground-truth index for unknown episodes

Labels schema (Option C)
------------------------
Every row — single-fault, multi-fault, and unknown — has a `fault_params` column
containing a JSON list of each fault's .params dict:
  single-fault : "[{\"onset\": 37, \"channel\": \"x_vel\"}]"
  multi-fault  : "[{\"onset\": 41, ...}, {\"onset\": 67, ...}]"
  unknown      : "[{\"onset\": 52, \"channel\": \"y_vel\", \"theta\": ..., ...}]"
No flat param_* columns are written.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    N_NOMINAL_EPISODES,
    N_FAULT_EPISODES_EACH,
    N_UNKNOWN_EPISODES,
    RANDOM_SEED_BASE,
    NOMINAL_DIR,
    FAULTED_DIR,
    MULTI_FAULTED_DIR,
    LABELS_CSV,
    UNKNOWN_DIR,
    UNKNOWN_LABELS_CSV,
    MAX_STEPS,
)
from .faults import FAULT_TYPES, build_fault, build_multi_fault, build_unknown_fault
from .simulation import run_episode
from .utils import safe_landing



# ── main entry point ──────────────────────────────────────────────────────────

def generate_dataset(
    n_nominal: int = N_NOMINAL_EPISODES,
    n_fault_each: int = N_FAULT_EPISODES_EACH,
    n_multi: int = 0,
) -> str:
    """
    Generate the complete dataset.

    Parameters
    ----------
    n_nominal    : Number of nominal (fault-free) episodes.
    n_fault_each : Number of episodes per single fault type.
    n_multi      : Number of dual-fault episodes (default 0 — opt-in).

    Returns the path to labels.csv.
    """
    os.makedirs(NOMINAL_DIR, exist_ok=True)
    for ft in FAULT_TYPES:
        os.makedirs(os.path.join(FAULTED_DIR, ft), exist_ok=True)

    label_rows: list[dict] = []

    # ── Nominal ───────────────────────────────────────────────────────────
    print(f"\n[1/3] Generating {n_nominal} nominal episodes …")
    for i in tqdm(range(n_nominal), unit="ep"):
        seed = RANDOM_SEED_BASE + i
        df   = run_episode(seed=seed, faults=None)

        fname = f"nominal_{i:04d}.csv"
        df.to_csv(os.path.join(NOMINAL_DIR, fname), index=False)

    # ── Single-fault ──────────────────────────────────────────────────────
    print(f"\n[2/3] Generating {n_fault_each} episodes × {len(FAULT_TYPES)} fault types …")
    episode_counter = 0
    fault_safe_counts: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}
    for ft in FAULT_TYPES:
        out_dir = os.path.join(FAULTED_DIR, ft)
        print(f"  fault type: {ft}")
        for i in tqdm(range(n_fault_each), unit="ep", leave=False):
            seed = RANDOM_SEED_BASE + n_nominal + episode_counter
            rng  = np.random.default_rng(seed)

            fault = build_fault(ft, rng, MAX_STEPS)
            df    = run_episode(seed=seed, faults=[fault])

            if safe_landing(df):
                fault_safe_counts[ft] += 1

            fname = f"{ft}_{i:04d}.csv"
            df.to_csv(os.path.join(out_dir, fname), index=False)

            # Option C labels schema: fault_params JSON list
            row = {
                "filename":    os.path.join(ft, fname),
                "fault_type":  ft,
                "seed":        seed,
                "n_steps":     len(df),
                "fault_params": json.dumps([fault.params]),
            }
            label_rows.append(row)

            episode_counter += 1

    # ── Multi-fault ───────────────────────────────────────────────────────
    if n_multi > 0:
        print(f"\n[3/3] Generating {n_multi} dual-fault episodes …")
        for i in tqdm(range(n_multi), unit="ep"):
            # Continue the global seed counter — never restart it.
            seed = RANDOM_SEED_BASE + n_nominal + episode_counter
            episode_counter += 1

            rng = np.random.default_rng(seed)
            fault_pair = build_multi_fault(rng, MAX_STEPS)
            fault_a, fault_b = fault_pair[0], fault_pair[1]

            df = run_episode(seed=seed, faults=fault_pair)

            combo = fault_a.fault_type + "+" + fault_b.fault_type
            out_dir = os.path.join(MULTI_FAULTED_DIR, combo)
            os.makedirs(out_dir, exist_ok=True)

            fname = f"multi_{combo}_{i:04d}.csv"
            df.to_csv(os.path.join(out_dir, fname), index=False)

            row = {
                "filename":    os.path.join(combo, fname),
                "fault_type":  combo,
                "seed":        seed,
                "n_steps":     len(df),
                "fault_params": json.dumps([fault_a.params, fault_b.params]),
            }
            label_rows.append(row)
    else:
        print("\n[3/3] Multi-fault generation skipped (n_multi=0).")

    # ── labels.csv ────────────────────────────────────────────────────────
    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(LABELS_CSV, index=False)

    # ── summary ───────────────────────────────────────────────────────────
    nominal_files = [
        f for f in os.listdir(NOMINAL_DIR) if f.endswith(".csv")
    ]
    safe_count = 0
    for fname in nominal_files:
        df = pd.read_csv(os.path.join(NOMINAL_DIR, fname))
        if safe_landing(df):
            safe_count += 1

    total = n_nominal + n_fault_each * len(FAULT_TYPES) + n_multi
    print("\n" + "=" * 52)
    print("DATASET SUMMARY")
    print("=" * 52)
    print(f"  Nominal episodes          : {n_nominal}")
    print(f"    └─ safe landings        : {safe_count} / {n_nominal}")
    for ft in FAULT_TYPES:
        safe_f = fault_safe_counts[ft]
        print(f"  {ft:<32}: {n_fault_each}  (safe: {safe_f:>3} / {n_fault_each})")
    if n_multi > 0:
        print(f"  {'dual-fault (total)':<32}: {n_multi}")
    print(f"  {'TOTAL':<34}: {total}")
    print("=" * 52)
    print(f"\nLabels CSV : {LABELS_CSV}")
    print(f"Data root  : {os.path.dirname(NOMINAL_DIR)}")

    return LABELS_CSV


# ── Seed-safety helper ────────────────────────────────────────────────────────

def _collect_used_seeds() -> set[int]:
    """
    Return the set of all RNG seeds already consumed by episodes on disk.

    Sources:
      1. labels.csv   — seeds for all single-fault and multi-fault episodes.
      2. unknown_labels.csv — seeds for any prior unknown episodes.
      3. Nominal episodes  — reconstructed as {RANDOM_SEED_BASE + i for i in
         range(len(nominal_files))}.

    Assumption on (3): nominal seeds are contiguous from RANDOM_SEED_BASE with
    no gaps, matching the documented generation scheme
    (seed = RANDOM_SEED_BASE + i).  This holds as long as nominal generation was
    never partially run in a way that left gaps.  Verify against on-disk state
    when using the guard in safety-critical contexts.
    """
    seeds: set[int] = set()

    # Faulted + multi-fault seeds from labels.csv
    if os.path.exists(LABELS_CSV):
        try:
            df = pd.read_csv(LABELS_CSV, usecols=["seed"])
            seeds.update(int(s) for s in df["seed"])
        except ValueError:
            # Degenerate case: labels.csv written with no rows (n_fault_each=0,
            # n_multi=0) — it has no columns, so usecols=["seed"] raises.
            pass

    # Previous unknown seeds from unknown_labels.csv
    if os.path.exists(UNKNOWN_LABELS_CSV):
        try:
            df = pd.read_csv(UNKNOWN_LABELS_CSV, usecols=["seed"])
            seeds.update(int(s) for s in df["seed"])
        except ValueError:
            pass

    # Nominal seeds (reconstructed from file count — assumed contiguous)
    if os.path.isdir(NOMINAL_DIR):
        n_nominal_on_disk = len([f for f in os.listdir(NOMINAL_DIR) if f.endswith(".csv")])
        seeds.update(RANDOM_SEED_BASE + i for i in range(n_nominal_on_disk))

    return seeds


# ── Unknown anomaly generation (separate from main dataset pipeline) ──────────

def generate_unknown_episodes(
    n_unknown: int = N_UNKNOWN_EPISODES,
    seed_offset: int = None,  # type: ignore[assignment]  — required; see docstring
) -> str:
    """
    Generate a batch of out-of-distribution UnknownAnomaly episodes.

    This function is intentionally separate from generate_dataset() — calling
    generate_dataset() with any arguments never produces unknown-labeled episodes.

    Parameters
    ----------
    n_unknown   : Number of unknown episodes to generate.
    seed_offset : First seed to use.  **Required — no safe default exists.**
                  The caller must supply a value that doesn't collide with any
                  seed already on disk.  A safe choice after a standard run:
                      max(pd.read_csv(LABELS_CSV)["seed"]) + 1
                  The function enforces no-collision via _collect_used_seeds()
                  and raises ValueError before touching any file if seeds clash.

    Returns the path to unknown_labels.csv.
    """
    if seed_offset is None:
        raise TypeError(
            "generate_unknown_episodes() requires an explicit seed_offset. "
            "Pass seed_offset=<int> to continue from the correct position in "
            "the global seed counter.  Safe choice after a standard run:\n"
            "    import pandas as pd\n"
            f"    seed_offset = int(pd.read_csv(r'{LABELS_CSV}')['seed'].max()) + 1"
        )

    # ── Seed-collision guard ──────────────────────────────────────────────
    used_seeds = _collect_used_seeds()
    intended   = {seed_offset + i for i in range(n_unknown)}
    collisions = intended & used_seeds
    if collisions:
        raise ValueError(
            f"seed_offset={seed_offset} causes {len(collisions)} collision(s) "
            f"with seeds already on disk: {sorted(collisions)[:10]}"
            + (" …" if len(collisions) > 10 else "")
        )

    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    label_rows: list[dict] = []

    print(f"\nGenerating {n_unknown} unknown-anomaly episodes "
          f"(seed_offset={seed_offset}) …")
    for i in tqdm(range(n_unknown), unit="ep"):
        seed = seed_offset + i
        rng  = np.random.default_rng(seed)
        fault = build_unknown_fault(rng, MAX_STEPS)
        df    = run_episode(seed=seed, faults=[fault])

        fname = f"unknown_{i:04d}.csv"
        df.to_csv(os.path.join(UNKNOWN_DIR, fname), index=False)

        label_rows.append({
            "filename":    fname,
            "fault_type":  "unknown",
            "seed":        seed,
            "n_steps":     len(df),
            "fault_params": json.dumps([fault.params]),
        })

    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(UNKNOWN_LABELS_CSV, index=False)

    print(f"\nUnknown episodes : {n_unknown}")
    print(f"Output dir       : {UNKNOWN_DIR}")
    print(f"Labels CSV       : {UNKNOWN_LABELS_CSV}")

    return UNKNOWN_LABELS_CSV
