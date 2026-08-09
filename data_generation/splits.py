"""
splits.py — Leave-One-Fault-Out (LOFO) split manifest generator.

For each of the 7 known fault types, produces a train/test split that fully
excludes that type from training.  No episode data is regenerated or duplicated —
manifests are pure CSV index files over episodes that already exist on disk.

Output layout
-------------
  data/splits/
    lofo_sensor_dropout_train.csv
    lofo_sensor_dropout_test.csv
    lofo_thrust_degradation_train.csv
    … (14 files total, 2 per fault type)

Manifest schema
---------------
Each manifest CSV has columns:
  path        — absolute path to the episode CSV
  fault_type  — the episode's fault_type string ("nominal" for nominals)
  split       — "train" or "test"

Multi-fault episode handling
----------------------------
A multi-fault episode labeled "actuator_stuck+sensor_dropout" is excluded from
the train set for *both* actuator_stuck and sensor_dropout LOFO splits.
The check uses split("+") — not substring matching — to avoid partial name
collisions if future types share name prefixes.

Nominal proportionality (Option B)
-----------------------------------
For held_out_type with f_held matching episodes out of f_total total faulted:
  p = f_held / f_total
  n_test_nominal = round(p * N_nominal)
The same shuffle seed is used across all 7 types, so nominal test slices
partially overlap — this is intentional (nominals are class-agnostic).

Usage
-----
  cd sentinel
  python data_generation/splits.py

  # or with explicit paths / seed:
  python data_generation/splits.py --labels data/labels.csv \\
      --nominal data/nominal --out data/splits --seed 0
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from data_generation.config import (
    LABELS_CSV, NOMINAL_DIR, DATA_ROOT, FAULTED_DIR, MULTI_FAULTED_DIR,
)
from data_generation.faults import FAULT_TYPES

# Splits are a derived analysis artifact — path lives here, not in config.py.
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fault_type_matches(fault_type_str: str, held_out: str) -> bool:
    """
    Return True if held_out is one of the fault types in fault_type_str.

    Works correctly for both single-fault ("sensor_dropout") and multi-fault
    ("actuator_stuck+sensor_dropout") strings.
    """
    return held_out in fault_type_str.split("+")


def _abs_path(row: pd.Series) -> str:
    """
    Reconstruct the absolute path to an episode CSV from a labels.csv row.

    labels.csv stores paths relative to the type-specific directory root:
      single-fault : "sensor_dropout/sensor_dropout_0000.csv"   → FAULTED_DIR
      multi-fault  : "actuator_stuck+sensor_bias_drift/multi_..." → MULTI_FAULTED_DIR
    """
    if "+" in str(row["fault_type"]):
        return os.path.join(MULTI_FAULTED_DIR, row["filename"])
    return os.path.join(FAULTED_DIR, row["filename"])


# ── main function ─────────────────────────────────────────────────────────────

def make_lofo_splits(
    labels_csv: str = LABELS_CSV,
    nominal_dir: str = NOMINAL_DIR,
    splits_dir: str = SPLITS_DIR,
    seed: int = 0,
) -> None:
    """
    Generate all 7 LOFO train/test manifests and write them to splits_dir.

    Parameters
    ----------
    labels_csv  : Path to labels.csv (faulted + multi-fault episodes).
    nominal_dir : Directory containing nominal episode CSVs.
    splits_dir  : Output directory for manifest files.
    seed        : RNG seed for nominal shuffle (same across all 7 types).
    """
    if not os.path.exists(labels_csv):
        print(f"[splits] labels.csv not found at {labels_csv}. "
              "Run generate_dataset.py first.")
        return

    if not os.path.isdir(nominal_dir):
        print(f"[splits] nominal dir not found at {nominal_dir}. "
              "Run generate_dataset.py first.")
        return

    labels = pd.read_csv(labels_csv)
    f_total = len(labels)

    # Sorted nominal file list — deterministic ordering before shuffle.
    nominal_fnames = sorted(f for f in os.listdir(nominal_dir) if f.endswith(".csv"))
    nominal_paths  = [os.path.join(nominal_dir, f) for f in nominal_fnames]
    n_nominal = len(nominal_paths)

    # Shuffle nominals once with the fixed seed — same order for all 7 types.
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n_nominal)
    shuffled_nominal_fnames = [nominal_fnames[i] for i in shuffled_idx]
    shuffled_nominal_paths  = [nominal_paths[i]  for i in shuffled_idx]

    os.makedirs(splits_dir, exist_ok=True)

    print(f"\nGenerating LOFO splits → {splits_dir}")
    print(f"  Total faulted episodes in labels.csv : {f_total}")
    print(f"  Total nominal episodes               : {n_nominal}")
    print()

    for held_out in FAULT_TYPES:
        # ── Faulted split ─────────────────────────────────────────────────
        mask_test  = labels["fault_type"].apply(
            lambda ft: _fault_type_matches(ft, held_out)
        )
        faulted_test  = labels[mask_test].copy()
        faulted_train = labels[~mask_test].copy()

        f_held = len(faulted_test)

        # ── Nominal split (Option B proportional) ─────────────────────────
        p = f_held / f_total if f_total > 0 else 0.0
        n_test_nom  = round(p * n_nominal)
        n_train_nom = n_nominal - n_test_nom

        nom_test_paths  = shuffled_nominal_paths[:n_test_nom]
        nom_test_fnames = shuffled_nominal_fnames[:n_test_nom]
        nom_train_paths  = shuffled_nominal_paths[n_test_nom:]
        nom_train_fnames = shuffled_nominal_fnames[n_test_nom:]

        # ── Build manifest DataFrames ──────────────────────────────────────
        def _faulted_rows(df_subset: pd.DataFrame, split_label: str) -> pd.DataFrame:
            return pd.DataFrame({
                "path":       df_subset.apply(_abs_path, axis=1).values,
                "fault_type": df_subset["fault_type"].values,
                "split":      split_label,
            })

        def _nominal_rows(paths: list[str], split_label: str) -> pd.DataFrame:
            return pd.DataFrame({
                "path":       paths,
                "fault_type": "nominal",
                "split":      split_label,
            })

        train_df = pd.concat([
            _faulted_rows(faulted_train, "train"),
            _nominal_rows(nom_train_paths, "train"),
        ], ignore_index=True)

        test_df = pd.concat([
            _faulted_rows(faulted_test, "test"),
            _nominal_rows(nom_test_paths, "test"),
        ], ignore_index=True)

        # ── Write manifests ────────────────────────────────────────────────
        slug = held_out
        train_path = os.path.join(splits_dir, f"lofo_{slug}_train.csv")
        test_path  = os.path.join(splits_dir, f"lofo_{slug}_test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        # ── Print summary ──────────────────────────────────────────────────
        print(f"  [{held_out}]")
        print(f"    held-out faulted : {f_held:>4}  |  "
              f"train faulted : {len(faulted_train):>4}  |  "
              f"test faulted : {f_held:>4}")
        print(f"    nominal p={p:.3f} : {n_nominal:>4}  |  "
              f"train nominal : {n_train_nom:>4}  |  "
              f"test nominal : {n_test_nom:>4}")
        print(f"    total train : {len(train_df):>4}    total test : {len(test_df):>4}")
        print()

    print(f"14 manifest files written to {splits_dir}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LOFO train/test split manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels",  default=LABELS_CSV,   help="Path to labels.csv")
    parser.add_argument("--nominal", default=NOMINAL_DIR,  help="Nominal episodes directory")
    parser.add_argument("--out",     default=SPLITS_DIR,   help="Output directory for manifests")
    parser.add_argument("--seed",    type=int, default=0,  help="RNG seed for nominal shuffle")
    args = parser.parse_args()

    make_lofo_splits(
        labels_csv  = args.labels,
        nominal_dir = args.nominal,
        splits_dir  = args.out,
        seed        = args.seed,
    )
