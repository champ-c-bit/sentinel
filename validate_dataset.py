#!/usr/bin/env python3
"""
validate_dataset.py — automated, programmatic validation of the generated
Sentinel dataset.

Run this after generate_dataset.py / run_all.py / generate_unknown_episodes()
/ splits.py to catch data-integrity problems that are impractical to eyeball
file-by-file once the dataset is large.  Every check here is either exhaustive
(cheap operations: seed sets, label parsing, path existence) or explicitly
sampled (expensive per-episode reads), never "look at a few files and hope."

Usage
-----
  cd sentinel
  python data_generation/validate_dataset.py
  python data_generation/validate_dataset.py --sample 15   # bigger Phase-0 sample

Exit code 0 = all checks passed. Exit code 1 = at least one check failed
(details are printed above the final summary).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from data_generation.config import (
    RANDOM_SEED_BASE, NOMINAL_DIR, FAULTED_DIR, MULTI_FAULTED_DIR,
    LABELS_CSV, UNKNOWN_DIR, UNKNOWN_LABELS_CSV, DATA_ROOT, SENSOR_NOISE_STD,
)
from data_generation.faults import (
    FAULT_TYPES, FAULT_CLASSES, build_fault,
    _ACTUATOR_FAULT_TYPES, _SENSOR_FAULT_TYPES,
)

SPLITS_DIR = os.path.join(DATA_ROOT, "splits")

ISSUES: list[str] = []


def run_check(name: str, fn) -> None:
    print(f"\n--- {name} ---")
    try:
        problems = fn()
    except Exception as e:  # noqa: BLE001 — a crashed check IS a failed check
        problems = [f"CHECK CRASHED: {e!r}"]
    if problems:
        print(f"FAIL — {len(problems)} issue(s)")
        for p in problems[:25]:
            print(f"  - {p}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        ISSUES.extend(problems)
    else:
        print("OK")


# ═════════════════════════════════════════════════════════════════════════
# 1. Seed uniqueness — no two episodes anywhere share a seed
# ═════════════════════════════════════════════════════════════════════════

def check_seed_uniqueness() -> list[str]:
    issues: list[str] = []
    all_seeds: list[int] = []
    sources: list[str] = []

    if os.path.isdir(NOMINAL_DIR):
        for f in sorted(os.listdir(NOMINAL_DIR)):
            if f.startswith("nominal_") and f.endswith(".csv"):
                idx = int(f[len("nominal_"):-len(".csv")])
                all_seeds.append(RANDOM_SEED_BASE + idx)
                sources.append(f"nominal/{f}")

    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV, usecols=["seed", "filename"])
        all_seeds.extend(int(s) for s in df["seed"])
        sources.extend(df["filename"].tolist())

    if os.path.exists(UNKNOWN_LABELS_CSV):
        df = pd.read_csv(UNKNOWN_LABELS_CSV, usecols=["seed", "filename"])
        all_seeds.extend(int(s) for s in df["seed"])
        sources.extend(df["filename"].tolist())

    seen: dict[int, str] = {}
    for seed, src in zip(all_seeds, sources):
        if seed in seen:
            issues.append(f"seed {seed} reused by {src!r} (first used by {seen[seed]!r})")
        else:
            seen[seed] = src

    print(f"  episodes checked: {len(all_seeds)}  |  unique seeds: {len(seen)}")
    return issues


# ═════════════════════════════════════════════════════════════════════════
# 2. Registry isolation — unknown class never leaks into the 7-class dataset
# ═════════════════════════════════════════════════════════════════════════

def check_registry_isolation() -> list[str]:
    issues: list[str] = []

    if len(FAULT_CLASSES) != 7:
        issues.append(f"FAULT_CLASSES has {len(FAULT_CLASSES)} entries, expected 7")
    if len(FAULT_TYPES) != 7:
        issues.append(f"FAULT_TYPES has {len(FAULT_TYPES)} entries, expected 7")

    try:
        build_fault("unknown", np.random.default_rng(0), 1000)
        issues.append('build_fault("unknown", ...) did not raise ValueError')
    except ValueError:
        pass

    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV, usecols=["fault_type"])
        leaked = df[df["fault_type"].apply(lambda ft: "unknown" in str(ft).split("+"))]
        if len(leaked) > 0:
            issues.append(f"{len(leaked)} row(s) in labels.csv reference 'unknown' "
                          f"— it must never appear there")

    return issues


# ═════════════════════════════════════════════════════════════════════════
# 3. Multi-fault conflict-rule integrity (exhaustive over labels.csv)
# ═════════════════════════════════════════════════════════════════════════

def check_multi_fault_conflicts() -> list[str]:
    issues: list[str] = []
    if not os.path.exists(LABELS_CSV):
        return issues

    df = pd.read_csv(LABELS_CSV)
    multi = df[df["fault_type"].str.contains(r"\+", regex=True, na=False)]
    print(f"  multi-fault rows: {len(multi)}")

    for _, row in multi.iterrows():
        parts = row["fault_type"].split("+")

        if len(parts) != 2:
            issues.append(f"{row['filename']}: fault_type has {len(parts)} parts, expected 2")
            continue

        type_a, type_b = parts

        # Rule: canonical alphabetical order
        if type_a > type_b:
            issues.append(f"{row['filename']}: combo '{row['fault_type']}' not "
                          f"alphabetically sorted")

        # Rule: no duplicate types
        if type_a == type_b:
            issues.append(f"{row['filename']}: both faults are the same type ({type_a})")

        # Rule: at most one actuator fault
        if type_a in _ACTUATOR_FAULT_TYPES and type_b in _ACTUATOR_FAULT_TYPES:
            issues.append(f"{row['filename']}: both faults are actuator-affecting "
                          f"({type_a} + {type_b})")

        # Rule: two sensor faults must target different channels
        try:
            params = json.loads(row["fault_params"])
        except (ValueError, TypeError):
            issues.append(f"{row['filename']}: fault_params did not parse as JSON")
            continue

        if len(params) != 2:
            issues.append(f"{row['filename']}: fault_params has {len(params)} "
                          f"entries, expected 2")
            continue

        if type_a in _SENSOR_FAULT_TYPES and type_b in _SENSOR_FAULT_TYPES:
            ch_a = params[0].get("channel")
            ch_b = params[1].get("channel")
            if ch_a is not None and ch_a == ch_b:
                issues.append(f"{row['filename']}: both sensor faults target "
                              f"channel '{ch_a}'")

    return issues


# ═════════════════════════════════════════════════════════════════════════
# 4. Phase 0 sanity — sensor faults actually diverge, actuator faults don't
#    corrupt sensor readings.  Sampled (per-episode CSV reads are the
#    expensive part), not exhaustive.
# ═════════════════════════════════════════════════════════════════════════

def check_phase0_behavior(sample_per_type: int = 5) -> list[str]:
    issues: list[str] = []
    if not os.path.exists(LABELS_CSV):
        return issues

    labels = pd.read_csv(LABELS_CSV)
    single = labels[~labels["fault_type"].str.contains(r"\+", regex=True, na=False)]

    for ft in FAULT_TYPES:
        rows = single[single["fault_type"] == ft]
        if rows.empty:
            continue
        sampled = rows.sample(n=min(sample_per_type, len(rows)), random_state=0)

        for _, row in sampled.iterrows():
            fpath = os.path.join(FAULTED_DIR, row["filename"])
            if not os.path.exists(fpath):
                issues.append(f"{row['filename']}: file listed in labels.csv but "
                              f"missing on disk")
                continue

            params = json.loads(row["fault_params"])[0]
            onset = params["onset"]
            channel = params.get("channel")  # None for actuator faults

            df = pd.read_csv(fpath)
            post = df[df["step"] >= onset]
            if post.empty:
                continue  # episode ended before onset — nothing to check

            if channel is not None:
                # Sensor fault — reported should diverge meaningfully from
                # true on the affected channel after onset.
                diff = (post[channel] - post[f"true_{channel}"]).abs().max()
                noise_floor = SENSOR_NOISE_STD.get(channel, 0.0)
                threshold = max(5 * noise_floor, 1e-6)
                if diff <= threshold:
                    issues.append(f"{row['filename']} ({ft}): max |reported-true| "
                                  f"on {channel} post-onset is {diff:.5f}, expected "
                                  f"> {threshold:.5f} (fault appears inactive)")
            else:
                # Actuator fault — no sensor channel should show corruption
                # beyond ordinary noise.
                for col in ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "angular_vel"]:
                    diff = (post[col] - post[f"true_{col}"]).abs().max()
                    noise_floor = SENSOR_NOISE_STD.get(col, 0.0)
                    threshold = max(8 * noise_floor, 1e-6)
                    if diff > threshold:
                        issues.append(f"{row['filename']} ({ft}): actuator fault "
                                      f"shows {diff:.5f} divergence on {col} "
                                      f"(expected only baseline noise)")

    return issues


# ═════════════════════════════════════════════════════════════════════════
# 5. LOFO split integrity (only runs if data/splits/ exists)
# ═════════════════════════════════════════════════════════════════════════

def check_splits() -> list[str]:
    issues: list[str] = []
    if not os.path.isdir(SPLITS_DIR):
        print("  data/splits/ not found — skipping (run splits.py first if needed)")
        return issues

    if not os.path.exists(LABELS_CSV):
        return issues
    labels = pd.read_csv(LABELS_CSV)
    n_nominal_disk = len([f for f in os.listdir(NOMINAL_DIR) if f.endswith(".csv")]) \
        if os.path.isdir(NOMINAL_DIR) else 0

    for held_out in FAULT_TYPES:
        train_path = os.path.join(SPLITS_DIR, f"lofo_{held_out}_train.csv")
        test_path  = os.path.join(SPLITS_DIR, f"lofo_{held_out}_test.csv")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            issues.append(f"missing split files for held_out='{held_out}'")
            continue

        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)

        # No path overlap between train and test.
        overlap = set(train["path"]) & set(test["path"])
        if overlap:
            issues.append(f"[{held_out}] {len(overlap)} path(s) appear in BOTH "
                          f"train and test — evaluation leak")

        # Train must contain zero episodes referencing held_out.
        leaked = train[train["fault_type"].apply(
            lambda ft: held_out in str(ft).split("+"))]
        if len(leaked) > 0:
            issues.append(f"[{held_out}] {len(leaked)} train row(s) reference "
                          f"the held-out type")

        # Test faulted rows must all reference held_out (nominal rows excepted).
        test_faulted = test[test["fault_type"] != "nominal"]
        bad_test = test_faulted[~test_faulted["fault_type"].apply(
            lambda ft: held_out in str(ft).split("+"))]
        if len(bad_test) > 0:
            issues.append(f"[{held_out}] {len(bad_test)} test row(s) don't "
                          f"reference the held-out type")

        # Nominal counts sum correctly.
        n_nom_train = len(train[train["fault_type"] == "nominal"])
        n_nom_test  = len(test[test["fault_type"] == "nominal"])
        if n_nominal_disk and (n_nom_train + n_nom_test) != n_nominal_disk:
            issues.append(f"[{held_out}] nominal train+test = "
                          f"{n_nom_train + n_nom_test}, expected {n_nominal_disk}")

        # Faulted counts sum to the full labels.csv total.
        n_fault_total = len(train[train["fault_type"] != "nominal"]) + len(test_faulted)
        if n_fault_total != len(labels):
            issues.append(f"[{held_out}] faulted train+test = {n_fault_total}, "
                          f"expected {len(labels)}")

        # Every path must exist on disk.
        for p in pd.concat([train["path"], test["path"]]):
            if not os.path.exists(p):
                issues.append(f"[{held_out}] manifest path does not exist: {p}")

    return issues


# ═════════════════════════════════════════════════════════════════════════
# 6. Class balance and unknown-vs-known visual-distinctness reminder
# ═════════════════════════════════════════════════════════════════════════

def check_class_balance() -> list[str]:
    issues: list[str] = []
    if not os.path.exists(LABELS_CSV):
        return issues

    labels = pd.read_csv(LABELS_CSV)
    single = labels[~labels["fault_type"].str.contains(r"\+", regex=True, na=False)]
    counts = single["fault_type"].value_counts()
    print("  per-type single-fault counts:")
    for ft in FAULT_TYPES:
        n = counts.get(ft, 0)
        print(f"    {ft:<28}: {n}")

    if counts.nunique() > 1:
        issues.append(f"single-fault class counts are not balanced: "
                      f"{counts.to_dict()}")

    if os.path.exists(UNKNOWN_LABELS_CSV):
        print("  NOTE: unknown-anomaly episodes exist — remember to visually "
              "compare unknown_sanity_check.png against sanity_check.png; "
              "that check can't be automated.")

    return issues


# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated Sentinel dataset.")
    parser.add_argument("--sample", type=int, default=5,
                        help="Episodes sampled per fault type for the Phase-0 check (default 5).")
    args = parser.parse_args()

    if not os.path.exists(LABELS_CSV):
        print(f"labels.csv not found at {LABELS_CSV}. Run generate_dataset.py first.")
        sys.exit(1)

    run_check("Seed uniqueness", check_seed_uniqueness)
    run_check("Registry isolation (unknown class)", check_registry_isolation)
    run_check("Multi-fault conflict rules", check_multi_fault_conflicts)
    run_check("Phase 0 behavior (sampled)", lambda: check_phase0_behavior(args.sample))
    run_check("LOFO split integrity", check_splits)
    run_check("Class balance", check_class_balance)

    print("\n" + "=" * 60)
    if ISSUES:
        print(f"RESULT: FAIL — {len(ISSUES)} total issue(s) found")
        print("=" * 60)
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
