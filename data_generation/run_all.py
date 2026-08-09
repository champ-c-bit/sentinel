#!/usr/bin/env python3
"""
run_all.py — one-shot script that generates the full dataset then
immediately runs sanity_check.py.

Usage
-----
  cd sentinel
  python data_generation/run_all.py                    # full dataset
  python data_generation/run_all.py --nominal 5 --fault-each 3  # smoke test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generation.scenario_generator import generate_dataset
from data_generation.config import (
    N_NOMINAL_EPISODES, N_FAULT_EPISODES_EACH, LABELS_CSV, NOMINAL_DIR,
)
from data_generation.faults import FAULT_TYPES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset + run sanity checks in one step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nominal",    type=int, default=N_NOMINAL_EPISODES, metavar="N")
    parser.add_argument("--fault-each", type=int, default=N_FAULT_EPISODES_EACH,
                        metavar="M", dest="fault_each")
    parser.add_argument("--multi-fault-episodes", type=int, default=0,
                        metavar="K", dest="multi_fault_episodes",
                        help="Number of dual-fault episodes to generate (default: 0).")
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip the sanity check plots after generation.")
    parser.add_argument("--make-splits", action="store_true",
                        dest="make_splits",
                        help="Generate LOFO train/test split manifests after dataset.")
    args = parser.parse_args()

    # ── Step 1: generate ─────────────────────────────────────────────────
    total = args.nominal + args.fault_each * len(FAULT_TYPES) + args.multi_fault_episodes
    print(f"==> Generating {total} episodes "
          f"({args.nominal} nominal + {args.fault_each}×{len(FAULT_TYPES)} faulted + "
          f"{args.multi_fault_episodes} dual-fault) …")
    labels_csv = generate_dataset(
        n_nominal=args.nominal,
        n_fault_each=args.fault_each,
        n_multi=args.multi_fault_episodes,
    )

    # ── Step 2: sanity check ─────────────────────────────────────────────
    if not args.skip_check:
        print("\n==> Running sanity_check.py …")
        # Import here so generation errors surface before plotting starts
        from data_generation.sanity_check import run_sanity_check
        run_sanity_check()
    else:
        print("Sanity check skipped (--skip-check).")

    # ── Step 3: LOFO splits ───────────────────────────────────────────────
    if args.make_splits:
        print("\n==> Generating LOFO splits …")
        from data_generation.splits import make_lofo_splits, SPLITS_DIR
        make_lofo_splits(LABELS_CSV, NOMINAL_DIR, SPLITS_DIR)
        print(f"Splits written to: {SPLITS_DIR}")

    print(f"\nAll done.  Labels: {labels_csv}")


if __name__ == "__main__":
    main()
