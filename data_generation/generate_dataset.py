#!/usr/bin/env python3
"""
generate_dataset.py — CLI entry point for the data-generation pipeline.

Usage
-----
  # Full dataset (300 nominal + 350 faulted)
  python data_generation/generate_dataset.py

  # Quick smoke test
  python data_generation/generate_dataset.py --nominal 5 --fault-each 3

Output
------
  data/nominal/          300 nominal CSV episodes
  data/faulted/<type>/   50 CSV episodes per fault type (7 types)
  data/labels.csv        ground-truth record for every faulted episode
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generation.scenario_generator import generate_dataset
from data_generation.config import N_NOMINAL_EPISODES, N_FAULT_EPISODES_EACH
from data_generation.faults import FAULT_TYPES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the labeled lunar-descent anomaly dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nominal", type=int, default=N_NOMINAL_EPISODES,
        metavar="N",
        help="Number of nominal (fault-free) episodes.",
    )
    parser.add_argument(
        "--fault-each", type=int, default=N_FAULT_EPISODES_EACH,
        metavar="M",
        dest="fault_each",
        help="Number of episodes per fault type.",
    )
    parser.add_argument(
        "--multi-fault-episodes", type=int, default=0,
        metavar="K",
        dest="multi_fault_episodes",
        help="Number of dual-fault episodes to generate (default: 0).",
    )
    args = parser.parse_args()

    total = args.nominal + args.fault_each * len(FAULT_TYPES) + args.multi_fault_episodes
    print(f"Generating {args.nominal} nominal + "
          f"{args.fault_each} × {len(FAULT_TYPES)} faulted + "
          f"{args.multi_fault_episodes} dual-fault = {total} episodes total.")

    labels_csv = generate_dataset(
        n_nominal=args.nominal,
        n_fault_each=args.fault_each,
        n_multi=args.multi_fault_episodes,
    )
    print(f"\nDone.  Labels: {labels_csv}")


if __name__ == "__main__":
    main()
