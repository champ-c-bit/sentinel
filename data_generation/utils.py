"""utils.py — shared helpers for the data-generation pipeline."""

from __future__ import annotations

import pandas as pd


def safe_landing(df: pd.DataFrame) -> bool:
    """
    Heuristic: episode ended with terminated=1 (ground contact) and
    the final true vertical velocity was gentle (> -0.3 m/s equivalent).

    Uses true_y_vel (raw environment reading) rather than y_vel (reported,
    fault-corrupted + noisy) so that sensor faults on the y_vel channel —
    sensor_dropout, measurement_delay, sensor_bias_drift, etc. — cannot
    corrupt the crash/safe verdict.  terminated comes from env.step() and
    is never sensor-corrupted.

    Falls back to y_vel when true_y_vel is absent (pre-Phase-0 CSVs on disk
    that lack the true_<channel> columns).  Nominal episodes only differ by
    small Gaussian noise so the verdict is unlikely to flip near the threshold,
    but this keeps the function robust to old files without crashing.
    """
    last = df.iloc[-1]
    vel_col = "true_y_vel" if "true_y_vel" in df.columns else "y_vel"
    return bool(last["terminated"] == 1 and last[vel_col] > -0.3)
