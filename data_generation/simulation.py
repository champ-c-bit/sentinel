"""
simulation.py — single-episode runner.

Wraps Gymnasium's LunarLander-v3 environment with:
  - Wind/turbulence enabled for natural variability
  - Scripted landing policy (heuristic, not pure "always fire main engine")
  - Stateful fault injection via BaseFault subclasses
  - Per-channel sensor noise
  - Full observation + action logging

Returned DataFrame columns:
    step,
    true_x_pos, true_y_pos, true_x_vel, true_y_vel,
    true_angle, true_angular_vel, true_left_leg, true_right_leg,
    x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg, right_leg,
    action, reward, terminated, truncated, fault_label, fault_active

The true_<channel> columns hold the raw environment reading for that step.
The plain <channel> columns hold the fault-corrupted + sensor-noisy reading
that the flight-computer policy actually acted on.  For nominal episodes the
two sets differ only by baseline sensor noise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym

from .config import (
    ENV_ID, MAX_STEPS, OBS_COLUMNS, SENSOR_NOISE_STD,
    ENABLE_WIND, WIND_POWER, TURBULENCE,
)
from .faults import BaseFault


# ── scripted landing policy ──────────────────────────────────────────────────

def _scripted_action(obs: np.ndarray) -> int:
    """
    Port of the official Gymnasium LunarLander heuristic policy.

    Key insight about the coordinate system:
      - y_vel > 0 means the lander is RISING  (just spawned, bouncing)
      - y_vel < 0 means the lander is FALLING (normal descent)
      - The main engine fires UPWARD — it decelerates a falling lander.
        Firing it while y_vel > 0 makes the lander climb further away.

    The policy:
      1. Computes an angle target to steer toward the landing pad centre.
      2. Computes a hover target just above the ground proportional to |x|.
      3. Decides whether to hover (main engine), rotate (side), or do nothing.
      4. Once legs touch, suppresses rotation and gently brakes the descent.

    obs layout: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, ll, rl]
    actions:    0=nothing, 1=left engine, 2=main engine, 3=right engine
    """
    x_pos, y_pos, x_vel, y_vel, angle, ang_vel, ll, rl = obs

    # Angle target: point nose toward pad centre, dampened by horizontal vel
    angle_targ = x_pos * 0.5 + x_vel * 1.0
    angle_targ = float(np.clip(angle_targ, -0.4, 0.4))

    # Hover target: stay proportionally above ground relative to |x offset|
    hover_targ = 0.55 * abs(x_pos)

    # PD control terms
    angle_todo = (angle_targ - angle) * 0.5 - ang_vel * 1.0
    hover_todo = (hover_targ - y_pos) * 0.5 - y_vel * 0.5   # y_vel<0 = falling → positive hover_todo

    # Once either leg touches, stop rotating and just brake the fall gently
    if ll or rl:
        angle_todo = 0.0
        hover_todo = -y_vel * 0.5

    # Action selection
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2   # main engine — need to decelerate descent
    elif angle_todo < -0.05:
        return 3   # right engine — rotate right
    elif angle_todo > 0.05:
        return 1   # left engine  — rotate left
    else:
        return 0   # do nothing


# ── sensor noise ─────────────────────────────────────────────────────────────

def _add_sensor_noise(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    obs = obs.copy()
    for i, col in enumerate(OBS_COLUMNS):
        std = SENSOR_NOISE_STD.get(col, 0.0)
        if std > 0.0:
            obs[i] += rng.normal(0.0, std)
    return obs


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(
    seed: int,
    faults: Optional[list[BaseFault]] = None,
    render: bool = False,
) -> pd.DataFrame:
    """
    Run one full LunarLander-v3 episode and return a per-timestep DataFrame.

    Parameters
    ----------
    seed  : RNG seed — controls both environment physics and noise.
    faults: List of instantiated fault objects, or None/[] for a nominal episode.
            Single-fault: pass [f]. Multi-fault: pass [f1, f2].
            Faults are applied in list order; (action, obs_faulted) threads
            through the chain — each fault sees the output of the previous one.
    render: Open a visual window (slow; debugging only).
    """
    render_mode = "human" if render else None

    env_kwargs: dict = {}
    if ENABLE_WIND:
        env_kwargs["enable_wind"]       = True
        env_kwargs["wind_power"]        = WIND_POWER
        env_kwargs["turbulence_power"]  = TURBULENCE

    env = gym.make(ENV_ID, render_mode=render_mode, **env_kwargs)

    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    # obs_reported is what the flight computer sees: fault-corrupted + noisy.
    # Initialise through the noise pipeline so step 0 is consistent with all
    # subsequent steps (avoids one perfectly clean policy decision at episode start).
    obs_reported = _add_sensor_noise(obs.copy(), rng)

    rows: list[dict] = []
    _faults: list[BaseFault] = faults if faults else []
    if _faults:
        fault_label = "+".join(f.fault_type for f in _faults)
    else:
        fault_label = "nominal"

    for step in range(MAX_STEPS):
        # 1. Scripted policy acts on the reported (faulted + noisy) observation
        action = _scripted_action(obs_reported)

        # 2. Chain fault injection — each fault receives the accumulated
        #    (action, obs_faulted) from the previous fault in the list.
        #    fault_active is True if any fault in the chain is active this step.
        obs_faulted = obs
        any_active  = False
        for fault in _faults:
            action, obs_faulted, active = fault.inject(step, action, obs_faulted)
            any_active = any_active or active
        fault_active = any_active

        # 3. Add sensor noise to the (post-fault) observation — this becomes
        #    what the policy sees on the NEXT step
        obs_reported = _add_sensor_noise(obs_faulted, rng)

        # 4. Step the environment with the (possibly fault-modified) action
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # 5. Record — two separate loops keep true_* columns grouped before
        #    the plain reported columns (not interleaved)
        row: dict = {"step": step}
        for i, col in enumerate(OBS_COLUMNS):
            row[f"true_{col}"] = float(obs[i])
        for i, col in enumerate(OBS_COLUMNS):
            row[col] = float(obs_reported[i])
        row["action"]       = int(action)
        row["reward"]       = float(reward)
        row["terminated"]   = int(terminated)
        row["truncated"]    = int(truncated)
        row["fault_label"]  = fault_label
        row["fault_active"] = int(fault_active)
        rows.append(row)

        obs = next_obs

        if terminated or truncated:
            break

    env.close()
    return pd.DataFrame(rows)
