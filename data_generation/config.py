"""
config.py — all tuneable constants for data generation.
"""

import os

# ── Gymnasium environment ────────────────────────────────────────────────────
ENV_ID       = "LunarLander-v3"
RENDER_MODE  = None               # set to "human" for visual debugging

# Wind / turbulence — enabled for natural variability in nominal episodes.
# LunarLander-v3 accepts enable_wind and wind_power as kwargs.
ENABLE_WIND  = True
WIND_POWER   = 10.0               # moderate; range 0–20
TURBULENCE   = 1.5                # turbulence_power; range 0–2

# ── Episode control ──────────────────────────────────────────────────────────
MAX_STEPS        = 1000
RANDOM_SEED_BASE = 42

# ── Dataset sizes ────────────────────────────────────────────────────────────
N_NOMINAL_EPISODES    = 300
N_FAULT_EPISODES_EACH = 50        # per fault type; 7 types → 350 faulted total
MAX_SIMULTANEOUS_FAULTS = 2       # faults per multi-fault episode
N_MULTI_FAULT_EPISODES  = 100     # dual-fault episodes (opt-in; CLI default is 0)
N_UNKNOWN_EPISODES      = 50      # unknown-anomaly episodes (opt-in; separate pipeline)

# ── Fault onset window (fraction of episode length) ─────────────────────────
FAULT_ONSET_EARLY = 0.15          # earliest onset: 15% into episode
FAULT_ONSET_LATE  = 0.55          # latest onset:   55% into episode

# ── Observation channel names (LunarLander-v3, 8-dim) ───────────────────────
OBS_COLUMNS = [
    "x_pos",        # 0 — horizontal position
    "y_pos",        # 1 — altitude (0 = ground)
    "x_vel",        # 2 — horizontal velocity
    "y_vel",        # 3 — vertical velocity (negative = descending)
    "angle",        # 4 — lander tilt (rad)
    "angular_vel",  # 5 — angular rate
    "left_leg",     # 6 — left leg ground contact (bool)
    "right_leg",    # 7 — right leg ground contact (bool)
]

# Channels that represent continuous physical quantities (not booleans).
# Fault injectors only target these.
CONTINUOUS_CHANNELS = ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "angular_vel"]

# ── Sensor noise (Gaussian std-dev added to every obs channel) ───────────────
SENSOR_NOISE_STD = {
    "x_pos":       0.005,
    "y_pos":       0.005,
    "x_vel":       0.010,
    "y_vel":       0.010,
    "angle":       0.005,
    "angular_vel": 0.010,
    "left_leg":    0.0,
    "right_leg":   0.0,
}

# ── Output layout ────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.abspath(os.path.join(_HERE, "..", "data"))
NOMINAL_DIR = os.path.join(DATA_ROOT, "nominal")
FAULTED_DIR       = os.path.join(DATA_ROOT, "faulted")
MULTI_FAULTED_DIR  = os.path.join(DATA_ROOT, "faulted_multi")
LABELS_CSV         = os.path.join(DATA_ROOT, "labels.csv")
UNKNOWN_DIR        = os.path.join(DATA_ROOT, "unknown")
UNKNOWN_LABELS_CSV = os.path.join(DATA_ROOT, "unknown_labels.csv")
