"""
faults.py — fault injectors for the lunar-descent dataset.

Design
------
Each fault is a class that:
  1. Holds all randomised parameters (onset, severity, channel …) sampled
     at construction time via a numpy Generator.
  2. Exposes  .inject(step, action, obs) -> (action, obs, fault_active)
     which is called once per timestep inside the simulation loop.
  3. Carries a  .params  dict that is written verbatim into labels.csv so
     every ground-truth record is fully self-describing.

The 7 labelled fault types are grounded in real mission failures:
  - sensor_dropout          (Luna-25)
  - thrust_degradation      (Chandrayaan-2)
  - filter_misclassification(HAKUTO-R M1)
  - measurement_delay       (Resilience)
  - sensor_bias_drift       (IMU calibration drift — generic)
  - noise_spike             (EMI / glitching channel — generic)
  - actuator_stuck          (hard actuator failure — generic)

UnknownAnomaly (fault_type="unknown") is also defined here but is deliberately
excluded from FAULT_CLASSES / FAULT_TYPES / build_fault() — it is only reachable
via build_unknown_fault() and is used solely for out-of-distribution validation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from .config import (
    OBS_COLUMNS, CONTINUOUS_CHANNELS, FAULT_ONSET_EARLY, FAULT_ONSET_LATE,
    MAX_SIMULTANEOUS_FAULTS,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _channel_index(name: str) -> int:
    return OBS_COLUMNS.index(name)


def _random_continuous_channel(rng: np.random.Generator) -> str:
    return CONTINUOUS_CHANNELS[rng.integers(0, len(CONTINUOUS_CHANNELS))]


# Typical nominal episode length with the correct heuristic policy is ~150–300
# steps.  Onset must fire within a realistic episode window, not relative to
# the hard MAX_STEPS cap (1000), which would place onset well past episode end.
_TYPICAL_EPISODE_LEN = 250

def _random_onset(rng: np.random.Generator, max_steps: int) -> int:
    ref = min(max_steps, _TYPICAL_EPISODE_LEN)
    lo = int(ref * FAULT_ONSET_EARLY)
    hi = int(ref * FAULT_ONSET_LATE)
    return int(rng.integers(lo, hi))


# ── base class ───────────────────────────────────────────────────────────────

class BaseFault:
    """Abstract base for all fault injectors."""

    fault_type: str = "base"

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def inject(
        self,
        step: int,
        action: int,
        obs: np.ndarray,
    ) -> tuple[int, np.ndarray, bool]:
        """
        Returns (action, obs, fault_active).
        Subclasses must override _apply(); this wrapper handles the
        pre/post-onset guard and obs copy.
        """
        if step < self.params["onset"]:
            return action, obs, False
        obs = obs.copy()
        action, obs = self._apply(step, action, obs)
        return action, obs, True

    def _apply(
        self, step: int, action: int, obs: np.ndarray
    ) -> tuple[int, np.ndarray]:
        raise NotImplementedError


# ── 1. Sensor Dropout ────────────────────────────────────────────────────────

class SensorDropout(BaseFault):
    """
    inject_sensor_dropout — Luna-25 grounded.

    After onset, freeze a chosen sensor reading at its last known value,
    simulating a telemetry channel going completely silent.  The reading
    stops updating while the physical state continues to evolve, creating
    a growing divergence between reported and true values.
    """

    fault_type = "sensor_dropout"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        channel = _random_continuous_channel(rng)
        onset   = _random_onset(rng, max_steps)
        self.channel = channel
        self.ch_idx  = _channel_index(channel)
        self._frozen_value: Optional[float] = None
        self.params = {"onset": onset, "channel": channel}

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        # Latch the value on the very first active step
        if self._frozen_value is None:
            self._frozen_value = float(obs[self.ch_idx])
        obs[self.ch_idx] = self._frozen_value
        return action, obs


# ── 2. Thrust Degradation ────────────────────────────────────────────────────

class ThrustDegradation(BaseFault):
    """
    inject_thrust_degradation — Chandrayaan-2 grounded.

    After onset, the main engine's effective authority is scaled down by a
    random factor, simulating reduced actuator output during a braking
    maneuver.  Implemented by stochastically suppressing the main-engine
    action command proportional to the degradation severity.
    """

    fault_type = "thrust_degradation"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        onset     = _random_onset(rng, max_steps)
        # degradation: fraction of thrust *lost* (0.3–0.8)
        severity  = float(rng.uniform(0.3, 0.8))
        self._rng = rng
        self.params = {"onset": onset, "severity": round(severity, 4)}

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        # Suppress the main engine command with probability = severity
        if action == 2 and self._rng.random() < self.params["severity"]:
            action = 0
        return action, obs


# ── 3. Filter Misclassification ──────────────────────────────────────────────

class FilterMisclassification(BaseFault):
    """
    inject_filter_misclassification — HAKUTO-R M1 grounded.

    When a genuinely large, correct jump occurs in a position channel,
    discard it and replace it with the previous reading, simulating an
    onboard filter that wrongly classifies valid sensor data as an outlier
    and rejects it.  (HAKUTO-R's altimeter readings were discarded this way.)
    """

    fault_type = "filter_misclassification"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        # Only acts on position channels where jumps are physically meaningful
        channel = rng.choice(["x_pos", "y_pos"])
        onset   = _random_onset(rng, max_steps)
        # Threshold: reject jumps larger than this fraction of the obs range
        threshold = float(rng.uniform(0.03, 0.08))
        self.channel   = channel
        self.ch_idx    = _channel_index(channel)
        self._prev: Optional[float] = None
        self.params = {
            "onset": onset,
            "channel": channel,
            "rejection_threshold": round(threshold, 4),
        }

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        current = float(obs[self.ch_idx])
        if self._prev is None:
            self._prev = current
            return action, obs
        delta = abs(current - self._prev)
        # If the jump exceeds the threshold, reject and report last known value
        if delta > self.params["rejection_threshold"]:
            obs[self.ch_idx] = self._prev
        else:
            self._prev = current
        return action, obs


# ── 4. Measurement Delay ─────────────────────────────────────────────────────

class MeasurementDelay(BaseFault):
    """
    inject_measurement_delay — Resilience grounded.

    After onset, report each new reading for a chosen sensor with a fixed
    N-step delay, simulating a lagged telemetry pipeline or a software bug
    that causes the GNC to act on stale data.  (Resilience's surface-relative
    navigation suffered from delayed altitude updates.)
    """

    fault_type = "measurement_delay"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        channel    = _random_continuous_channel(rng)
        onset      = _random_onset(rng, max_steps)
        delay_steps = int(rng.integers(5, 25))   # 5–24 step delay
        self.channel    = channel
        self.ch_idx     = _channel_index(channel)
        self._buffer: deque[float] = deque()
        self._delay = delay_steps
        self.params = {"onset": onset, "channel": channel, "delay_steps": delay_steps}

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        current = float(obs[self.ch_idx])
        self._buffer.append(current)
        if len(self._buffer) > self._delay:
            delayed = self._buffer.popleft()
            obs[self.ch_idx] = delayed
        else:
            # Buffer not yet full — report the oldest buffered value
            obs[self.ch_idx] = self._buffer[0]
        return action, obs


# ── 5. Sensor Bias Drift ─────────────────────────────────────────────────────

class SensorBiasDrift(BaseFault):
    """
    inject_sensor_bias_drift — IMU calibration drift (generic).

    After onset, add a slowly growing offset to a chosen sensor reading,
    simulating gradual IMU-style calibration drift rather than a sudden
    failure.  The offset grows linearly at a randomised rate so that the
    fault is undetectable at onset but becomes obvious over time.
    """

    fault_type = "sensor_bias_drift"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        channel    = _random_continuous_channel(rng)
        onset      = _random_onset(rng, max_steps)
        # drift_rate: obs-units added per step after onset
        drift_rate = float(rng.uniform(0.0005, 0.003))
        direction  = rng.choice([-1, 1])
        self.channel    = channel
        self.ch_idx     = _channel_index(channel)
        self._rate      = drift_rate * direction
        self.params = {
            "onset": onset,
            "channel": channel,
            "drift_rate": round(float(self._rate), 6),
        }

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        steps_since = step - self.params["onset"]
        obs[self.ch_idx] += self._rate * steps_since
        return action, obs


# ── 6. Noise Spike ───────────────────────────────────────────────────────────

class NoiseSpike(BaseFault):
    """
    inject_noise_spike — electromagnetic interference / glitching channel
    (generic).

    At randomised isolated timesteps after onset, add large transient outlier
    noise to a chosen sensor reading.  The spike is not sustained — it appears
    for exactly one step — simulating EMI bursts or a bit-flip in the ADC.
    """

    fault_type = "noise_spike"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        channel     = _random_continuous_channel(rng)
        onset       = _random_onset(rng, max_steps)
        spike_mag   = float(rng.uniform(0.3, 1.2))   # obs-unit magnitude
        spike_prob  = float(rng.uniform(0.03, 0.12)) # probability per step
        self.channel    = channel
        self.ch_idx     = _channel_index(channel)
        self._rng       = rng
        self._mag       = spike_mag
        self._prob      = spike_prob
        self.params = {
            "onset": onset,
            "channel": channel,
            "spike_magnitude": round(spike_mag, 4),
            "spike_probability": round(spike_prob, 4),
        }

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        if self._rng.random() < self._prob:
            sign = self._rng.choice([-1, 1])
            obs[self.ch_idx] += sign * self._mag
        return action, obs


# ── 7. Actuator Stuck ────────────────────────────────────────────────────────

class ActuatorStuck(BaseFault):
    """
    inject_actuator_stuck — hard actuator failure (generic).

    After onset, force one thruster's output to a fixed stuck value —
    either fully on (action=2, main engine always firing) or fully off
    (action=0, engine cut) — regardless of the commanded action.  This
    is a hard failure mode, distinct from the gradual thrust_degradation.
    """

    fault_type = "actuator_stuck"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        onset      = _random_onset(rng, max_steps)
        stuck_val  = int(rng.choice([0, 2]))   # 0=off, 2=main engine on
        self._stuck = stuck_val
        self.params = {
            "onset": onset,
            "stuck_action": stuck_val,
            "description": "engine_always_on" if stuck_val == 2 else "engine_cut",
        }

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        return self._stuck, obs


# ── 8. Unknown Anomaly (out-of-distribution; NOT in registry) ────────────────

class UnknownAnomaly(BaseFault):
    """
    UnknownAnomaly — Ornstein-Uhlenbeck modulated sensor corruption.

    Deliberately excluded from FAULT_CLASSES and FAULT_TYPES so it never
    appears in the default dataset or multi-fault pool.  Its sole purpose is
    out-of-distribution validation: does the anomaly detector flag signals it
    was never explicitly trained on?

    Mechanism (Option B — continuously drifting corruption magnitude):
    After onset, a randomly chosen continuous channel receives an additive
    offset each step whose magnitude is driven by a slow Ornstein-Uhlenbeck
    (mean-reverting) random walk:

        ou[t] = ou[t-1] + θ × (0 - ou[t-1]) × dt + σ × sqrt(dt) × N(0,1)
        obs[ch] += ou[t]

    Parameters sampled at construction:
      θ (theta)  : mean-reversion rate  — uniform(0.05, 0.30)
      σ (sigma)  : volatility           — uniform(0.02, 0.15)
      dt         : step size            — fixed at 1.0

    Why this has no single learnable fixed signature:
      - Not a freeze  : channel is never held at a constant value.
      - Not a lag     : no memory of past true readings; OU state is independent.
      - Not threshold-reject : corruption is unconditional and continuous.
      - Not fixed drift: mean-reversion causes sign changes; no monotone ramp.
      - Not fixed spikes: continuous state with step-to-step autocorrelation;
        no fixed probability or magnitude.
    """

    fault_type = "unknown"

    def __init__(self, rng: np.random.Generator, max_steps: int) -> None:
        super().__init__()
        channel = _random_continuous_channel(rng)
        onset   = _random_onset(rng, max_steps)
        theta   = float(rng.uniform(0.05, 0.30))   # mean-reversion rate
        sigma   = float(rng.uniform(0.02, 0.15))   # volatility
        dt      = 1.0
        self.channel  = channel
        self.ch_idx   = _channel_index(channel)
        self._theta   = theta
        self._sigma   = sigma
        self._dt      = dt
        self._ou      = 0.0          # OU state; zero at onset (undetectable)
        self._rng     = rng
        self.params   = {
            "onset":   onset,
            "channel": channel,
            "theta":   round(theta, 6),
            "sigma":   round(sigma, 6),
            "dt":      dt,
        }

    def _apply(self, step: int, action: int, obs: np.ndarray) -> tuple[int, np.ndarray]:
        # Advance OU process one step
        self._ou += (
            self._theta * (0.0 - self._ou) * self._dt
            + self._sigma * np.sqrt(self._dt) * self._rng.standard_normal()
        )
        obs[self.ch_idx] += self._ou
        return action, obs


# ── Registry ─────────────────────────────────────────────────────────────────

FAULT_CLASSES: list[type[BaseFault]] = [
    SensorDropout,
    ThrustDegradation,
    FilterMisclassification,
    MeasurementDelay,
    SensorBiasDrift,
    NoiseSpike,
    ActuatorStuck,
]

FAULT_TYPES: list[str] = [cls.fault_type for cls in FAULT_CLASSES]


def build_fault(fault_type: str, rng: np.random.Generator, max_steps: int) -> BaseFault:
    """Instantiate a fault by name string."""
    for cls in FAULT_CLASSES:
        if cls.fault_type == fault_type:
            return cls(rng, max_steps)
    raise ValueError(f"Unknown fault type: {fault_type!r}. "
                     f"Valid types: {FAULT_TYPES}")


# ── Multi-fault conflict rules ────────────────────────────────────────────────

# At most one actuator-affecting fault per episode.
# ThrustDegradation + ActuatorStuck must never be combined: ActuatorStuck
# unconditionally overrides action, silently negating ThrustDegradation.
_ACTUATOR_FAULT_TYPES: frozenset[str] = frozenset({
    "thrust_degradation",
    "actuator_stuck",
})

# Sensor-targeting faults must target different channels per episode.
_SENSOR_FAULT_TYPES: frozenset[str] = frozenset({
    "sensor_dropout",
    "filter_misclassification",
    "measurement_delay",
    "sensor_bias_drift",
    "noise_spike",
})


def build_multi_fault(
    rng: np.random.Generator,
    max_steps: int,
) -> list[BaseFault]:
    """
    Sample a valid pair of faults for a multi-fault episode.

    Conflict rules enforced:
      1. No duplicate fault types.
      2. At most one actuator-affecting fault (thrust_degradation / actuator_stuck).
      3. Two sensor faults must target different channels.

    Returns a list of exactly MAX_SIMULTANEOUS_FAULTS (2) BaseFault instances,
    sorted alphabetically by fault_type for canonical combo naming — so the same
    unordered pair always maps to the same directory / label string.

    Resample strategy: fault A is instantiated once.  Only fault B is re-sampled
    (and re-instantiated) on each conflict, preserving fault A's RNG-derived
    parameters and keeping the sequence fully deterministic for a given rng state.
    """
    assert MAX_SIMULTANEOUS_FAULTS == 2, (
        "build_multi_fault is written for exactly 2 faults; "
        "update the function if MAX_SIMULTANEOUS_FAULTS changes."
    )

    # Sample and instantiate fault A once.
    type_a = FAULT_TYPES[rng.integers(0, len(FAULT_TYPES))]
    fault_a = build_fault(type_a, rng, max_steps)

    # Loop until a valid fault B is found.
    while True:
        type_b = FAULT_TYPES[rng.integers(0, len(FAULT_TYPES))]

        # Rule 1: no duplicate types.
        if type_b == type_a:
            continue

        # Rule 2: at most one actuator fault.
        if type_a in _ACTUATOR_FAULT_TYPES and type_b in _ACTUATOR_FAULT_TYPES:
            continue

        # Type pair is valid — instantiate B (consumes RNG).
        fault_b = build_fault(type_b, rng, max_steps)

        # Rule 3: two sensor faults must target different channels.
        if (
            type_a in _SENSOR_FAULT_TYPES
            and type_b in _SENSOR_FAULT_TYPES
            and fault_a.channel == fault_b.channel  # type: ignore[attr-defined]
        ):
            # Discard this B instance and resample type B.
            continue

        break

    # Sort alphabetically so combo name is canonical regardless of draw order.
    pair = sorted([fault_a, fault_b], key=lambda f: f.fault_type)
    return pair


# ── Unknown anomaly factory (separate from main dispatch) ────────────────────

def build_unknown_fault(rng: np.random.Generator, max_steps: int) -> UnknownAnomaly:
    """
    Instantiate an UnknownAnomaly fault.

    Not reachable via build_fault() — intentionally excluded from the registry
    so unknown episodes only enter the dataset through an explicit call here.
    """
    return UnknownAnomaly(rng, max_steps)
