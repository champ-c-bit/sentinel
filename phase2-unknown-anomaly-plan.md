# Phase 2 — Unknown / Generic Anomaly Class Plan

## Top-Level Overview

Add an `UnknownAnomaly` fault class that is deliberately kept **outside** the
main registry and dataset pipeline — it exists to validate that anomaly detection
generalizes beyond the 7 known fault types.

The work touches four files:

| File | Nature of change |
|---|---|
| `config.py` | Add `N_UNKNOWN_EPISODES`, `UNKNOWN_DIR`, `UNKNOWN_LABELS_CSV` |
| `faults.py` | Add `UnknownAnomaly(BaseFault)` class + `build_unknown_fault()` factory — `FAULT_CLASSES` / `FAULT_TYPES` / `build_fault()` untouched |
| `scenario_generator.py` | Add `generate_unknown_episodes()` — standalone function, not called from `generate_dataset()` |
| `sanity_check.py` | Add `run_unknown_sanity_check()` — separate plot, not folded into 7-row grid |

`simulation.py`, `generate_dataset.py`, `run_all.py` are **not touched**.

**Invariant that must hold:** `generate_dataset()` with any combination of
arguments must produce zero unknown-labeled episodes. `build_fault("unknown", …)`
must still raise `ValueError`.

---

## Sub-Tasks

---

### Sub-Task 1 — Add constants to `config.py`

**Intent**
Introduce the three new values that `scenario_generator.py` and `sanity_check.py`
will import for the unknown batch.

**Expected Outcomes**
- `config.py` exports:
  - `N_UNKNOWN_EPISODES = 50` — default batch size (opt-in, like multi-fault).
  - `UNKNOWN_DIR` pointing to `data/unknown/`.
  - `UNKNOWN_LABELS_CSV` pointing to `data/unknown_labels.csv`.

**Todo List**
1. After the `N_MULTI_FAULT_EPISODES` line add:
   ```python
   N_UNKNOWN_EPISODES = 50
   ```
2. After the `LABELS_CSV` line add:
   ```python
   UNKNOWN_DIR        = os.path.join(DATA_ROOT, "unknown")
   UNKNOWN_LABELS_CSV = os.path.join(DATA_ROOT, "unknown_labels.csv")
   ```

**Relevant Context**
- [`config.py:25`](sentinel/data_generation/config.py:25) — dataset sizes block
- [`config.py:65`](sentinel/data_generation/config.py:65) — output layout block

**Status:** [x] done

---

### Sub-Task 2 — Add `UnknownAnomaly` and `build_unknown_fault()` to `faults.py`

**Intent**
Implement the `UnknownAnomaly` fault class using Option B (modulated noise via
slow random walk) and expose it through a dedicated factory. The class must not
appear in `FAULT_CLASSES`, `FAULT_TYPES`, or be reachable via `build_fault()`.

**Corruption mechanism — Option B (Ornstein-Uhlenbeck modulated noise):**
After onset, a single randomly chosen continuous channel receives additive
corruption each step. The corruption magnitude is driven by a slow random walk:

```
corruption[t] = corruption[t-1] + θ × (0 - corruption[t-1]) × dt + σ × sqrt(dt) × N(0,1)
```

where `θ` (mean-reversion rate), `σ` (volatility), and `dt` are sampled at
construction time. This produces a signal that drifts, reverses, amplifies, and
subsides — never settling into a fixed bias (unlike `SensorBiasDrift`) or a
fixed spike probability (unlike `NoiseSpike`). The corruption is always
"present" after onset but continuously changes character.

**Expected Outcomes**
- `UnknownAnomaly.fault_type == "unknown"`.
- After onset, `obs[ch_idx]` is perturbed by the current OU random-walk value
  each step.
- `self.params` contains: `onset`, `channel`, `theta`, `sigma`, `dt` — enough
  to reproduce the episode given the same RNG seed.
- `FAULT_CLASSES` and `FAULT_TYPES` are unchanged (7 entries each).
- `build_fault("unknown", rng, max_steps)` still raises `ValueError`.
- `build_unknown_fault(rng, max_steps)` returns an `UnknownAnomaly` instance.

**Todo List**
1. Add `UnknownAnomaly(BaseFault)` below the `ActuatorStuck` class (before the
   `# ── Registry` block):
   - `fault_type = "unknown"`.
   - `__init__`: sample `channel` via `_random_continuous_channel(rng)`, sample
     `onset` via `_random_onset(rng, max_steps)`, sample `theta` from
     `rng.uniform(0.05, 0.3)` (slow mean-reversion), `sigma` from
     `rng.uniform(0.02, 0.15)` (volatility), `dt = 1.0`. Store
     `self._ou_state = 0.0`. Record `params`.
   - `_apply`: advance the OU state:
     `self._ou_state += theta * (0 - self._ou_state) * dt + sigma * sqrt(dt) * rng.standard_normal()`,
     then `obs[ch_idx] += self._ou_state`.
   - Store the per-step RNG call inside `_apply` (the fault's `__init__` RNG is
     already consumed — use a separate `self._rng` stored at construction,
     identical pattern to `ThrustDegradation` and `NoiseSpike`).
2. After `build_multi_fault()`, add:
   ```python
   def build_unknown_fault(rng: np.random.Generator, max_steps: int) -> "UnknownAnomaly":
       """Instantiate an UnknownAnomaly fault. Not reachable via build_fault()."""
       return UnknownAnomaly(rng, max_steps)
   ```
3. Update the module docstring to note that `UnknownAnomaly` exists but is not
   in the registry.

**Relevant Context**
- `ActuatorStuck` ends at [`faults.py:333`](sentinel/data_generation/faults.py:333).
- Registry block starts at [`faults.py:336`](sentinel/data_generation/faults.py:336).
- `ThrustDegradation` (stores `self._rng`) and `NoiseSpike` (stores `self._rng`)
  are the pattern to follow for per-step RNG use.
- `_random_continuous_channel`, `_random_onset`, `_channel_index` are all
  available as module-level helpers.
- `math.sqrt` or `np.sqrt` — prefer `np.sqrt(dt)` since `dt` is a float.

**Status:** [x] done

---

### Sub-Task 3 — Add `generate_unknown_episodes()` to `scenario_generator.py`

**Intent**
Provide a standalone batch-generation function for unknown episodes that writes
to its own directory and labels file, and continues the global seed counter from
wherever `generate_dataset()` left off — without calling into it.

**Seed-collision safety:** the function must derive `seed_offset` from the
*actual* seeds already recorded on disk, not from config constants. Config
constants cannot track what was really generated (e.g. `--multi-fault-episodes`
defaults to 0, but config has `N_MULTI_FAULT_EPISODES = 100`; a caller who ran
a 500-episode multi-fault batch would get collisions from a hardcoded formula).
The function reads `labels.csv` and `unknown_labels.csv` (if they exist) to find
the highest seed already used, then asserts that all intended unknown seeds are
strictly above that watermark.

**Expected Outcomes**
- `generate_unknown_episodes(n_unknown, seed_offset)` creates:
  - `data/unknown/unknown_0000.csv … unknown_<N-1>.csv`
  - `data/unknown_labels.csv` with columns: `filename`, `fault_type`, `seed`,
    `n_steps`, `fault_params` (same Option C JSON schema).
- `fault_type` in labels is `"unknown"` for every row.
- `fault_label` in the episode CSV is `"unknown"` (set by `run_episode` from the
  fault's `fault_type`).
- `seed_offset` has **no default** — the caller must pass an explicit value.
  This makes the "which seed to start from" decision explicit and auditable,
  rather than silently computing it from constants that may not reflect reality.
- The function reads `LABELS_CSV` and `UNKNOWN_LABELS_CSV` (if they exist),
  collects all seeds already present, and raises `ValueError` before generating
  any episode if `seed_offset` through `seed_offset + n_unknown - 1` would
  collide with any of them.
- `generate_dataset()` is completely unchanged and does not call this function.

**Todo List**
1. Import `N_UNKNOWN_EPISODES`, `UNKNOWN_DIR`, `UNKNOWN_LABELS_CSV` from `config`.
2. Import `build_unknown_fault` from `faults`.
3. Add a helper `_collect_used_seeds() -> set[int]` above `generate_unknown_episodes()`:
   - Read `LABELS_CSV` if it exists; collect the `seed` column.
   - Read `UNKNOWN_LABELS_CSV` if it exists; collect the `seed` column.
   - Also count nominal episodes on disk: `seeds = {RANDOM_SEED_BASE + i for i in range(len(nominal_files))}`.
   - Return the union as a `set[int]`.
4. Add `generate_unknown_episodes(n_unknown, seed_offset)` below `generate_dataset()`:
   - No default for `seed_offset` — caller must supply it explicitly.
   - Call `_collect_used_seeds()` to get the set of all seeds already on disk.
   - Compute `intended = {seed_offset + i for i in range(n_unknown)}`.
   - If `intended & used_seeds` is non-empty, raise `ValueError` with the
     colliding seeds listed — fail before touching any file.
   - `os.makedirs(UNKNOWN_DIR, exist_ok=True)`.
   - Loop `i` in `range(n_unknown)`:
     - `seed = seed_offset + i`
     - `rng = np.random.default_rng(seed)`
     - `fault = build_unknown_fault(rng, MAX_STEPS)`
     - `df = run_episode(seed=seed, faults=[fault])`
     - Write CSV to `UNKNOWN_DIR/unknown_{i:04d}.csv`.
     - Label row: `{"filename": f"unknown_{i:04d}.csv", "fault_type": "unknown",
       "seed": seed, "n_steps": len(df), "fault_params": json.dumps([fault.params])}`.
   - Write `unknown_labels.csv`.
   - Print a short summary.
   - Return `UNKNOWN_LABELS_CSV`.
5. Update the module docstring to describe the new output directory.

**Relevant Context**
- Single-fault loop in [`scenario_generator.py:97-123`](sentinel/data_generation/scenario_generator.py:97) —
  follow the same seed/rng/run_episode pattern.
- `json` and `os` already imported; `tqdm` already imported.
- `LABELS_CSV` and `NOMINAL_DIR` are already imported — reuse for `_collect_used_seeds()`.
- `FAULT_TYPES` is already imported.

**Status:** [x] done

---

### Sub-Task 4 — Add `run_unknown_sanity_check()` to `sanity_check.py`

**Intent**
Add a separate spot-check function that plots a few unknown-episode trajectories
so a human can visually confirm the corruption looks anomalous but distinct from
the 7 existing fault types. This must not touch `run_sanity_check()`'s 7-row
grid or the episode summary block.

**Expected Outcomes**
- `run_unknown_sanity_check(n_examples=3)` produces `data/unknown_sanity_check.png`:
  a single row of `n_examples` subplots, each showing `y_pos` (altitude) and the
  corrupted channel on a twin axis, with a vertical onset marker.
- If `UNKNOWN_DIR` does not exist or is empty, the function prints a message and
  returns without error.
- `run_sanity_check()` is completely unchanged.
- The `__main__` block at the bottom of `sanity_check.py` calls both
  `run_sanity_check()` and `run_unknown_sanity_check()` so a single
  `python data_generation/sanity_check.py` invocation produces both plots.

**Todo List**
1. Import `UNKNOWN_DIR`, `UNKNOWN_LABELS_CSV` from `config`.
2. Add `run_unknown_sanity_check(n_examples: int = 3)` after `run_sanity_check()`:
   - Early-exit if `UNKNOWN_DIR` doesn't exist or contains no CSVs.
   - Load `UNKNOWN_LABELS_CSV` if it exists (for onset lookup), else use an
     empty DataFrame.
   - Randomly sample up to `n_examples` CSV files from `UNKNOWN_DIR`.
   - For each, create a subplot: plot `y_pos` in blue, overlay the corrupted
     channel (read from `fault_params[0]["channel"]` in labels) on a twin y-axis
     in orange, place a vertical onset marker.
   - Save to `data/unknown_sanity_check.png`.
3. Update the `__main__` block at the end of the file:
   ```python
   if __name__ == "__main__":
       run_sanity_check()
       run_unknown_sanity_check()
   ```
4. Update the module docstring to describe the new plot.

**Relevant Context**
- `run_sanity_check()`: [`sanity_check.py:100`](sentinel/data_generation/sanity_check.py:100)
  — follow the same matplotlib pattern (headless Agg backend already set).
- `_onset_step()` is not reusable here because it looks up `LABELS_CSV`, not
  `UNKNOWN_LABELS_CSV`. Read onset directly from `fault_params` JSON in the
  local labels DataFrame.
- `__main__` block: [`sanity_check.py:211`](sentinel/data_generation/sanity_check.py:211).

**Status:** [x] done

---

## Implementation Order

Sub-Tasks 1 → 2 → 3 → 4

- **1 first**: constants needed by all other sub-tasks.
- **2 before 3**: `build_unknown_fault` imported by `scenario_generator.py`.
- **3 before 4**: `run_unknown_sanity_check` reads files written by Sub-Task 3.
- **4 last**: purely additive to `sanity_check.py`.

## Validation Checklist (Definition of Done)

- [ ] `generate_dataset()` (any args) produces zero `unknown`-labeled rows in `labels.csv`.
- [ ] `build_fault("unknown", rng, MAX_STEPS)` raises `ValueError`.
- [ ] `FAULT_CLASSES` still has exactly 7 entries; `FAULT_TYPES` still has exactly 7 strings.
- [ ] `generate_unknown_episodes(n_unknown=5, seed_offset=S)` runs without error; 5 CSVs appear in `data/unknown/`; `unknown_labels.csv` has 5 rows with `fault_type == "unknown"` and valid `fault_params` JSON.
- [ ] **Seed-collision guard fires:** calling `generate_unknown_episodes` with a `seed_offset` that overlaps any seed in `labels.csv` raises `ValueError` before generating any file.
- [ ] **Seed uniqueness enforced on disk:** after a successful run, the seeds in `unknown_labels.csv` have zero intersection with the seeds in `labels.csv` — verified programmatically, not by inspection.
- [ ] Unknown episode CSVs have `fault_label == "unknown"` and `fault_active == 1` after onset.
- [ ] OU random-walk corruption is visible: `y_pos` diverges from `true_y_pos` after onset in a non-monotone, non-periodic way.
- [ ] **Visual distinctness (manual):** open `unknown_sanity_check.png` alongside `sanity_check.png` and confirm the unknown trajectories are not visually identical to any single fault type's characteristic pattern (monotone ramp, staircase freeze, fixed-lag stale value, etc.).
- [ ] `run_unknown_sanity_check()` produces `data/unknown_sanity_check.png` without error.
- [ ] `run_unknown_sanity_check()` exits cleanly (no error, short message) when `data/unknown/` does not exist.
- [ ] `python data_generation/sanity_check.py` produces both `sanity_check.png` and `unknown_sanity_check.png`.

## Implementation Notes

- **`seed_offset` is required, not defaulted:** the formula
  `RANDOM_SEED_BASE + N_NOMINAL + N_FAULT_EACH * 7 + N_MULTI_FAULT_EPISODES`
  does not track what was actually generated — `--multi-fault-episodes` defaults
  to 0 at the CLI, so the multi-fault term in the formula is almost never the
  right number. Making `seed_offset` a required argument forces the caller to
  compute it from reality (e.g. `max(pd.read_csv(LABELS_CSV)["seed"]) + 1`),
  and the in-function collision check then provides a hard safety net regardless.
- **OU state at onset:** `self._ou_state` is initialised to `0.0` (zero
  corruption at onset), so the fault is undetectable at the exact onset step and
  builds gradually — consistent with `SensorBiasDrift`'s design intent but with
  a non-monotone trajectory.
- **Why this doesn't reduce to `SensorBiasDrift`:** `SensorBiasDrift` adds
  `rate × steps_since_onset` — a perfectly linear, monotone ramp with fixed
  sign. The OU process has mean-reversion, so it wanders both above and below
  zero, and its instantaneous slope changes every step. A classifier that learned
  "monotone growing offset" would not recognize OU-modulated noise.
- **Why this doesn't reduce to `NoiseSpike`:** `NoiseSpike` fires at a fixed
  probability per step with a fixed magnitude. The OU process has continuous
  state that persists across steps — the corruption on step N depends on step N-1,
  creating temporal autocorrelation that a per-step spike model cannot capture.
- **Why this doesn't reduce to `SensorDropout` (freeze):** `SensorDropout` sets
  `obs[ch] = frozen_value` — a constant. The OU process adds a *changing* offset
  each step; the channel is never held at a fixed value.
- **Why this doesn't reduce to `MeasurementDelay` (fixed-lag buffer):** a lag
  buffer reports a past value with fixed offset in time. The OU process has no
  memory of previous *true* values — it adds a state that is independent of the
  observation history.
- **Why this doesn't reduce to `FilterMisclassification` (threshold-and-reject):**
  that fault conditionally replaces the reading with a stale value only when a
  jump threshold is exceeded. The OU process adds corruption unconditionally
  after onset, and its magnitude is continuous and unbounded rather than binary.
- **`UnknownAnomaly` and `build_multi_fault()`:** the prompt specifies keeping
  `UnknownAnomaly` out of the multi-fault pool. `build_multi_fault()` already
  only samples from `FAULT_TYPES` (the 7-entry list), so no change is needed
  there.
