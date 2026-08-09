# Phase 0 — Simulation Loop Fix Plan

## Top-Level Overview

The `simulation.py` episode loop has a structural bug: the scripted policy always
acts on the raw environment observation (`obs`), not the faulted/noisy one. This
means five of the seven sensor-side fault injectors have no effect on trajectory
physics — the fault only ever shows up as a corrupted CSV column, never as a
realistic flight anomaly.

**Goal:** Correct the loop so the policy acts on what the flight computer
*actually reports* (fault-corrupted + sensor-noisy), and add `true_<channel>`
columns to every CSV so ground truth is preserved alongside the reported values.

**Scope:** Only `simulation.py` changes. `faults.py`, `config.py`,
`scenario_generator.py`, `sanity_check.py`, `generate_dataset.py`, and
`run_all.py` are untouched.

**Non-goals:** Multi-fault composition, new fault types, any backend work.

---

## Sub-Tasks

---

### Sub-Task 1 — Fix the episode loop in `simulation.py`

**Intent**
Restructure `run_episode()` so that:
1. The policy acts on `obs_reported` (fault-corrupted + noisy) instead of raw
   ground-truth `obs`.
2. Fault injection continues to receive the true `obs` (unchanged contract with
   all `BaseFault.inject()` implementations).
3. Sensor noise is applied to the fault-corrupted reading before it is fed back
   to the next policy call.
4. Both `true_<channel>` and `<channel>` (reported) columns are written to
   every CSV row for all episode types (nominal and faulted).

**Expected Outcomes**
- Running a sensor-bias-drift or sensor-dropout episode now produces a trajectory
  where the lander visibly reacts to the corrupted reading (erratic burn timing,
  mis-timed descent braking).
- Nominal episodes will differ slightly from previously generated ones: the
  policy now sees baseline sensor noise on every step (fault or not) because
  `_add_sensor_noise` runs unconditionally before the next policy call. This is
  a deliberate realism improvement — real sensors aren't noise-free even during
  nominal flight, which is exactly why `SENSOR_NOISE_STD` exists. Regenerated
  nominal trajectories will therefore not be bit-identical to the old ones.
  The `true_<channel>` columns in nominal CSVs will equal the clean environment
  reading; the plain `<channel>` columns will carry that reading plus baseline
  noise.
- All existing actuator-side fault behavior (thrust_degradation, actuator_stuck)
  is unchanged.
- The DataFrame column order is: `step`, `true_<ch>` × 8, `<ch>` × 8, `action`,
  `reward`, `terminated`, `truncated`, `fault_label`, `fault_active`.

**Todo List**
1. In `run_episode()`, introduce `obs_reported` before the loop by passing the
   initial reset observation through the noise pipeline:
   `obs_reported = _add_sensor_noise(obs.copy(), rng)`. This ensures the very
   first policy call is on a noisy reading, consistent with every subsequent
   step, rather than a perfectly clean one.
2. Inside the loop, change `action = _scripted_action(obs)` to
   `action = _scripted_action(obs_reported)`.
3. Keep `fault.inject(step, action, obs)` receiving the true `obs` — no change
   to the call signature.
4. Rename the second return value to `obs_faulted` for clarity.
   For the `else` (nominal) branch set `obs_faulted = obs`.
5. Apply sensor noise to `obs_faulted` to produce the next step's reported
   observation: `obs_reported = _add_sensor_noise(obs_faulted, rng)`.
6. In the row-building block, use **two separate loops** over `OBS_COLUMNS`:
   first a loop writing `row[f"true_{col}"] = float(obs[i])` for all 8
   channels, then a second loop writing `row[col] = float(obs_reported[i])` for
   all 8 channels. This is required to produce the grouped column order
   (`true_*` × 8 then plain × 8) stated in Expected Outcomes — a single
   interleaved loop would silently produce `true_x_pos, x_pos, true_y_pos,
   y_pos, …` instead.
7. Update the module-level docstring to list the new `true_*` columns.
8. `obs = next_obs` stays at the bottom of the loop — ground truth advances from
   the environment as before.

**Relevant Context**
- Loop to change: [`simulation.py:125-157`](sentinel/data_generation/simulation.py:125)
- Fault contract (must not change): [`BaseFault.inject()`](sentinel/data_generation/faults.py:65)
  always receives true `obs`, returns `(action, corrupted_obs, fault_active)`.
- Noise helper: [`_add_sensor_noise()`](sentinel/data_generation/simulation.py:84) —
  no changes needed; just call it on `obs_faulted` instead of `obs_logged`.
- Column name list: [`OBS_COLUMNS`](sentinel/data_generation/config.py:30) —
  used to generate both the `true_*` keys and the plain keys.
- Downstream readers (`sanity_check.py`, `scenario_generator._safe_landing()`)
  reference columns by name string — they will continue to work because the plain
  channel names are still present; the new `true_*` columns are additive.

**Status:** [x] done

---

### Sub-Task 2 — Update the module docstring and column list comment

**Intent**
The module-level docstring in `simulation.py` lists the returned DataFrame
columns. After Sub-Task 1 the column set doubles in the observation section.
Update it to reflect reality so future readers aren't misled.

**Expected Outcomes**
- The docstring "Returned DataFrame columns" section accurately lists all columns
  in emission order, including the `true_*` prefix columns.

**Todo List**
1. Edit the docstring at [`simulation.py:11-14`](sentinel/data_generation/simulation.py:11)
   to prepend `true_x_pos, true_y_pos, …, true_right_leg` before the plain
   channel names.

**Relevant Context**
- Only the comment block; no logic changes.

**Status:** [x] done

---

## Implementation Notes

- Sub-Task 2 can be done in the same diff as Sub-Task 1 — they touch the same
  file and the plan separates them only for reviewability.
- No dataset regeneration is required *during* planning; the existing data under
  `data/` will be stale after the fix but that is expected and the dataset will
  be regenerated as part of normal workflow.
- The `sanity_check.py` plot loop reads `df[channel]` by name — since it reads
  the *reported* channel name (e.g. `"y_pos"`), not `"true_y_pos"`, it will
  continue to show the corrupted signal, which is the desired view for anomaly
  visualisation.
- **`true_left_leg` / `true_right_leg` are permanently redundant.** No fault
  type ever targets leg-contact channels (`_random_continuous_channel` and
  `FilterMisclassification` both exclude them), and their `SENSOR_NOISE_STD` is
  `0.0`. These two columns will be byte-identical to `left_leg` / `right_leg`
  in every single row. They are kept here for schema symmetry (all 8 obs
  channels get a `true_*` twin), but could be dropped without any information
  loss. Flagged for a future cleanup decision; not worth blocking this fix.
