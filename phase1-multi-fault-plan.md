# Phase 1 — Multi-Fault Episodes Plan

## Top-Level Overview

Extend the data-generation pipeline to produce episodes where **two faults fire
simultaneously, independently timed**, without changing any existing single-fault
or nominal behavior.

The work touches six files:

| File | Nature of change |
|---|---|
| `config.py` | Add two new constants + `MULTI_FAULTED_DIR` path |
| `faults.py` | Add `build_multi_fault()` factory function + two conflict-rule constants — no existing fault class or registry entry is modified |
| `simulation.py` | `fault:` → `faults:` param; chain injection loop |
| `scenario_generator.py` | Multi-fault batch generation; new labels schema (Option C) |
| `sanity_check.py` | Update `_affected_channel` / `_onset_step` to read JSON `fault_params` |
| `generate_dataset.py` + `run_all.py` | Add `--multi-fault-episodes N` CLI flag |

**Invariant that must hold:** the CLI default for `--multi-fault-episodes` is `0`,
so an unmodified invocation of `generate_dataset.py` or `run_all.py` produces
byte-identical output to the pre-Phase-1 codebase for all nominal and single-fault
episodes. Passing a non-zero value opts into multi-fault generation explicitly.

---

## Sub-Tasks

---

### Sub-Task 1 — Add constants to `config.py`

**Intent**
Introduce the two new tuneable values the batch builder will import.

**Expected Outcomes**
- `config.py` exports `MAX_SIMULTANEOUS_FAULTS = 2` and
  `N_MULTI_FAULT_EPISODES = 100`.
- Add a new `MULTI_FAULTED_DIR` path constant pointing to
  `data/faulted_multi/` alongside the existing `FAULTED_DIR`.

**Todo List**
1. After the `N_FAULT_EPISODES_EACH` line add:
   ```
   MAX_SIMULTANEOUS_FAULTS = 2
   N_MULTI_FAULT_EPISODES  = 100
   ```
2. After the `FAULTED_DIR` line add:
   ```
   MULTI_FAULTED_DIR = os.path.join(DATA_ROOT, "faulted_multi")
   ```

**Relevant Context**
- [`config.py:22-23`](sentinel/data_generation/config.py:22) — dataset sizes block
- [`config.py:61`](sentinel/data_generation/config.py:61) — `FAULTED_DIR` definition

**Status:** [x] done

---

### Sub-Task 2 — Generalise `run_episode()` in `simulation.py`

**Intent**
Change the `fault` parameter to `faults` (a list) and thread the injection chain
correctly. Each fault in the list calls `.inject()` in order; the output
`(action, obs)` of fault N becomes the input to fault N+1. `fault_active` is
True if *any* fault in the list is active that step.

Single-fault callers pass `faults=[f]`; nominal callers pass `faults=None` or
`faults=[]`. Both must produce byte-identical output to the current code
(same RNG state, same env steps, same row values).

**Expected Outcomes**
- `run_episode(seed=s, faults=None)` → same DataFrame as before (nominal).
- `run_episode(seed=s, faults=[f])` → same DataFrame as `run_episode(seed=s, fault=f)` did before.
- `run_episode(seed=s, faults=[f1, f2])` → both faults' corruptions visible in
  the reported channel columns; `fault_active=1` from the earlier onset onward.
- `fault_label` column: for a list it becomes the `+`-joined fault types,
  e.g. `"sensor_dropout+noise_spike"`. Single-fault and nominal labels unchanged.
- Module docstring updated to reflect the new parameter name.

**Todo List**
1. Change the signature from `fault: Optional[BaseFault] = None` to
   `faults: Optional[list[BaseFault]] = None`.
2. Replace `fault_label = fault.fault_type if fault is not None else "nominal"`
   with logic that joins the list: `"+".join(f.fault_type for f in faults)` or
   `"nominal"` when the list is empty/None.
3. Replace the per-step fault block (the `if fault is not None: … else: …` block)
   with a chain loop:
   - Start with `action_out, obs_faulted = action, obs` (true obs, unmodified
     action as baseline).
   - For each fault in the (non-empty) list call
     `action_out, obs_faulted, active = fault.inject(step, action_out, obs_faulted)`.
     Accumulate `any_active = any_active or active`.
   - When the list is empty/None, `obs_faulted = obs`, `any_active = False`.
4. Continue using `obs_faulted` as the input to `_add_sensor_noise` (unchanged).
5. Record `fault_active = int(any_active)` (unchanged column name).
6. Update the module docstring's parameter description for `faults`.

**Relevant Context**
- Current injection block: [`simulation.py:141-146`](sentinel/data_generation/simulation.py:141)
- `fault_label` assignment: [`simulation.py:135`](sentinel/data_generation/simulation.py:135)
- `BaseFault.inject()` contract: receives `(step, action, obs)`, returns
  `(action, obs, fault_active)` — **unchanged**, no edits to any existing code in `faults.py`.
- The chain must pass `obs_faulted` (not true `obs`) from one fault to the next.
  This is the correct data-flow for faults on *different* channels: each fault
  sees the observation as already modified by any prior fault in the chain.

**Status:** [x] done

---

### Sub-Task 3 — Add `build_multi_fault()` to `faults.py`

**Intent**
Provide a single function that samples a valid 2-fault combination respecting
the conflict rules, used by the batch generator. Keeping this logic in `faults.py`
co-locates it with the fault classes it reasons about.

**Expected Outcomes**
- `build_multi_fault(rng, max_steps)` returns a `list[BaseFault]` of exactly
  `MAX_SIMULTANEOUS_FAULTS` (2) faults that satisfy all three conflict rules.
- The function is deterministic given the same `rng` state.
- `faults.py` exports `build_multi_fault`.

**Todo List**
1. Import `MAX_SIMULTANEOUS_FAULTS` from `.config` at the top of `faults.py`
   (alongside the existing `OBS_COLUMNS` / `CONTINUOUS_CHANNELS` import).
2. Define the two conflict rule sets as module-level constants inside `faults.py`:
   ```python
   # Faults that affect actuator (action) — at most one per episode
   _ACTUATOR_FAULT_TYPES = {"thrust_degradation", "actuator_stuck"}

   # Faults that target a specific sensor channel — must be on different channels
   _SENSOR_FAULT_TYPES = {
       "sensor_dropout", "filter_misclassification", "measurement_delay",
       "sensor_bias_drift", "noise_spike",
   }
   ```
3. Implement `build_multi_fault(rng, max_steps)`:
   - Assert `MAX_SIMULTANEOUS_FAULTS == 2` at entry — cheap guard that surfaces
     immediately if the constant is changed without updating this function.
   - Sample fault type A uniformly from `FAULT_TYPES`.
   - Instantiate fault A (consuming RNG).
   - Loop to find fault B:
     - Sample type B uniformly from `FAULT_TYPES`.
     - Reject (resample type B only) if: B == A, or both A and B are in
       `_ACTUATOR_FAULT_TYPES`.
     - If the type pair is valid, instantiate fault B (consuming RNG).
     - If both are sensor faults and share the same `.channel`, discard the
       B instance and resample type B again (re-instantiate on each attempt).
   - Sort the two faults alphabetically by `fault_type` so that the same
     unordered pair always produces the same canonical ordering regardless of
     which type was drawn as A vs B. This fixes directory/label fragmentation —
     `noise_spike+sensor_dropout` and `sensor_dropout+noise_spike` can never
     appear as separate combos.
   - Return `[fault_sorted_first, fault_sorted_second]`.
4. Note on resample strategy: fault A is instantiated exactly once before the
   loop. Only fault B is re-instantiated on each resample attempt. This is
   required because instantiation consumes RNG state; discarding a constructed
   fault wastes RNG draws but keeps A's parameters stable and the sequence
   fully deterministic given the same `rng` input.

**Relevant Context**
- [`faults.py:333-354`](sentinel/data_generation/faults.py:333) — registry and
  `build_fault()` — `build_multi_fault()` lives alongside these.
- The `channel` attribute is set on `self.channel` by `SensorDropout`,
  `FilterMisclassification`, `MeasurementDelay`, `SensorBiasDrift`, `NoiseSpike`.
  `ThrustDegradation` and `ActuatorStuck` do not have `.channel`.

**Status:** [x] done

---

### Sub-Task 4 — Migrate labels schema to Option C in `scenario_generator.py`

**Intent**
Replace the flat `param_*` column approach with a single `fault_params` JSON list
column on all rows (single-fault and multi-fault alike). This is the agreed
Option C schema.

**Expected Outcomes**
- Every row in `labels.csv` has a `fault_params` column containing a JSON string
  like `"[{\"onset\": 37, \"channel\": \"x_vel\"}]"` (one-element list for
  single-fault, two-element for multi-fault).
- No `param_*` flat columns appear anywhere in `labels.csv`.
- `fault_type` for multi-fault rows is the `+`-joined string (matching the CSV
  `fault_label` column), e.g. `"sensor_dropout+noise_spike"`.
- `generate_dataset()` gains an `n_multi: int = 0` parameter (wired to CLI in
  Sub-Task 5).
- Multi-fault CSVs are written to
  `data/faulted_multi/<typeA>+<typeB>/multi_<typeA>+<typeB>_<NNNN>.csv`.
- The global seed counter continues from where single-fault episodes left off
  (no restart).
- Dataset summary print block extended to show multi-fault episode count.
- Module docstring updated to describe the new output layout and labels schema.

**Todo List**
1. Import `MULTI_FAULTED_DIR` and `N_MULTI_FAULT_EPISODES` from `config`.
2. Import `build_multi_fault` from `faults`.
3. Change the single-fault label row builder from
   `row.update({f"param_{k}": v for k, v in fault.params.items()})` to
   `row["fault_params"] = json.dumps([fault.params])`.
4. Add `n_multi: int = N_MULTI_FAULT_EPISODES` parameter to `generate_dataset()`.
5. After the single-fault loop, add a `[3/3]` multi-fault generation block:
   - `os.makedirs` for each `<typeA>+<typeB>` combo as encountered (not
     pre-created, because combos are random).
   - Seed continues by incrementing the **same `episode_counter` variable** that
     the single-fault loop left at its final value — do not re-derive it as
     `n_fault_each * len(FAULT_TYPES)`. The two expressions are equivalent today
     but the counter is already in scope and will stay correct if the single-fault
     loop shape ever changes.
     ```python
     seed = RANDOM_SEED_BASE + n_nominal + episode_counter
     episode_counter += 1
     ```
   - `rng = np.random.default_rng(seed)` — same pattern as single-fault.
   - Call `build_multi_fault(rng, MAX_STEPS)` to get `[fault_a, fault_b]`
     (alphabetically sorted by `fault_type` — canonical ordering guaranteed by
     Sub-Task 3).
   - Call `run_episode(seed=seed, faults=[fault_a, fault_b])`.
   - Build the label row using the sorted fault order:
     ```python
     combo = fault_a.fault_type + "+" + fault_b.fault_type
     fname = f"multi_{combo}_{i:04d}.csv"
     row = {
         "filename":     os.path.join(combo, fname),
         "fault_type":   combo,
         "seed":         seed,
         "n_steps":      len(df),
         "fault_params": json.dumps([fault_a.params, fault_b.params]),
     }
     ```
   - Write the CSV to `MULTI_FAULTED_DIR / combo / fname`.
6. Update `generate_dataset()`'s `[2/2]` → `[2/3]` and print block accordingly.
7. Update module docstring.

**Relevant Context**
- Current label row builder: [`scenario_generator.py:93-101`](sentinel/data_generation/scenario_generator.py:93)
- `json` is already imported at [`scenario_generator.py:20`](sentinel/data_generation/scenario_generator.py:20)
- `run_episode` call: [`scenario_generator.py:88`](sentinel/data_generation/scenario_generator.py:88) —
  must become `run_episode(seed=seed, faults=[fault])` after Sub-Task 2.

**Status:** [x] done

---

### Sub-Task 5 — Update `sanity_check.py` for Option C schema

**Intent**
`_affected_channel()` and `_onset_step()` currently read flat `param_channel` /
`param_onset` columns. Both must be updated to parse the `fault_params` JSON list.

**Multi-fault rows must not silently return partial data.** The master prompt
explicitly required these functions not to "silently return a default or wrong
value for multi-fault rows." Since multi-fault episodes are excluded from the
plot grid this phase, the correct behaviour is to return a skip sentinel
(`None` / `-1`) when the JSON list has more than one element, so any accidental
call on a multi-fault row fails loud rather than quietly reporting only one of
two faults' data. The plot loop checks the sentinel and hides the subplot.

**Expected Outcomes**
- `_affected_channel()` parses `fault_params` JSON:
  - 1-element list → `str(params[0].get("channel", "y_pos"))` (unchanged in effect for single-fault).
  - >1-element list → `None` (explicit skip sentinel for multi-fault rows).
  - Row absent / column missing → `"y_pos"` (existing fallback).
- `_onset_step()` parses `fault_params` JSON:
  - 1-element list → `int(params[0]["onset"])`.
  - >1-element list → `-1` (explicit skip sentinel).
  - Row absent / column missing → `0`.
- The plot loop guards on the sentinel: if `channel is None or onset < 0`, hide
  the subplot and continue — no silent wrong plot.
- No `param_channel` / `param_onset` column accesses remain anywhere in the file.
- Printed summary block includes multi-fault episode count if `MULTI_FAULTED_DIR` exists.

**Todo List**
1. Rewrite `_affected_channel()`:
   - Look up the row by filename (same as now).
   - If row is empty or `fault_params` column absent → return `"y_pos"`.
   - Parse `json.loads(row.iloc[0]["fault_params"])` → list of dicts.
   - If `len(params) > 1` → return `None`.
   - Return `str(params[0].get("channel", "y_pos"))`.
2. Rewrite `_onset_step()`:
   - Same lookup pattern.
   - If absent → return `0`.
   - Parse JSON list.
   - If `len(params) > 1` → return `-1`.
   - Return `int(params[0]["onset"])`.
3. In `run_sanity_check()`'s plot loop, after obtaining `channel` and `onset`,
   add a guard before the plot calls:
   ```python
   if channel is None or onset < 0:
       ax.set_visible(False)
       continue
   ```
4. Add `import json` at the top of the imports block.
5. Import `MULTI_FAULTED_DIR` from `config`.
6. In the printed summary block, after the single-fault loop, add:
   ```python
   if os.path.isdir(MULTI_FAULTED_DIR):
       multi_total = sum(
           len([f for f in os.listdir(os.path.join(MULTI_FAULTED_DIR, combo)) if f.endswith(".csv")])
           for combo in os.listdir(MULTI_FAULTED_DIR)
           if os.path.isdir(os.path.join(MULTI_FAULTED_DIR, combo))
       )
       print(f"  {'multi-fault (all combos)':<34}: {multi_total}")
       total += multi_total
   ```

**Relevant Context**
- `_affected_channel()`: [`sanity_check.py:59-67`](sentinel/data_generation/sanity_check.py:59)
- `_onset_step()`: [`sanity_check.py:70-74`](sentinel/data_generation/sanity_check.py:70)
- Plot loop (needs sentinel guard): [`sanity_check.py:119-144`](sentinel/data_generation/sanity_check.py:119)
- Printed summary: [`sanity_check.py:152-173`](sentinel/data_generation/sanity_check.py:152)

**Status:** [x] done

---

### Sub-Task 6 — Wire `--multi-fault-episodes` CLI flag

**Intent**
Add the `--multi-fault-episodes N` argument to both CLI entry points so
multi-fault generation is actually reachable from the command line. The default
is **`0`** — consistent with the top-level invariant that an unmodified invocation
is byte-identical to pre-Phase-1 output. Users opt into multi-fault generation
explicitly by passing a non-zero value. `N_MULTI_FAULT_EPISODES` (100) is
available as a named constant for production full-run invocations but is not the
CLI default.

**Expected Outcomes**
- `generate_dataset.py` (no flags) → zero multi-fault episodes; output identical
  to current behavior.
- `generate_dataset.py --multi-fault-episodes 10` → 10 dual-fault episodes.
- `run_all.py --multi-fault-episodes 10` does the same then runs sanity check.
- Both files use `default=0` for `--multi-fault-episodes`.
- `generate_dataset.py` passes `n_multi=args.multi_fault_episodes` to
  `generate_dataset()`.
- `run_all.py` passes it through and includes the multi-fault count in its
  opening summary print.

**Todo List**
1. In `generate_dataset.py`:
   - Add `parser.add_argument("--multi-fault-episodes", type=int,
     default=0, metavar="K", dest="multi_fault_episodes",
     help="Number of dual-fault episodes to generate (default: 0).")`.
   - Pass `n_multi=args.multi_fault_episodes` to `generate_dataset()`.
2. In `run_all.py`:
   - Same `add_argument` call with `default=0`.
   - Pass `n_multi=args.multi_fault_episodes` to `generate_dataset()`.
   - Update the opening summary print to include the multi-fault count.

**Relevant Context**
- `generate_dataset.py` argparse block: [`generate_dataset.py:36-47`](sentinel/data_generation/generate_dataset.py:36)
- `run_all.py` argparse block: [`run_all.py:29-34`](sentinel/data_generation/run_all.py:29)

**Status:** [x] done

---

## Implementation Order

Sub-Tasks 1 → (2 and 3 independently) → 4 → 5 → 6

- **1 first**: `MAX_SIMULTANEOUS_FAULTS` is imported by `faults.py` (Sub-Task 3)
  and `MULTI_FAULTED_DIR` / `N_MULTI_FAULT_EPISODES` by `scenario_generator.py`
  (Sub-Task 4) — config must be done first.
- **2 and 3 are independent**: Sub-Task 2 touches only `simulation.py`;
  Sub-Task 3 touches only `faults.py`. Neither imports the other. They can be
  done in either order or simultaneously.
- **4 requires both 2 and 3**: `scenario_generator.py` imports `build_multi_fault`
  (from Sub-Task 3) and calls `run_episode(faults=[…])` (new signature from
  Sub-Task 2).
- **5 after 4**: `sanity_check.py` reads `labels.csv` in the Option C schema
  that Sub-Task 4 writes.
- **6 last**: purely additive CLI wiring; no code in Sub-Tasks 2–5 depends on it.

## Validation Checklist (Definition of Done)

- [ ] `run_episode(seed=s, faults=None)` output byte-identical to pre-Phase-1 output for same seed.
- [ ] `run_episode(seed=s, faults=[f])` output byte-identical to pre-Phase-1 `run_episode(seed=s, fault=f)` for same seed.
- [ ] Multi-fault batch runs without error; CSVs land in `data/faulted_multi/<combo>/`.
- [ ] `labels.csv` contains `fault_params` JSON list column; no `param_*` columns.
- [ ] Multi-fault label row contains both faults' full `.params` (two-element JSON list).
- [ ] Same seed → byte-identical multi-fault CSV output (determinism).
- [ ] `sanity_check.py` runs to completion; single-fault `_onset_step` and `_affected_channel` return correct values from the new JSON schema.
- [ ] `python generate_dataset.py --multi-fault-episodes 5 --nominal 2 --fault-each 1` completes cleanly.
- [ ] **Conflict-rule integrity check:** generate ≥ 500 multi-fault episodes, then programmatically assert over all `labels.csv` multi-fault rows: (a) no row has two fault types both in `_ACTUATOR_FAULT_TYPES`; (b) no row has two sensor faults with the same `channel` value in their `fault_params` JSON; (c) all `fault_type` combo strings are in sorted order (i.e. `typeA <= typeB` alphabetically), confirming no fragmentation occurred.

## Implementation Notes

- **`fault_label` in the CSV vs `fault_type` in labels.csv:** both use the
  alphabetically-sorted `+`-joined string for multi-fault episodes. This is
  intentional — canonical ordering prevents directory/label fragmentation, and
  the backend can split on `+` to recover the individual types.
- **Call order and actuator faults:** alphabetical sorting of fault types means
  `actuator_stuck` comes before most sensor fault names (e.g. before
  `sensor_dropout`). Since a sensor fault never modifies `action`, order is
  provably immaterial for any actuator+sensor pairing — the actuator fault's
  `action` override is the only one, regardless of whether it fires first or
  last in the chain. No special ordering rule is needed.
- **Channel-collision resample note (Sub-Task 3):** `FilterMisclassification`
  only targets `["x_pos", "y_pos"]`, narrower than the 6-channel
  `_random_continuous_channel` pool. If fault A is `FilterMisclassification` and
  fault B is any sensor fault that also drew `x_pos` or `y_pos`, the resample
  loop will trigger. This is expected and fine — it just means some (A, B) type
  pairs are slightly more likely to require a resample.
