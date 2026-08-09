# Phase 3 — Leave-One-Fault-Out (LOFO) Split Generator Plan

## Top-Level Overview

Add a pure-filtering manifest generator that, for each of the 7 known fault
types, produces a train/test split that fully excludes that type from training.
No episode data is regenerated or duplicated — the manifests are CSV index files
over whatever already exists on disk.

**Also addressed in this phase (optional cleanups):**
- Consolidate the duplicated `_safe_landing()` helper from
  `scenario_generator.py` and `sanity_check.py` into a shared module.
- Add per-fault-type crash-rate logging to `scenario_generator.py`'s summary.

The work touches/creates five files:

| File | Nature of change |
|---|---|
| `data_generation/utils.py` | **New** — shared `_safe_landing()` helper |
| `data_generation/splits.py` | **New** — LOFO split generator |
| `data_generation/scenario_generator.py` | Import `_safe_landing` from `utils`; add per-fault crash-rate to summary |
| `data_generation/sanity_check.py` | Import `_safe_landing` from `utils` |
| `data_generation/run_all.py` | Add optional `--make-splits` flag |

`config.py`, `faults.py`, `simulation.py`, `generate_dataset.py` are **not
touched**.

---

## Sub-Tasks

---

### Sub-Task 1 — Create `utils.py` with shared `_safe_landing()`

**Intent**
Eliminate the identical `_safe_landing()` definition that currently exists in
both `scenario_generator.py` and `sanity_check.py`. Move it to a single shared
module so future changes only need to happen in one place.

**Expected Outcomes**
- `data_generation/utils.py` exports `safe_landing(df) -> bool`.
  (Public name without leading underscore — it's an exported helper now.)
- `scenario_generator.py` removes its local `_safe_landing` and imports
  `safe_landing` from `.utils`.
- `sanity_check.py` removes its local `_safe_landing` and imports `safe_landing`
  from `data_generation.utils`.
- All existing callers (`scenario_generator._safe_landing(df)`,
  `sanity_check._safe_landing(df)`) updated to use `safe_landing(df)`.

**Todo List**
1. Create `sentinel/data_generation/utils.py`:
   ```python
   """utils.py — shared helpers for the data-generation pipeline."""
   from __future__ import annotations
   import pandas as pd

   def safe_landing(df: pd.DataFrame) -> bool:
       """
       Heuristic: episode ended with terminated=1 (ground contact) and
       the final y_vel was gentle (> -0.3 m/s equivalent in obs units).
       """
       last = df.iloc[-1]
       return bool(last["terminated"] == 1 and last["y_vel"] > -0.3)
   ```
2. In `scenario_generator.py`:
   - Remove the `_safe_landing` function definition.
   - Add `from .utils import safe_landing` to the imports.
   - Replace the two `_safe_landing(df)` call sites with `safe_landing(df)`.
3. In `sanity_check.py`:
   - Remove the `_safe_landing` function definition.
   - Add `from data_generation.utils import safe_landing` to the imports.
   - Replace the one `_safe_landing(df)` call site with `safe_landing(df)`.

**Relevant Context**
- `_safe_landing` in `scenario_generator.py`:
  [`scenario_generator.py:62-68`](sentinel/data_generation/scenario_generator.py:62)
- `_safe_landing` in `sanity_check.py`:
  [`sanity_check.py:46-48`](sentinel/data_generation/sanity_check.py:46)
- Both definitions are byte-identical — no reconciliation needed.

**Status:** [x] done

---

### Sub-Task 2 — Add per-fault crash-rate to `scenario_generator.py` summary

**Intent**
After Phase 0, the faulted episode CSVs contain `y_vel` at the final step, so
`safe_landing()` works on them. Adding per-fault crash rates to the summary
printout gives an immediately visible sanity signal: `actuator_stuck` should
crash far more often than `sensor_bias_drift`, for example.

**Expected Outcomes**
- The dataset summary block in `generate_dataset()` logs, for each fault type:
  ```
    sensor_dropout                  : 50  (safe: 38 / 50)
    thrust_degradation              : 50  (safe:  9 / 50)
    ...
  ```
- Nominal safe-landing line is unchanged.
- Safe-landing counts are accumulated **inline during generation** — the
  `df` returned by `run_episode()` is still in memory at the point it is
  written to CSV, so `safe_landing(df)` is called there at zero I/O cost.
  No disk re-scan, no re-read, no `os.listdir`, no fragile "last N files"
  slice.

**Todo List**
1. Before the single-fault generation loop, initialise a per-type counter:
   ```python
   fault_safe_counts: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}
   ```
2. Inside the single-fault inner loop, immediately after `run_episode()` returns
   `df` and before writing it to disk, add:
   ```python
   if safe_landing(df):
       fault_safe_counts[ft] += 1
   ```
3. In the summary block, replace:
   ```python
   for ft in FAULT_TYPES:
       print(f"  {ft:<32}: {n_fault_each}")
   ```
   with:
   ```python
   for ft in FAULT_TYPES:
       safe_f = fault_safe_counts[ft]
       print(f"  {ft:<32}: {n_fault_each}  (safe: {safe_f:>3} / {n_fault_each})")
   ```

**Relevant Context**
- Single-fault generation loop: [`scenario_generator.py:97-123`](sentinel/data_generation/scenario_generator.py:97)
  — `df` is available on the line immediately before `df.to_csv(...)`.
- Summary block: [`scenario_generator.py:171-185`](sentinel/data_generation/scenario_generator.py:171)
- `safe_landing` will be available from Sub-Task 1's import.

**Status:** [x] done

---

### Sub-Task 3 — Create `splits.py`

**Intent**
Implement the LOFO split generator. For each of the 7 fault types, produce a
train and test manifest CSV. No episode data is touched — manifests are pure
index tables with absolute paths.

**Multi-fault episode handling (Phase 1 schema):**
`labels.csv` has a `fault_type` column that for multi-fault episodes contains
the `+`-joined pair string, e.g. `"filter_misclassification+sensor_dropout"`.
Checking whether `held_out_type` is present in an episode means checking whether
that string appears as a substring of `fault_type` (or equivalently, whether
`held_out_type` is in `fault_type.split("+")`). An episode labeled
`"actuator_stuck+sensor_dropout"` is excluded from the `sensor_dropout` LOFO
train set even though it also involves `actuator_stuck`.

**Nominal proportionality (Option B):**
For a given `held_out_type`, let `f_held` = number of faulted episodes
referencing that type, `f_total` = total faulted episodes in `labels.csv`.
Hold-out fraction `p = f_held / f_total`. The test set gets
`round(p * N_nominal)` nominal episodes (sampled with a fixed seed for
reproducibility); the train set gets the remaining `N_nominal - n_test_nominal`
nominal episodes.

**Manifest schema:**
Each manifest CSV has columns:
- `path` — absolute path to the episode CSV
- `fault_type` — the episode's `fault_type` string (`"nominal"` for nominals)
- `split` — `"train"` or `"test"`

**Expected Outcomes**
- `make_lofo_splits(labels_csv, nominal_dir, splits_dir, seed)` produces 14
  files: `lofo_<type>_train.csv` and `lofo_<type>_test.csv` for all 7 types.
- For each split: train contains zero episodes (including multi-fault) where
  `held_out_type` appears in `fault_type`.
- For each split: test contains all episodes where `held_out_type` appears, plus
  `round(p * N_nominal)` nominals.
- Per-split counts are printed to stdout.
- `splits.py` follows the importable-function + `if __name__ == "__main__":`
  pattern.
- The `__main__` block runs all 7 types in one invocation.

**Todo List**
1. Create `sentinel/data_generation/splits.py` with:
   - Imports: `os`, `sys`, `json`, `math`, `pandas`, `numpy`, paths from
     `data_generation.config` (`LABELS_CSV`, `NOMINAL_DIR`, `DATA_ROOT`),
     `FAULT_TYPES` from `data_generation.faults`.
   - `SPLITS_DIR = os.path.join(DATA_ROOT, "splits")` — define locally (not
     in `config.py`; splits are a derived artifact, not a data root).
2. Implement `_fault_type_matches(fault_type_str: str, held_out: str) -> bool`:
   - Returns `True` if `held_out` appears in `fault_type_str.split("+")`.
   - Handles both single-fault (`"sensor_dropout"`) and multi-fault
     (`"actuator_stuck+sensor_dropout"`) strings correctly.
3. Implement `make_lofo_splits(labels_csv, nominal_dir, splits_dir, seed=0)`:
   - Load `labels_csv` into a DataFrame.
   - Build the nominal file list from `nominal_dir` (files ending in `.csv`),
     sorted for reproducibility. Construct absolute paths.
   - For each `held_out_type` in `FAULT_TYPES`:
     a. **Faulted split:** partition `labels.csv` rows into
        `faulted_test` (where `_fault_type_matches`) and
        `faulted_train` (where it does not).
     b. **Nominal split:** compute `p = len(faulted_test) / len(labels)`.
        `n_test_nom = round(p * len(nominal_paths))`.
        Shuffle nominals with `np.random.default_rng(seed)` and split:
        first `n_test_nom` → test, rest → train.
     c. Build train and test manifest DataFrames, each with columns
        `path`, `fault_type`, `split`.
        - For faulted rows, `path` = `os.path.join(DATA_ROOT, row["filename"])`.
          Note: `labels.csv` `filename` column stores a relative path like
          `"sensor_dropout/sensor_dropout_0000.csv"` for single-fault or
          `"measurement_delay+sensor_dropout/multi_...csv"` for multi-fault —
          prefix with the appropriate base dir:
          - If `fault_type` contains `"+"` → prefix with `MULTI_FAULTED_DIR`.
          - Otherwise → prefix with `FAULTED_DIR`.
        - For nominal rows, `path` = `os.path.join(nominal_dir, fname)`,
          `fault_type` = `"nominal"`.
     d. Write train and test manifests to `splits_dir`.
     e. Print a count summary for this split.
4. `__main__` block:
   ```python
   if __name__ == "__main__":
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument("--labels",  default=LABELS_CSV)
       parser.add_argument("--nominal", default=NOMINAL_DIR)
       parser.add_argument("--out",     default=SPLITS_DIR)
       parser.add_argument("--seed",    type=int, default=0)
       args = parser.parse_args()
       make_lofo_splits(args.labels, args.nominal, args.out, args.seed)
   ```

**Relevant Context**
- `labels.csv` `filename` field examples:
  - Single-fault: `"sensor_dropout/sensor_dropout_0042.csv"`
  - Multi-fault: `"measurement_delay+sensor_dropout/multi_measurement_delay+sensor_dropout_0003.csv"`
- `FAULTED_DIR` and `MULTI_FAULTED_DIR` from `config` are the correct prefixes.
- `FAULT_TYPES` from `faults` is the authoritative list of 7 type strings.

**Status:** [x] done

---

### Sub-Task 4 — Wire `--make-splits` flag into `run_all.py`

**Intent**
Make the split generation reachable as a one-liner alongside dataset generation,
following the same opt-in pattern as `--multi-fault-episodes`.

**Expected Outcomes**
- `run_all.py --make-splits` generates splits after the dataset + sanity check.
- Default behavior (no flag) is unchanged — splits are not generated unless
  requested.
- The flag is a boolean `store_true` argument, not a count, because
  `make_lofo_splits` has no user-tunable parameters beyond those already in
  `config.py`.

**Todo List**
1. In `run_all.py`:
   - Add `parser.add_argument("--make-splits", action="store_true",
     dest="make_splits", help="Generate LOFO train/test split manifests.")`.
   - After the sanity-check block, add:
     ```python
     if args.make_splits:
         print("\n==> Generating LOFO splits …")
         from data_generation.splits import make_lofo_splits, SPLITS_DIR
         make_lofo_splits(LABELS_CSV, NOMINAL_DIR, SPLITS_DIR)
         print(f"Splits written to: {SPLITS_DIR}")
     ```
   - Import `LABELS_CSV` and `NOMINAL_DIR` from `data_generation.config` at
     the top (both already available in the module's existing import).

**Relevant Context**
- `run_all.py` main function: [`run_all.py:24-55`](sentinel/data_generation/run_all.py:24)
- Existing imports in `run_all.py` already include `generate_dataset` and
  `FAULT_TYPES` — `LABELS_CSV` and `NOMINAL_DIR` are not yet imported there.

**Status:** [x] done

---

## Implementation Order

Sub-Tasks 1 → 2 → 3 → 4

- **1 first**: `safe_landing` must be in `utils.py` before `scenario_generator`
  and `sanity_check` can import it. Sub-Task 2 also uses it.
- **2 after 1**: needs `safe_landing` from utils.
- **3 independent of 1 and 2**: `splits.py` doesn't use `safe_landing` and
  doesn't import from `scenario_generator` or `sanity_check`.
- **4 last**: purely additive wiring; imports from `splits.py` (Sub-Task 3).

## Validation Checklist (Definition of Done)

- [ ] `safe_landing` is defined exactly once (in `utils.py`); both
  `scenario_generator.py` and `sanity_check.py` import and use it.
- [ ] All existing tests and summary calls still work after the consolidation.
- [ ] For each of the 7 LOFO splits: train manifest has zero rows where
  `held_out_type` appears anywhere in `fault_type` (including multi-fault pairs).
- [ ] For each split: test manifest contains all episodes with `held_out_type`,
  plus a proportional slice of nominals.
- [ ] Train + test nominal counts sum to `N_nominal` for every split.
- [ ] Train + test faulted counts equal the total episodes in `labels.csv`.
- [ ] `python data_generation/splits.py` runs end-to-end for all 7 types;
  14 files written to `data/splits/`.
- [ ] `run_all.py --make-splits --nominal 2 --fault-each 1` completes cleanly
  and produces all 14 split files.
- [ ] Per-fault crash-rate is visible in `generate_dataset()` summary output.

## Implementation Notes

- **`SPLITS_DIR` not in `config.py`:** splits are a derived analysis artifact,
  not raw data. Keeping the path local to `splits.py` avoids polluting `config`
  with paths that aren't needed by the generation pipeline.
- **`filename` path prefix logic:** `labels.csv` stores relative paths from the
  *type-specific* output directory root, not from `DATA_ROOT`. Single-fault
  filenames are relative to `FAULTED_DIR`; multi-fault filenames are relative to
  `MULTI_FAULTED_DIR`. The split generator must prefix accordingly, not blindly
  use `DATA_ROOT`.
- **Nominal shuffle seed:** the same `seed` is used for all 7 LOFO types'
  nominal shuffle. This means the nominal test slices for different LOFO types
  will be drawn from the same shuffled order — they will partially overlap (both
  test sets include ~14% of nominals, with some of the same episodes). This is
  intentional: nominals aren't typed, so there's no reason to exclude any
  specific nominal episode from any type's test set.
- **Multi-fault episodes in splits:** a `"filter_misclassification+noise_spike"`
  episode appears in the train set for all LOFO types *except*
  `filter_misclassification` and `noise_spike` — it is excluded from both.
  This is the correct strict interpretation of "fully excludes that type from
  training."
