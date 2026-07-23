# Readable English Scenario Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace internal scenario slugs in the English matching-fairness report with deterministic natural-language labels while preserving all raw experiment artifacts.

**Architecture:** Add two English-only label functions beside the existing Chinese label functions in `reporting.py`. Route only English Markdown table rendering through those functions; keep `ScenarioSpec`, aggregate records, CSV fields, manifests, metrics, and Chinese rendering unchanged.

**Tech Stack:** Python 3.10+, pytest, Ruff, Markdown, SHA-256 verification.

## Global Constraints

- Work only on the existing `ckw` branch; do not create another branch.
- Standard labels cover all 27 combinations of `drop_query ∈ {0,5,10}`, `drop_gallery ∈ {0,5,10}`, and `duplicate_gallery ∈ {0,10,20}`.
- Duplicate-EEG labels cover scenario indices `27`, `28`, and `29`.
- English and Chinese Markdown reports must contain no internal `dropq`, `dropg`, `dropp`, `dupg`, or `dupq` slug.
- Raw CSV, manifests, ledgers, checkpoints, score matrices, algorithms, and metric values must remain unchanged.
- Regenerate `RESULTS.md` from the existing 450 records; do not rerun training or matching experiments.
- `aggregate_metrics.csv` must retain SHA-256 `09585b114eff11a6a4062b1b244de6848b38ffc1c49899e91b989f0aba93f83c`.
- `RESULTS_ZH.md` must retain SHA-256 `f2d70f70dbaf87823ae8fc7c4be61016dd72a3afb178c3b97954d989581ef0ce`.

---

### Task 1: Render all English scenarios as natural language

**Files:**
- Modify: `experiments/matching_fairness/tests/test_reporting.py:310-342`
- Modify: `experiments/matching_fairness/matching_fairness/reporting.py:1289-1420`

**Interfaces:**
- Consumes: `ScenarioSpec`, `standard_scenarios()`, and each aggregate row's integer `scenario_index`.
- Produces: `_english_standard_scenario_label(spec: ScenarioSpec) -> str` and `_english_duplicate_scenario_label(index: int) -> str`.

- [ ] **Step 1: Replace the obsolete English-slug assertions with a failing public-report test**

Update `test_standard_presentation_represents_all_27_scenarios` and the English assertions in `test_chinese_report_uses_readable_scenario_labels_only` so the tests exercise all 27 standard labels, all three duplicate-EEG labels, and absence of internal slugs:

```python
def test_english_report_uses_readable_scenario_labels_only() -> None:
    aggregate = aggregate_records(valid_records())
    english = render_english_report(aggregate, audit_rows())

    expected_standard = {
        "Baseline one-to-one matching (200 EEG queries × 200 images)",
        "Duplicate 10 images (200 EEG queries × 210 images)",
        "Duplicate 20 images (200 EEG queries × 220 images)",
        "Remove 5 images (200 EEG queries × 195 images)",
        "Remove 5 images, duplicate 10 images (200 EEG queries × 205 images)",
        "Remove 5 images, duplicate 20 images (200 EEG queries × 215 images)",
        "Remove 10 images (200 EEG queries × 190 images)",
        "Remove 10 images, duplicate 10 images (200 EEG queries × 200 images)",
        "Remove 10 images, duplicate 20 images (200 EEG queries × 210 images)",
        "Remove 5 EEG queries (195 EEG queries × 200 images)",
        "Remove 5 EEG queries, duplicate 10 images (195 EEG queries × 210 images)",
        "Remove 5 EEG queries, duplicate 20 images (195 EEG queries × 220 images)",
        "Remove 5 EEG queries, remove 5 images (195 EEG queries × 195 images)",
        "Remove 5 EEG queries, remove 5 images, duplicate 10 images (195 EEG queries × 205 images)",
        "Remove 5 EEG queries, remove 5 images, duplicate 20 images (195 EEG queries × 215 images)",
        "Remove 5 EEG queries, remove 10 images (195 EEG queries × 190 images)",
        "Remove 5 EEG queries, remove 10 images, duplicate 10 images (195 EEG queries × 200 images)",
        "Remove 5 EEG queries, remove 10 images, duplicate 20 images (195 EEG queries × 210 images)",
        "Remove 10 EEG queries (190 EEG queries × 200 images)",
        "Remove 10 EEG queries, duplicate 10 images (190 EEG queries × 210 images)",
        "Remove 10 EEG queries, duplicate 20 images (190 EEG queries × 220 images)",
        "Remove 10 EEG queries, remove 5 images (190 EEG queries × 195 images)",
        "Remove 10 EEG queries, remove 5 images, duplicate 10 images (190 EEG queries × 205 images)",
        "Remove 10 EEG queries, remove 5 images, duplicate 20 images (190 EEG queries × 215 images)",
        "Remove 10 EEG queries, remove 10 images (190 EEG queries × 190 images)",
        "Remove 10 EEG queries, remove 10 images, duplicate 10 images (190 EEG queries × 200 images)",
        "Remove 10 EEG queries, remove 10 images, duplicate 20 images (190 EEG queries × 210 images)",
    }
    assert len(expected_standard) == 27
    for label in expected_standard:
        assert english.count(label) == 3

    assert english.count(
        "Real duplicate-EEG baseline (200 EEG-A queries × 200 images)"
    ) == 15
    assert english.count(
        "Add 10 real duplicate EEG-B queries (210 EEG queries × 200 images)"
    ) == 15
    assert english.count(
        "Add 20 real duplicate EEG-B queries (220 EEG queries × 200 images)"
    ) == 15
    assert all(
        token not in english
        for token in ("dropq", "dropg", "dropp", "dupg", "dupq")
    )
```

Retain the existing Chinese label/count assertions, but delete these now-obsolete lines:

```python
english = render_english_report(aggregate, audit_rows())
assert "00 dropq0_dropg0_dropp0_dupg0" in english
assert "dupq20" in english
```

- [ ] **Step 2: Run the focused test and verify the RED state**

Run:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest -q \
experiments/matching_fairness/tests/test_reporting.py::test_english_report_uses_readable_scenario_labels_only
```

Expected: FAIL because the current English report still contains `00 dropq0_dropg0_dropp0_dupg0` and does not contain `Baseline one-to-one matching (200 EEG queries × 200 images)`.

- [ ] **Step 3: Add the minimal English label functions**

Add beside the Chinese helpers in `reporting.py`:

```python
def _english_standard_scenario_label(spec: ScenarioSpec) -> str:
    operations = []
    if spec.drop_query:
        operations.append(f"remove {spec.drop_query} EEG queries")
    if spec.drop_gallery:
        operations.append(f"remove {spec.drop_gallery} images")
    if spec.duplicate_gallery:
        operations.append(f"duplicate {spec.duplicate_gallery} images")
    description = ", ".join(operations)
    if description:
        description = description[0].upper() + description[1:]
    else:
        description = "Baseline one-to-one matching"
    query_count = 200 - spec.drop_query - spec.drop_pair
    gallery_count = 200 - spec.drop_gallery - spec.drop_pair + spec.duplicate_gallery
    return f"{description} ({query_count} EEG queries × {gallery_count} images)"


def _english_duplicate_scenario_label(index: int) -> str:
    labels = {
        27: "Real duplicate-EEG baseline (200 EEG-A queries × 200 images)",
        28: "Add 10 real duplicate EEG-B queries (210 EEG queries × 200 images)",
        29: "Add 20 real duplicate EEG-B queries (220 EEG queries × 200 images)",
    }
    try:
        return labels[index]
    except KeyError:
        raise ValueError(f"invalid duplicate-EEG scenario index: {index}") from None
```

Route the standard table through the language-specific functions:

```python
scenario_label = (
    _chinese_standard_scenario_label(spec)
    if language == "zh"
    else _english_standard_scenario_label(spec)
)
```

Route the duplicate-EEG table through the language-specific functions:

```python
(
    _chinese_duplicate_scenario_label(int(row["scenario_index"]))
    if language == "zh"
    else _english_duplicate_scenario_label(int(row["scenario_index"]))
)
```

- [ ] **Step 4: Run focused and reporting tests to verify GREEN**

Run:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest -q \
experiments/matching_fairness/tests/test_reporting.py
```

Expected: `28 passed` with no warnings or failures.

- [ ] **Step 5: Run Ruff and commit the tested implementation**

Run:

```bash
/hpc2hdd/home/ckwong627/miniconda3/bin/ruff check \
experiments/matching_fairness/matching_fairness/reporting.py \
experiments/matching_fairness/tests/test_reporting.py
git diff --check
```

Expected: `All checks passed!` and no `git diff --check` output.

Commit:

```bash
git add experiments/matching_fairness/matching_fairness/reporting.py \
  experiments/matching_fairness/tests/test_reporting.py
git commit -m "feat(fairness): clarify English scenario labels"
```

---

### Task 2: Regenerate and verify the English result report

**Files:**
- Modify outside the Git repository: `../test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS.md`
- Verify unchanged: `../test/brain-rw/results/matching_fairness_v3/aggregate/aggregate_metrics.csv`
- Verify unchanged: `../test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS_ZH.md`

**Interfaces:**
- Consumes: `load_run_records(root / "runs")`, `aggregate_records(...)`, and `render_english_report(aggregate, ())`.
- Produces: an atomically replaced UTF-8 `RESULTS.md` generated from exactly 450 existing records.

- [ ] **Step 1: Record all three pre-render hashes**

Run:

```bash
sha256sum \
  ../test/brain-rw/results/matching_fairness_v3/aggregate/aggregate_metrics.csv \
  ../test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS_ZH.md \
  ../test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS.md
```

Expected first two hashes:

```text
09585b114eff11a6a4062b1b244de6848b38ffc1c49899e91b989f0aba93f83c  aggregate_metrics.csv
f2d70f70dbaf87823ae8fc7c4be61016dd72a3afb178c3b97954d989581ef0ce  RESULTS_ZH.md
```

- [ ] **Step 2: Atomically regenerate only `RESULTS.md` from the existing run records**

Run from the repository root:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python - <<'PY'
from pathlib import Path
import os
import tempfile

from matching_fairness.reporting import (
    aggregate_records,
    load_run_records,
    render_english_report,
)

root = Path(
    "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/"
    "AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/"
    "test/brain-rw/results/matching_fairness_v3"
)
target = root / "aggregate" / "RESULTS.md"
aggregate = aggregate_records(load_run_records(root / "runs"))
assert len(aggregate.records) == 450
payload = render_english_report(aggregate, ()).encode("utf-8")
fd, temporary = tempfile.mkstemp(prefix=".RESULTS.md.tmp-", dir=target.parent)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(f"wrote {target} ({len(aggregate.records)} records, {len(payload)} bytes)")
PY
```

Expected: one `wrote ...RESULTS.md (450 records, ... bytes)` line.

- [ ] **Step 3: Verify rendered labels and immutable artifacts**

Run:

```bash
RESULT=../test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS.md \
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python - <<'PY'
from pathlib import Path
import os

text = Path(os.environ["RESULT"]).read_text(encoding="utf-8")
for token in ("dropq", "dropg", "dropp", "dupg", "dupq"):
    assert token not in text, token
expected = {
    "Baseline one-to-one matching (200 EEG queries × 200 images)": 3,
    "Remove 5 EEG queries, remove 10 images (195 EEG queries × 190 images)": 3,
    "Remove 10 EEG queries, remove 10 images, duplicate 20 images (190 EEG queries × 210 images)": 3,
    "Real duplicate-EEG baseline (200 EEG-A queries × 200 images)": 15,
    "Add 10 real duplicate EEG-B queries (210 EEG queries × 200 images)": 15,
    "Add 20 real duplicate EEG-B queries (220 EEG queries × 200 images)": 15,
}
for label, count in expected.items():
    assert text.count(label) == count, (label, text.count(label), count)
print("English scenario labels verified:", len(expected))
PY

sha256sum \
  ../test/brain-rw/results/matching_fairness_v3/aggregate/aggregate_metrics.csv \
  ../test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS_ZH.md
```

Expected: the label verification prints `6`, while the CSV and Chinese report hashes remain exactly equal to the two hashes in Global Constraints.

- [ ] **Step 4: Run the complete matching-fairness suite**

Run:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest -q \
experiments/matching_fairness/tests
```

Expected: `427 passed` with zero failures.

- [ ] **Step 5: Verify the current `ckw` branch before review**

Run:

```bash
git status --short --branch
git log --oneline origin/ckw..HEAD
```

Expected: the branch is `ckw`, no new branch exists, and the worktree is clean. The Task 2 implementer must not push; the controller pushes `ckw` only after the task review and final whole-branch review are approved.

## Controller integration after all reviews

After Task 2 and the final whole-branch review are approved, the controller runs:

```bash
git -c http.version=HTTP/1.1 push origin ckw
git status --short --branch
git rev-parse HEAD
git rev-parse origin/ckw
```

Expected: local `HEAD` equals `origin/ckw`, and no branch was created or deleted.
