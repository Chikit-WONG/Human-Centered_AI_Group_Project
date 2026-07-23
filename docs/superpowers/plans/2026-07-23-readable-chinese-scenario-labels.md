# Readable Chinese Scenario Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace internal scenario slugs in the Chinese matching-fairness report with deterministic natural-language experiment descriptions while preserving every raw machine-readable identifier and metric.

**Architecture:** Add two pure formatting helpers to the existing reporting module: one derives a Chinese label and matrix dimensions from a standard `ScenarioSpec`, and one maps duplicate-EEG scenario indices 27–29. Only Chinese table rendering calls these helpers; English rendering and raw aggregation remain unchanged. Regenerate only the published Chinese Markdown from the already validated 450 records.

**Tech Stack:** Python 3.10, pytest, existing `matching_fairness.reporting` Markdown renderer, Git.

## Global Constraints

- Cover the existing 27 standard scenarios and 3 real duplicate-EEG scenarios only.
- Do not change checkpoints, matrices, algorithms, metrics, raw CSV, scenario manifests, summaries, or per-query ledgers.
- Keep English reports and their internal slugs unchanged.
- Standard labels must describe nonzero delete-EEG, delete-image, and duplicate-image operations and include the exact matrix dimensions.
- Duplicate labels must distinguish the 200×200 EEG-A base from +10 and +20 real EEG-B queries.
- Continue on branch `ckw`; create no branch.

---

### Task 1: Add TDD coverage for readable Chinese labels

**Files:**
- Modify: `experiments/matching_fairness/tests/test_reporting.py:307-325`

**Interfaces:**
- Consumes: `render_chinese_report(aggregate: AggregateBundle, audits=()) -> str` and `render_english_report(...) -> str`.
- Produces: regression expectations that Chinese output uses natural-language labels and English output retains internal slugs.

- [ ] **Step 1: Write the failing test**

Add this test after `test_standard_presentation_represents_all_27_scenarios`:

```python
def test_chinese_report_uses_readable_scenario_labels_only() -> None:
    aggregate = aggregate_records(valid_records())
    chinese = render_chinese_report(aggregate, audit_rows())
    english = render_english_report(aggregate, audit_rows())

    expected_standard = {
        0: "标准一一匹配（200 条 EEG × 200 张图片）",
        1: "重复 10 张图片（200 条 EEG × 210 张图片）",
        15: "删除 5 条 EEG、删除 10 张图片（195 条 EEG × 190 张图片）",
        17: "删除 5 条 EEG、删除 10 张图片、重复 20 张图片（195 条 EEG × 210 张图片）",
        26: "删除 10 条 EEG、删除 10 张图片、重复 20 张图片（190 条 EEG × 210 张图片）",
    }
    for label in expected_standard.values():
        assert chinese.count(label) == 3

    assert chinese.count("真实重复 EEG 基准（200 条 EEG-A × 200 张图片）") == 5
    assert chinese.count("加入 10 条真实重复 EEG-B（210 条 EEG × 200 张图片）") == 5
    assert chinese.count("加入 20 条真实重复 EEG-B（220 条 EEG × 200 张图片）") == 5
    assert all(token not in chinese for token in ("dropq", "dropp", "dupg", "dupq"))
    assert "00 dropq0_dropg0_dropp0_dupg0" in english
    assert "dupq20" in english
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=experiments/matching_fairness \
  /hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest -q \
  experiments/matching_fairness/tests/test_reporting.py::test_chinese_report_uses_readable_scenario_labels_only
```

Expected: FAIL because the Chinese report still contains `dropq...` and `dupq...`.

### Task 2: Implement deterministic Chinese scenario labels

**Files:**
- Modify: `experiments/matching_fairness/matching_fairness/reporting.py:1289-1395`
- Test: `experiments/matching_fairness/tests/test_reporting.py`

**Interfaces:**
- Consumes: `ScenarioSpec.drop_query`, `drop_gallery`, `drop_pair`, `duplicate_gallery`, and scenario indices 27–29.
- Produces: `_chinese_standard_scenario_label(spec: ScenarioSpec) -> str` and `_chinese_duplicate_scenario_label(index: int) -> str`.

- [ ] **Step 1: Add the minimal formatting helpers**

Add before `_standard_table`:

```python
def _chinese_standard_scenario_label(spec: object) -> str:
    operations = []
    if spec.drop_query:
        operations.append(f"删除 {spec.drop_query} 条 EEG")
    if spec.drop_gallery:
        operations.append(f"删除 {spec.drop_gallery} 张图片")
    if spec.duplicate_gallery:
        operations.append(f"重复 {spec.duplicate_gallery} 张图片")
    description = "、".join(operations) if operations else "标准一一匹配"
    query_count = 200 - spec.drop_query - spec.drop_pair
    gallery_count = 200 - spec.drop_gallery - spec.drop_pair + spec.duplicate_gallery
    return f"{description}（{query_count} 条 EEG × {gallery_count} 张图片）"


def _chinese_duplicate_scenario_label(index: int) -> str:
    labels = {
        27: "真实重复 EEG 基准（200 条 EEG-A × 200 张图片）",
        28: "加入 10 条真实重复 EEG-B（210 条 EEG × 200 张图片）",
        29: "加入 20 条真实重复 EEG-B（220 条 EEG × 200 张图片）",
    }
    try:
        return labels[index]
    except KeyError:
        raise ValueError(f"invalid duplicate-EEG scenario index: {index}") from None
```

- [ ] **Step 2: Route only Chinese tables through the helpers**

In `_standard_table`, replace the unconditional scenario cell with:

```python
scenario_label = (
    _chinese_standard_scenario_label(spec)
    if language == "zh"
    else f"{index:02d} {spec.slug}"
)
cells = [scenario_label, model]
```

In `_duplicate_table`, replace `str(row["scenario"])` with:

```python
(
    _chinese_duplicate_scenario_label(int(row["scenario_index"]))
    if language == "zh"
    else str(row["scenario"])
)
```

- [ ] **Step 3: Run the new test to verify GREEN**

Run the Task 1 command again.

Expected: `1 passed`.

- [ ] **Step 4: Run the complete reporting regression suite and Ruff**

Run:

```bash
PYTHONPATH=experiments/matching_fairness \
  /hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest -q \
  experiments/matching_fairness/tests/test_reporting.py

/hpc2hdd/home/ckwong627/miniconda3/envs/atm_native/bin/ruff check \
  experiments/matching_fairness/matching_fairness/reporting.py \
  experiments/matching_fairness/tests/test_reporting.py
```

Expected: all reporting tests pass and Ruff prints `All checks passed!`.

- [ ] **Step 5: Commit the tested implementation**

```bash
git add \
  experiments/matching_fairness/matching_fairness/reporting.py \
  experiments/matching_fairness/tests/test_reporting.py
git commit -m "feat(fairness): clarify Chinese scenario labels"
```

### Task 3: Regenerate and verify the published Chinese report

**Files:**
- Replace generated runtime file: `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS_ZH.md`
- Preserve: `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/aggregate/aggregate_metrics.csv`

**Interfaces:**
- Consumes: the existing validated `runs/` tree and `aggregate_records(load_run_records(...))`.
- Produces: one updated Chinese Markdown report; no experimental artifact changes.

- [ ] **Step 1: Record raw CSV and English report hashes**

```bash
sha256sum \
  /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/aggregate/aggregate_metrics.csv \
  /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS.md
```

Expected: two SHA-256 values retained for comparison.

- [ ] **Step 2: Render Chinese Markdown from the existing 450 records**

Use `load_run_records`, `aggregate_records`, and `render_chinese_report` to write a temporary sibling file, `fsync` it, then use `os.replace` to replace only `RESULTS_ZH.md`. Pass an empty audit tuple because the rapid-mode aggregate intentionally contains no best-test audit.

- [ ] **Step 3: Verify display and immutable artifacts**

Run:

```bash
! rg -n 'dropq|dropp|dupg|dupq' \
  /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS_ZH.md

rg -n '标准一一匹配|删除 5 条 EEG|重复 20 张图片|加入 20 条真实重复 EEG-B' \
  /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/aggregate/RESULTS_ZH.md
```

Expected: no internal slug in Chinese output and all representative natural-language labels are present. Re-run Step 1 and confirm both raw CSV and English-report hashes are unchanged.

- [ ] **Step 4: Push the existing ckw branch**

```bash
git push origin ckw
```

Expected: remote `ckw` advances to the implementation commit; no new branch is created.
