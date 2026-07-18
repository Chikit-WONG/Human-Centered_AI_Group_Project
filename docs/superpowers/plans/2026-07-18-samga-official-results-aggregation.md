# SAMGA Official Results Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only, deterministic aggregator for official SAMGA reproduction `result.csv` cells, with strict validation, completeness reporting, protocol-group statistics, and bilingual reports.

**Architecture:** `aggregate_official_results.py` separates input discovery/validation from pure aggregation/rendering and writes only after the complete input scan succeeds. A single pytest module builds temporary official-run trees, exercises public functions and the CLI, and proves that malformed or ambiguous input never produces aggregate files.

**Tech Stack:** Python standard library (`argparse`, `csv`, `json`, `math`, `pathlib`, `statistics`) and pytest.

## Global Constraints

- Never modify the released SAMGA implementation or anything below `artifacts/samga_reproduction/official_runs`.
- Do not submit SLURM jobs and do not create a git commit.
- Interpret `top1 acc`/`top5 acc` as final or stopping-epoch metrics.
- Interpret `best *` metrics as per-epoch test-selected diagnostics with test leakage.
- Describe the metrics as standard independent per-query retrieval, not Hungarian assignment.
- Permit incomplete scans and list missing cells; reject malformed and ambiguous completed cells.
- Produce byte-stable CSV, JSON, English Markdown, and Chinese Markdown.

---

### Task 1: Strict cell discovery and CSV validation

**Files:**
- Create: `experiments/samga_reproduction/aggregate_official_results.py`
- Test: `experiments/samga_reproduction/tests/test_aggregate_official_results.py`

**Interfaces:**
- Consumes: an `official_runs` directory with `<variant>/seed<seed>/sub-XX-b<B>-p<P>/<timestamp>/result.csv`.
- Produces: `scan_official_runs(input_root: Path, expected_subjects: tuple[int, ...] | None, expected_seeds: tuple[int, ...] | None) -> ScanResult`.

- [ ] **Step 1: Write failing temporary-tree tests**

```python
def test_scan_parses_official_cell_and_reports_expected_missing_cells(tmp_path):
    runs = tmp_path / "official_runs"
    write_result(runs, variant="raw", seed=2025, subject=1)
    scan = module.scan_official_runs(
        runs, expected_subjects=(1, 2), expected_seeds=(2025, 2026)
    )
    assert scan.rows[0].final_top1_percent == 75.0
    assert len(scan.missing_cells) == 3
```

- [ ] **Step 2: Run RED**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py`

Expected: collection fails because `aggregate_official_results.py` does not exist.

- [ ] **Step 3: Implement minimal scanner**

Define immutable row, protocol-key, missing-cell, and scan-result dataclasses; validate exact path components, unique result per logical cell, exact single data row, unique required headers, finite numeric values, percentage/loss/epoch ranges, and Top-5 ≥ Top-1.

- [ ] **Step 4: Run scanner tests GREEN**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py`

Expected: scanner cases pass.

### Task 2: Protocol statistics and semantic output contract

**Files:**
- Modify: `experiments/samga_reproduction/aggregate_official_results.py`
- Modify: `experiments/samga_reproduction/tests/test_aggregate_official_results.py`

**Interfaces:**
- Consumes: validated `ResultRow` objects.
- Produces: `build_group_summaries(rows: Sequence[ResultRow]) -> tuple[GroupSummary, ...]`, with groups keyed by `(variant, batch_size, early_stop_patience)`.

- [ ] **Step 1: Add failing statistics tests**

Assert `n`, means, sample standard deviations (`ddof=1`, zero for singleton groups), and signed percentage-point gaps from paper Top-1 `91.3` and Top-5 `98.8` for both final and test-selected metrics.

- [ ] **Step 2: Run RED**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py -k statistics`

Expected: failure because group summaries are not implemented.

- [ ] **Step 3: Implement summaries**

Sort groups and rows explicitly and compute all fields from validated finite floats.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py -k statistics`

Expected: statistics cases pass.

### Task 3: Deterministic CLI reports

**Files:**
- Modify: `experiments/samga_reproduction/aggregate_official_results.py`
- Modify: `experiments/samga_reproduction/tests/test_aggregate_official_results.py`

**Interfaces:**
- Consumes: `--input-root`, `--output-dir`, `--expected-subjects`, and `--expected-seeds`.
- Produces: `official_results.csv`, `official_results.json`, `official_results.md`, and `official_results_zh.md`.

- [ ] **Step 1: Add failing CLI/output tests**

Run the CLI entry function twice against identical temporary fixtures and separate output directories; compare all output bytes and assert bilingual leakage/protocol warnings, missing-cell rows, semantic JSON keys, and stable CSV ordering.

- [ ] **Step 2: Run RED**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py -k output`

Expected: failure because rendering and CLI output are not implemented.

- [ ] **Step 3: Implement deterministic rendering and safe writes**

Render all content in memory, reject output paths inside the input tree and pre-existing target files, create the output directory only after validation, and write UTF-8 text with fixed newlines.

- [ ] **Step 4: Run all target tests GREEN**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py`

Expected: all target tests pass.

### Task 4: Fresh verification

**Files:**
- Verify: `experiments/samga_reproduction/aggregate_official_results.py`
- Verify: `experiments/samga_reproduction/tests/test_aggregate_official_results.py`

- [ ] **Step 1: Compile**

Run: `python -m py_compile experiments/samga_reproduction/aggregate_official_results.py experiments/samga_reproduction/tests/test_aggregate_official_results.py`

Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run: `pytest -q experiments/samga_reproduction/tests/test_aggregate_official_results.py`

Expected: zero failures.

- [ ] **Step 3: Run the relevant test directory**

Run: `pytest -q experiments/samga_reproduction/tests`

Expected: zero failures.

- [ ] **Step 4: Inspect only intended changes**

Run: `git status --short` and `git diff --no-index /dev/null <new-file>` for both implementation files.

Expected: no official SAMGA or official-run artifact has changed.
