# Root README Matching-Fairness Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a compact, numerically audited three-baseline matching-fairness summary in the repository-root English and Chinese READMEs.

**Architecture:** Preserve the existing pending README draft and refine only its matching-fairness scope text and summary section. Treat the local audited `main_table.csv` and aggregate duplicate-EEG report as numerical sources, while linking GitHub readers to the version-controlled bilingual experiment guides for the full protocol.

**Tech Stack:** GitHub-flavored Markdown, Python 3 standard library, POSIX shell, Git.

## Global Constraints

- Work on branch `ckw`; do not create or switch branches.
- Modify only `README.md` and `README_ZH.md`; preserve all pre-existing pending content in those files unless this plan explicitly refines it.
- Do not change experiment outputs, score matrices, checkpoints, or matching code.
- Do not resume the paused SFT+DPO experiment.
- Use the audited local `matching_fairness_v3/aggregate/main_table.csv` with SHA-256 `54f00400eb5c9c9c41c0a855b1d60bc3094672c842a405cd7d7cfca4af151952`.
- Keep `README.md` fully English and the corresponding section of `README_ZH.md` fully Chinese; retain the actual filename `README_ZH.md` and existing mutual navigation.
- Report Independent Top-1/Top-5 as retrieval metrics and Greedy/Hungarian/Stable Matching/Sinkhorn as assignment accuracies; never invent assignment Top-5.
- Retain the Sinkhorn non-convergence warning and the real disjoint-trial duplicate-EEG caveat.

---

### Task 1: Refine and verify the bilingual root summaries

**Files:**
- Modify: `README.md`
- Modify: `README_ZH.md`

**Interfaces:**
- Consumes: local audited `main_table.csv`, aggregate `RESULTS.md`/`RESULTS_ZH.md`, and the version-controlled matching-fairness guides.
- Produces: two numerically identical bilingual summary tables with language-appropriate prose and valid guide links.

- [ ] **Step 1: Record the existing dirty scope and run a failing content check**

Run:

```bash
git status --short --branch
git diff -- README.md README_ZH.md
python - <<'PY'
from pathlib import Path

en = Path("README.md").read_text(encoding="utf-8")
zh = Path("README_ZH.md").read_text(encoding="utf-8")

assert "local audited artifact `matching_fairness_v3/aggregate/main_table.csv`" in en
assert "本地审计产物 `matching_fairness_v3/aggregate/main_table.csv`" in zh
assert "| 基线 | 检查点 / 训练来源 | 独立检索 Top-1 | 独立检索 Top-5 |" in zh
PY
```

Expected: the Python check fails because the pending draft does not yet identify the CSV as a local untracked artifact and the Chinese table header still mixes English and Chinese.

- [ ] **Step 2: Refine the English summary without changing its numbers**

Keep the existing `### Three-baseline matching-fairness comparison` section and its three rows. Replace its source sentence with:

```markdown
This controlled implementation/re-evaluation uses the same seed-42 `sub-08` test queries and the same 200-image gallery for **NICE**, **ATM-S**, and **Our project**. It is a single-subject/single-seed diagnostic rather than a perfect reproduction of the paper results or evidence of cross-subject significance. The table was generated from the local audited artifact `matching_fairness_v3/aggregate/main_table.csv` (SHA-256 `54f00400eb5c9c9c41c0a855b1d60bc3094672c842a405cd7d7cfca4af151952`); that result artifact is not tracked by Git, while the version-controlled experiment guide records the reproducible protocol.
```

Use exactly this table:

```markdown
| Baseline | Checkpoint / training source | Independent Top-1 | Independent Top-5 | Greedy assignment | Hungarian assignment | Stable Matching assignment | Sinkhorn assignment |
|---|---|---:|---:|---:|---:|---:|---:|
| NICE | One seed-42 training run; validation-loss-selected epoch 94 | 18.00% | 41.00% | 22.00% | 31.50% | 23.50% | 31.50%† |
| ATM-S | One seed-42 training run; validation-loss-selected epoch 205 | 46.00% | 76.00% | 52.00% | 62.50% | 50.50% | 65.00%† |
| **Our project** | Existing fixed BrainRW epoch-24/final export plus vision-LoRA adapter; not retrained or validation-selected | **91.00%** | **99.50%** | **95.50%** | **100.00%** | **98.00%** | **100.00%**† |
```

Retain the existing metric-semantics paragraph and duplicate-EEG/Sinkhorn paragraph, but use `Our project's Hungarian` rather than `Our-project Hungarian`. Keep the final link to `experiments/matching_fairness/README.md`.

- [ ] **Step 3: Refine and fully localize the Chinese summary**

Keep the existing `### 三套 baseline 的匹配公平性比较` section and its three numerical rows. Replace its source sentence with:

```markdown
这项受控实现/重新评估对 **NICE**、**ATM-S** 和 **Our project** 使用相同的随机种子 `42`、`sub-08` 测试查询及同一套 200 张图片图库。它是单被试、单随机种子的诊断实验，不是论文结果的完美复现，也不能建立跨被试显著性结论。下表由本地审计产物 `matching_fairness_v3/aggregate/main_table.csv` 生成（SHA-256：`54f00400eb5c9c9c41c0a855b1d60bc3094672c842a405cd7d7cfca4af151952`）；该结果产物不由 Git 跟踪，版本控制中的实验指南则记录了可复现协议。
```

Use exactly this localized header and the same values as English:

```markdown
| 基线 | 检查点 / 训练来源 | 独立检索 Top-1 | 独立检索 Top-5 | 贪心分配准确率 | 匈牙利分配准确率 | 稳定匹配准确率 | Sinkhorn 分配准确率 |
|---|---|---:|---:|---:|---:|---:|---:|
| NICE | seed-42 单次训练；验证损失选中第 94 个 epoch | 18.00% | 41.00% | 22.00% | 31.50% | 23.50% | 31.50%† |
| ATM-S | seed-42 单次训练；验证损失选中第 205 个 epoch | 46.00% | 76.00% | 52.00% | 62.50% | 50.50% | 65.00%† |
| **Our project** | 复用固定 BrainRW epoch-24/final 导出及 vision-LoRA adapter；未重新训练，也未依据验证集选模 | **91.00%** | **99.50%** | **95.50%** | **100.00%** | **98.00%** | **100.00%**† |
```

Use `独立检索` and `分配准确率` consistently in the explanatory prose. Keep the final link to `experiments/matching_fairness/README_ZH.md`.

- [ ] **Step 4: Verify the authoritative data and bilingual numerical parity**

Run this exact checker from the repository root:

```bash
python - <<'PY'
import csv
import hashlib
import re
from pathlib import Path

artifact = Path("../test/brain-rw/results/matching_fairness_v3/aggregate/main_table.csv")
expected_hash = "54f00400eb5c9c9c41c0a855b1d60bc3094672c842a405cd7d7cfca4af151952"
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_hash

rows = list(csv.DictReader(artifact.open(encoding="utf-8")))
expected = {
    "NICE": ("18.00%", "41.00%", "22.00%", "31.50%", "23.50%", "31.50%"),
    "ATM-S": ("46.00%", "76.00%", "52.00%", "62.50%", "50.50%", "65.00%"),
    "Our project": ("91.00%", "99.50%", "95.50%", "100.00%", "98.00%", "100.00%"),
}
columns = (
    "independent_top1", "independent_top5", "greedy_assignment_accuracy",
    "hungarian_assignment_accuracy", "stable_matching_assignment_accuracy",
    "sinkhorn_assignment_accuracy",
)
for row in rows:
    rendered = tuple(f"{float(row[column]):.2f}%" for column in columns)
    assert rendered == expected[row["model"]]

en = Path("README.md").read_text(encoding="utf-8")
zh = Path("README_ZH.md").read_text(encoding="utf-8")
for values in expected.values():
    for value in values:
        assert value in en
        assert value in zh
assert "89.09%" in en and "90.45%" in en
assert "89.09%" in zh and "90.45%" in zh
assert "65/90" not in en or "Sinkhorn" in en
assert Path("experiments/matching_fairness/README.md").is_file()
assert Path("experiments/matching_fairness/README_ZH.md").is_file()

for path in (Path("README.md"), Path("README_ZH.md")):
    headings = re.findall(r"^### (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert len(headings) == len(set(headings)), f"duplicate headings in {path}"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and line.endswith("|"):
            assert line.count("|") >= 3, f"malformed table row in {path}: {line}"
print("README matching-fairness verification passed")
PY
```

Expected: `README matching-fairness verification passed`.

- [ ] **Step 5: Run final formatting and scope checks**

Run:

```bash
git diff --check -- README.md README_ZH.md
git status --short --branch
git diff -- README.md README_ZH.md
```

Expected: no whitespace errors; only the two intended README files are modified beyond the already committed design and plan documents; the diff contains the bilingual summary, scope refinements, and no experimental artifact changes.

- [ ] **Step 6: Commit the bilingual README update**

```bash
git add README.md README_ZH.md
git commit -m "docs: summarize matching-fairness results"
```

Expected: the commit contains exactly `README.md` and `README_ZH.md`.
