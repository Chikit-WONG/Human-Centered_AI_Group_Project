# Bilingual READMEs and `.gitignore` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide reciprocal English/Chinese documentation at both the repository root and the standalone SAMGA reproduction directory while keeping generated HPC artifacts out of Git.

**Architecture:** The root README pair remains the project-wide overview, while the SAMGA reproduction README pair owns detailed public-code reproduction instructions. The root documents link to the same-language reproduction guide. A minimal anchored ignore rule hides only local agent progress state; existing artifact, run, log, result, model-weight, cache, and dataset exclusions remain unchanged.

**Tech Stack:** Git ignore patterns, GitHub-flavored Markdown, Bash validation with `rg`, `git ls-files`, `stat`, `git check-ignore`, and `git diff --check`.

## Global Constraints

- Preserve all reported metrics, protocol distinctions, leakage warnings, commands, and references.
- Keep `artifacts/`, `runs/`, `logs/`, `results/`, model weights, NumPy arrays, caches, and SLURM output ignored.
- Do not ignore `experiments/`, `previous_work/`, or `docs/superpowers/`.
- Add only `/.superpowers/` to the ignore policy.
- Do not retrain models, move result artifacts, or modify the official SAMGA source.
- Do not stage unrelated pre-existing worktree files.

---

### Task 1: Ignore local agent progress state

**Files:**

- Modify: `.gitignore`
- Add: `docs/superpowers/plans/2026-07-18-bilingual-readmes-gitignore.md`

**Interfaces:**

- Consumes: existing generated-artifact ignore policy.
- Produces: an anchored rule for the repository-local `.superpowers/` state directory.

- [ ] **Step 1: Verify the desired rule is currently absent**

Run:

```bash
rg -n '^/\.superpowers/$' .gitignore
```

Expected: exit status `1` with no output.

- [ ] **Step 2: Verify current important decisions before editing**

Run:

```bash
git check-ignore -q artifacts/probe.npy
git check-ignore -q logs/probe.out
git check-ignore -q results/probe.json
git check-ignore -q runs/probe/checkpoint.pth
test "$(git check-ignore experiments/samga_reproduction/README.md 2>/dev/null || true)" = ""
```

Expected: all commands succeed; generated paths are ignored and the experiment README is not ignored.

- [ ] **Step 3: Add the minimal anchored rule**

Append this section after the editor/OS rules:

```gitignore
# Local agent workflow state
/.superpowers/
```

- [ ] **Step 4: Verify the complete policy**

Run:

```bash
git check-ignore -q .superpowers/sdd/progress.md
git check-ignore -q artifacts/probe.npy
git check-ignore -q logs/probe.out
git check-ignore -q results/probe.json
git check-ignore -q runs/probe/checkpoint.pth
test "$(git check-ignore experiments/samga_reproduction/README.md 2>/dev/null || true)" = ""
test "$(git check-ignore previous_work/clip_lora_baseline/README.md 2>/dev/null || true)" = ""
test "$(git check-ignore docs/superpowers/specs/2026-07-18-bilingual-readmes-gitignore-design.md 2>/dev/null || true)" = ""
```

Expected: every command succeeds.

- [ ] **Step 5: Commit only the ignore-policy change**

```bash
git add .gitignore docs/superpowers/plans/2026-07-18-bilingual-readmes-gitignore.md
git commit -m "chore: ignore local agent workflow state"
```

### Task 2: Create the standalone bilingual SAMGA reproduction guide

**Files:**

- Modify: `experiments/samga_reproduction/README.md`
- Create: `experiments/samga_reproduction/README_ZH.md`
- Add without content changes: `experiments/samga_reproduction/DOWNLOADER_SAFETY.md`
- Add without content changes: `experiments/samga_reproduction/V2_5_FEATURE_PIPELINE.md`
- Add without content changes: `experiments/samga_lora/README.md`

**Interfaces:**

- Consumes: the verified English reproduction guide and the canonical SAMGA metrics already recorded there.
- Produces: a reciprocal language pair with identical protocol scope and executable commands.

- [ ] **Step 1: Verify the Chinese guide and reciprocal links are absent**

Run:

```bash
test ! -e experiments/samga_reproduction/README_ZH.md
! rg -q '^\[?English.*简体中文' experiments/samga_reproduction/README.md
```

Expected: both commands succeed.

- [ ] **Step 2: Add the English-to-Chinese link**

Immediately below the English H1, add:

```markdown
English | [简体中文](README_ZH.md)
```

- [ ] **Step 3: Create the complete Chinese guide**

Start the new file with:

```markdown
# 经审计的 SAMGA 公共代码复现尝试

[English](README.md) | 简体中文
```

Translate the complete English guide without changing commands, paths,
hashes, model revisions, seeds, subject counts, or metric values. Use this
heading mapping in the same order:

```text
Scope and public-material ambiguities -> 范围与公开材料中的不确定性
Pinned source, model, data, and feature protocol -> 固定的源码、模型、数据与特征协议
Metric definitions and checkpoint-selection guardrails -> 指标定义与检查点选择边界
Verified results -> 已验证结果
Released-launcher-compatible seed-2025 confirmation -> 与发布启动器兼容的 seed-2025 验证
Project-defined five-seed stability grid -> 项目自定五随机种子稳定性网格
Reproduction workflow -> 复现流程
Sensitivity-only switches -> 仅用于敏感性分析的开关
Limitations -> 局限性
```

Preserve these canonical values verbatim in both languages:

```text
Paper: 91.30% / 98.80%
Seed 2025 fixed epoch 60: 89.55% / 98.65%
Seed 2025 test-selected: 91.95% / 98.95%
Patience-10 endpoint: 88.95% / 98.90%
Patience-10 test-selected: 91.50% / 98.75%
Project seeds 42–46 fixed epoch 60: 89.02% ± 0.36 / 98.87% ± 0.06
Project seeds 42–46 test-selected: 91.82% ± 0.20 / 98.87% ± 0.16
```

The Chinese text must explicitly retain all four boundaries:

```text
InternViT V2.5 and patch-mean semantics are inferred, not author-confirmed.
Seeds 42–46 are project-defined, not the paper's undisclosed seeds.
Test-selected metrics inspect the formal test set every epoch.
Patience-10 endpoints are also test-conditioned because early stopping monitors formal-test Top-1.
```

- [ ] **Step 4: Verify structural and numerical parity**

Run:

```bash
rg -q 'English \| \[简体中文\]\(README_ZH\.md\)' experiments/samga_reproduction/README.md
rg -q '\[English\]\(README\.md\) \| 简体中文' experiments/samga_reproduction/README_ZH.md
for value in \
  '91.30%' '98.80%' \
  '89.55%' '98.65%' \
  '91.95%' '98.95%' \
  '88.95%' '98.90%' \
  '91.50%' '98.75%' \
  '89.02% ± 0.36' '98.87% ± 0.06' \
  '91.82% ± 0.20' '98.87% ± 0.16'; do
  rg -Fq "$value" experiments/samga_reproduction/README.md
  rg -Fq "$value" experiments/samga_reproduction/README_ZH.md
done
test "$(rg -c '^```' experiments/samga_reproduction/README.md)" -eq "$(rg -c '^```' experiments/samga_reproduction/README_ZH.md)"
```

Expected: all commands succeed and the code-fence counts match.

- [ ] **Step 5: Commit the standalone bilingual guide**

```bash
git add \
  experiments/samga_reproduction/README.md \
  experiments/samga_reproduction/README_ZH.md \
  experiments/samga_reproduction/DOWNLOADER_SAFETY.md \
  experiments/samga_reproduction/V2_5_FEATURE_PIPELINE.md \
  experiments/samga_lora/README.md
git commit -m "docs: add bilingual SAMGA reproduction guide"
```

### Task 3: Improve the root bilingual navigation

**Files:**

- Modify: `README.md`
- Modify: `README_ZH.md`

**Interfaces:**

- Consumes: the standalone README pair from Task 2.
- Produces: same-language links from each root project overview while preserving the existing root reciprocal link.

- [ ] **Step 1: Verify current root language links and the missing Chinese target**

Run:

```bash
rg -q '^English \| \[简体中文\]\(README_ZH\.md\)' README.md
rg -q '^\[English\]\(README\.md\) \| 简体中文' README_ZH.md
! rg -q 'experiments/samga_reproduction/README_ZH\.md' README_ZH.md
```

Expected: all commands succeed.

- [ ] **Step 2: Add a prominent same-language reproduction-guide callout**

After the existing scope block in `README.md`, add:

```markdown
> **Reproduction guide.** For pinned assets, executable commands, complete
> protocol details, and claim boundaries, see the
> [English SAMGA public-code reproduction guide](experiments/samga_reproduction/README.md).
```

After the corresponding scope block in `README_ZH.md`, add:

```markdown
> **复现指南。** 固定资产、可执行命令、完整协议细节和声明边界见
> [SAMGA 公共代码复现中文指南](experiments/samga_reproduction/README_ZH.md)。
```

- [ ] **Step 3: Point later Chinese SAMGA-guide links to the Chinese file**

In `README_ZH.md`, replace each link target
`experiments/samga_reproduction/README.md` with
`experiments/samga_reproduction/README_ZH.md`. Keep the English root targets
on `experiments/samga_reproduction/README.md`.

- [ ] **Step 4: Verify root parity and local targets**

Run:

```bash
rg -q 'experiments/samga_reproduction/README\.md' README.md
! rg -q 'experiments/samga_reproduction/README_ZH\.md' README.md
rg -q 'experiments/samga_reproduction/README_ZH\.md' README_ZH.md
! rg -q 'experiments/samga_reproduction/README\.md' README_ZH.md
test -f experiments/samga_reproduction/README.md
test -f experiments/samga_reproduction/README_ZH.md
```

Expected: all commands succeed.

- [ ] **Step 5: Commit only the root README pair**

```bash
git add README.md README_ZH.md
git commit -m "docs: link project READMEs to bilingual SAMGA guides"
```

### Task 4: Final documentation and ignore-policy verification

**Files:**

- Verify: `.gitignore`
- Verify: `README.md`
- Verify: `README_ZH.md`
- Verify: `experiments/samga_reproduction/README.md`
- Verify: `experiments/samga_reproduction/README_ZH.md`

**Interfaces:**

- Consumes: Tasks 1–3.
- Produces: evidence that generated output remains ignored and both language pairs render and navigate correctly.

- [ ] **Step 1: Check whitespace, placeholders, and code fences**

Run:

```bash
BASE="$(git merge-base main HEAD)"
git diff --check "$BASE"..HEAD
rg -n 'TODO|TBD|PLACEHOLDER|FINAL_|/fixed/local/model' \
  README.md README_ZH.md \
  experiments/samga_reproduction/README.md \
  experiments/samga_reproduction/README_ZH.md && exit 1 || true
for file in README.md README_ZH.md experiments/samga_reproduction/README.md experiments/samga_reproduction/README_ZH.md; do
  count="$(rg -c '^```' "$file")"
  test $((count % 2)) -eq 0
done
```

Expected: no whitespace errors, no stale markers, and every fence count is even.

- [ ] **Step 2: Validate all local Markdown/image links**

Run:

```bash
python - <<'PY'
from pathlib import Path
from urllib.parse import unquote
import re

files = [
    Path("README.md"),
    Path("README_ZH.md"),
    Path("experiments/samga_reproduction/README.md"),
    Path("experiments/samga_reproduction/README_ZH.md"),
]
missing = []
for source in files:
    text = source.read_text(encoding="utf-8")
    for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = raw.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = unquote(target.split("#", 1)[0])
        if relative and not (source.parent / relative).exists():
            missing.append(f"{source}: {target}")
if missing:
    raise SystemExit("Missing local links:\n" + "\n".join(missing))
print(f"Validated local links in {len(files)} Markdown files")
PY
```

Expected: `Validated local links in 4 Markdown files`.

- [ ] **Step 3: Recheck Git visibility and large files**

Run:

```bash
git check-ignore -q .superpowers/sdd/progress.md
git check-ignore -q artifacts/probe.npy
git check-ignore -q runs/probe/checkpoint.pth
git check-ignore -q logs/probe.err
git check-ignore -q results/probe.csv
test "$(git check-ignore experiments/samga_reproduction/README.md 2>/dev/null || true)" = ""
test "$(git check-ignore experiments/samga_reproduction/README_ZH.md 2>/dev/null || true)" = ""
oversized=()
while IFS= read -r -d '' file; do
  if test -f "$file" && test "$(stat -c %s -- "$file")" -gt 10485760; then
    oversized+=("$file")
  fi
done < <(git ls-files --cached --others --exclude-standard -z -- experiments)
if test "${#oversized[@]}" -ne 0; then
  printf 'Unignored experiment file exceeds 10 MiB:\n' >&2
  printf '  %s\n' "${oversized[@]}" >&2
  exit 1
fi
```

Expected: all generated paths are ignored, both guides remain visible to Git,
and no unignored experiment file exceeds 10 MiB.

- [ ] **Step 4: Inspect the final scoped history and worktree**

Run:

```bash
BASE="$(git merge-base main HEAD)"
git log --oneline "$BASE"..HEAD
git status --short
```

Expected: the design and all implementation/review-fix commits are visible; unrelated
pre-existing untracked experiment files remain untouched.
