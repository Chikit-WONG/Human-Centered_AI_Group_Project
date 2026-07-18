# SAMGA brain-rw Task 2 report

## Scope and commit

- Base: `ed293f032189305176d08c8e8bdb42c34d5e3620`
- Implementation commit: `360726c7af7a31b0defc88eec41668fec0913177`
- Runtime generation reads only the explicitly constructed
  `sub-01_train.json` through `sub-10_train.json` paths.
- Runtime subject-expanded files remain ignored under
  `artifacts/samga_brain_rw/protocol/manifests/`.
- Byte-identical compact copies are tracked under
  `experiments/samga_brain_rw/registries/protocol_v1/`.

## TDD evidence

Initial RED:

```text
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon \
  pytest -q experiments/samga_brain_rw/tests/test_splits.py

ModuleNotFoundError: No module named 'samga_brain_rw.hashing'
1 error in 0.75s
```

Subsequent focused RED cases locked the real integer subject IDs, GPFS
publication behavior, summary-last completion marker, concurrent exclusive
destination acquisition, distinct canonical-payload/file hashes, tracked
compact copies, and the Task 3-compatible role payload descriptor type.

Final GREEN:

```text
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon \
  pytest -q experiments/samga_brain_rw/tests/test_splits.py

17 passed in 22.89s
```

Final full experiment suite after the last descriptor regeneration:

```text
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon \
  pytest -q experiments/samga_brain_rw/tests

87 passed in 23.58s
```

## Generation and atomicity evidence

The approved Stage 0 command exited 0, and an immediate second invocation also
exited 0 through complete byte-identical reuse:

```text
env PYTHONPATH=experiments/samga_brain_rw conda run -n test python \
  experiments/samga_brain_rw/scripts/build_protocol_manifests.py \
  --protocol experiments/samga_brain_rw/configs/protocol_v1.json \
  --source-manifest-dir artifacts/samga_lora/manifests \
  --output-dir artifacts/samga_brain_rw/protocol/manifests
```

Verified output:

- 12 runtime files: one split assignment, ten subject sidecars, and one
  summary completion marker;
- roles: 1,254 train / 200 val-dev / 200 val-confirm concepts;
- rows per subject: 12,540 train / 200 val-dev queries / 200 val-confirm
  queries;
- ten subjects share one canonical record hash and split payload hash;
- each role descriptor exposes and recomputes `schema_version`,
  `payload_type`, `scope`, `source_records_sha256`, `ordered_ids_sha256`,
  `role_payload_sha256`, and `provenance_sha256`;
- `manifest_summary.json` is written and fsynced last.

GPFS rejects `renameat2(RENAME_NOREPLACE)` with `EINVAL`. The verified
GPFS-safe publication protocol therefore acquires the absent destination with
exclusive `os.mkdir`, writes every file with `xb`, fsyncs each file, writes the
hash-linked summary last, then fsyncs the output and parent directories.
Pre-existing, partial, conflicting, and race-created destinations are never
overwritten. A caught pre-summary failure cleans only the directory exclusively
created by that invocation; a crash leaves a detectable partial directory that
future runs refuse.

## Locked hashes

- Protocol semantic SHA-256:
  `0a9bb1dc750145ec94c35aaaddf5a834d303be3e6f69c9740237d9b967fd48bd`
- Shared canonical records SHA-256:
  `f59500f36e273f66fce5c2019670b076d75d538feccf296c7d7ed75f19ae3fac`
- Split-assignment canonical payload SHA-256:
  `4463e408af8644eed4c73a4d82832d402ba0b4f70b338f2e797216fd3698d912`
- `split_assignment.json` file SHA-256:
  `1d5ad2344797b359a3aeb04f1c298a7785b1492f2eee44e1d1a178e929ad70dc`
- `manifest_summary.json` file SHA-256:
  `bbd84cd87dda3ac5f02270a03923857e1e79c13a6acaa4c0a4d4556a1c413dce`
- Train concept-list SHA-256:
  `ae5aeda4101f8740ebcb63464ca9cf5e126c81b2f124f5caa8f7b57b7a9fad24`
- Val-dev concept-list SHA-256:
  `c8c00ff2b15d98cdcb74d533037d52435bc12e09797151e46b52b86aedba1d15`
- Val-dev query-list SHA-256:
  `512c222859a31b753ee31c5d6a1ddd1c81bb06e2dd5784d325f4480967162314`
- Val-confirm concept-list SHA-256:
  `27cfd5b3d0b46f3e8303953ede106e0716f410aee2b5756dd6fb5ad0324908bb`
- Val-confirm query-list SHA-256:
  `7a77db6d8d214e4a8192472dc7a760b58763d49c8c9f88fcb55c97bb124ec9fd`

The runtime and tracked compact file hashes were verified byte-identical after
the final idempotent generation pass.

Source train-manifest raw SHA-256 values, subjects 01 through 10:

```text
42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85
1d6275829da9f423c090d48350dfe106ac27759225265b9a3c796ddb4f77d0a0
123ba9dfdd983173fe6b5f6a739c515ca2b12ab101898549756a6d9a8462086e
eb133c98c761de61bb87154dd140df2f82047512fbd8e170a50e7cfaf005e7e5
f278c3a6efafeffc278b871ae111792fbb0cf41ee05cd11e6e24d3497afd7b6b
a88bdf485d0d05548c45ffda0b9fdbd9aad69207bcc88b258ea860da0d7244e8
12c6629989cf6b0fdf0aff963c0f690f21a2e46978b40aed54a39e3230d8d52b
703f9e305822da747c4fa5ee61c277578e5e7d3da42947bf2b17742909e3425d
6d30eca14797961805d3d113de2cbabbc448f1f3f83abb48f51c3565e440377b
abde70e302375e9ca3d94c5d2ce593e4be699fe817e17bdfc255112d8523483e
```
