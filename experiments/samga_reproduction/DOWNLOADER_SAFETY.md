# InternViT V2.5 downloader safety

Use `download_v2_5_safe.py` for every new or resumed InternViT V2.5 model
download:

```bash
python experiments/samga_reproduction/download_v2_5_safe.py \
  --output-dir /path/to/model
```

`download_v2_5.py` is now a fail-fast compatibility stub that refuses to
download and directs callers here. The original implementation was renamed to
`download_v2_5_legacy_unsafe.py` for audit only; do not execute it because it
may expose unverified preallocated shards and lacks the complete pinned
metadata and HTTP range checks.

The safe downloader:

- invalidates an earlier complete provenance record before doing download work;
- rejects every pre-existing symlink directly under the output directory;
- writes shards and small files to `.partial` files and publishes them with
  `os.replace` only after exact size and SHA-256 verification;
- content-locks all six small files to revision
  `9d1a4344077479c93d42584b6941c64d795d508d`, including mirror responses;
- requires exact `Content-Range` start, end, and total values, rejects excess
  response bytes, and handles short `os.pwrite` calls; and
- atomically publishes `model_provenance.json` with `complete: true` only after
  every artifact passes final verification.

An interrupted download can leave regular `.partial` files. A subsequent safe
run removes and rebuilds them. It never treats a `.partial` file as a completed
model artifact.
