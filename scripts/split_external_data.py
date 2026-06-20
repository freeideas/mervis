#!/usr/bin/env python
"""Split model_q4.onnx's single 3.63 GB external-data file into <2 GB shards.

Why: V8 caps a single ArrayBuffer at ~2 GB (verified: 2.0 GB allocates, 2.5 GB
throws "Array buffer allocation failed" -- in headless AND real Chrome, it's a
V8 limit). Transformers.js's `use_external_data_format: true` fetches the whole
model_q4.onnx_data into ONE Uint8Array, so our 3.63 GB blob can never load,
even though ORT-web's wasm heap itself grows to 4 GB.

Transformers.js's OTHER path -- `session_options.externalData = [{path, data}]`
-- fetches each listed file as a SEPARATE buffer. So we bin-pack the initializers
into a few shards, each under ~1.7 GB, rewrite each tensor's external-data
location to its shard, and emit a manifest the web app loads.

Run with: ./tmp/convenv/bin/python scripts/split_external_data.py
"""
import json
from pathlib import Path
import onnx
from onnx.external_data_helper import set_external_data

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "web" / "model" / "onnx"
MODEL = ONNX_DIR / "model_q4.onnx"
SHARD_TARGET = 1_700_000_000  # keep every shard well under the 2 GB ArrayBuffer cap
BASENAME = "model_q4.onnx_data"

print(f"loading {MODEL} (+ external data)...", flush=True)
m = onnx.load(str(MODEL), load_external_data=True)
inits = list(m.graph.initializer)

# only tensors with raw bytes get externalized; tiny inline tensors stay in-graph
big = [t for t in inits if t.raw_data]
total = sum(len(t.raw_data) for t in big)
print(f"{len(big)}/{len(inits)} tensors externalized, {total/1e9:.3f} GB total", flush=True)

# greedy bin-pack: largest first into the first shard with room
bucket_of, bucket_sizes = {}, []
for t in sorted(big, key=lambda t: len(t.raw_data), reverse=True):
    sz = len(t.raw_data)
    for i, bs in enumerate(bucket_sizes):
        if bs + sz <= SHARD_TARGET:
            bucket_of[t.name] = i
            bucket_sizes[i] += sz
            break
    else:
        bucket_of[t.name] = len(bucket_sizes)
        bucket_sizes.append(sz)

n = len(bucket_sizes)
print(f"{n} shards: " + ", ".join(f"{s/1e9:.2f} GB" for s in bucket_sizes), flush=True)

# write shards (graph order) + rewrite each tensor's external reference
for i in range(n):
    (ONNX_DIR / f"{BASENAME}_{i}").unlink(missing_ok=True)
files = {i: open(ONNX_DIR / f"{BASENAME}_{i}", "wb") for i in range(n)}
offsets = {i: 0 for i in range(n)}
for t in m.graph.initializer:
    if not t.raw_data:
        continue
    i = bucket_of[t.name]
    raw = t.raw_data
    files[i].write(raw)
    loc, off, ln = f"{BASENAME}_{i}", offsets[i], len(raw)
    offsets[i] += ln
    # set_external_data guards on raw_data being present, so set it BEFORE clearing
    set_external_data(t, location=loc, offset=off, length=ln)
    t.ClearField("raw_data")
for f in files.values():
    f.close()

# write the graph proto directly (external refs already set; avoid onnx.save re-inlining)
with open(MODEL, "wb") as f:
    f.write(m.SerializeToString())

shards = [f"{BASENAME}_{i}" for i in range(n)]
(ONNX_DIR / "external_data_manifest.json").write_text(json.dumps(shards, indent=2))

# drop the old single-file blob + its precompressed siblings
for p in (ONNX_DIR / BASENAME, ONNX_DIR / f"{BASENAME}.zst", ONNX_DIR / f"{BASENAME}.gz"):
    p.unlink(missing_ok=True)

print("manifest:", shards)
print("DONE. shards on disk:")
for i in range(n):
    p = ONNX_DIR / f"{BASENAME}_{i}"
    print(f"  {p.name}  {p.stat().st_size/1e9:.3f} GB")
