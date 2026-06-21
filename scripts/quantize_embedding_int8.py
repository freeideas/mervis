#!/usr/bin/env python
"""Quantize the embedding table to per-row int8, in place, single-file output.

Why: ORT's WebGPU backend uploads each weight tensor as ONE GPU buffer. The
fp16 embedding ([200064, 3072] = 1.23 GB) exceeds the `maxBufferSize` of
software/low-end adapters (e.g. SwiftShader caps at 1.07 GB), so the first
generate() throws "Buffer size 1229193216 exceeds the max buffer size limit".
int8 halves it to ~0.61 GB -- under even 1 GB-class limits -- with negligible
quality loss (per-row symmetric quantization is standard for embeddings).

Graph rewrite (the embedding is consumed by one Gather, then a Cast->fp32):
  before:  Gather(embed_fp16, ids) -> Cast(->fp32) -> out
  after:   Gather(embed_int8,  ids) -> Cast(->fp32) --\\
           Gather(embed_scale, ids) ------------------- Mul -> out
embed_scale is [V,1] so its gather is [B,S,1] and broadcasts over H -- no
Unsqueeze needed.

Run: ./tmp/convenv/bin/python scripts/quantize_embedding_int8.py
Then re-shard (split_external_data.py) and re-compress.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "web" / "model" / "onnx"
MODEL = ONNX_DIR / "model_q4.onnx"

print(f"loading {MODEL} (+ external data)...", flush=True)
m = onnx.load(str(MODEL), load_external_data=True)
g = m.graph

emb = next(t for t in g.initializer if t.name.endswith("embed_tokens.weight"))
print(f"embedding: {emb.name} dims={list(emb.dims)} "
      f"dtype={TensorProto.DataType.Name(emb.data_type)}", flush=True)

x = numpy_helper.to_array(emb).astype(np.float32)        # [V, H]
absmax = np.abs(x).max(axis=1, keepdims=True)            # [V, 1]
absmax[absmax == 0] = 1.0
scale = (absmax / 127.0).astype(np.float32)              # [V, 1]
q = np.round(x / scale).clip(-127, 127).astype(np.int8)  # [V, H]
err = np.abs(q.astype(np.float32) * scale - x).max()
print(f"int8 quantized: q={q.nbytes/1e9:.3f} GB, scale={scale.nbytes/1e6:.1f} MB, "
      f"max abs dequant err={err:.4g}", flush=True)

int8_name = "embed_tokens.int8"
scale_name = "embed_tokens.scale"
g.initializer.remove(emb)
g.initializer.append(numpy_helper.from_array(q, int8_name))
g.initializer.append(numpy_helper.from_array(scale, scale_name))

# locate the Gather that read the embedding, and the Cast that followed it
gather = next(n for n in g.node if n.op_type == "Gather" and emb.name in n.input)
ids = gather.input[1]
gather_out = gather.output[0]
cast = next(n for n in g.node if n.op_type == "Cast" and n.input and n.input[0] == gather_out)
final_out = cast.output[0]   # the fp32 name the rest of the graph consumes
print(f"gather={gather.name} ids={ids} -> {gather_out}; cast -> {final_out}", flush=True)

# rewire: data-gather now reads int8 table; reuse its output as the int8 path
gather.input[0] = int8_name                 # data tensor -> int8
int8_gathered = gather_out                  # [B,S,H] int8
i8_f32 = int8_gathered + "_f32"
cast.input[0] = int8_gathered               # Cast int8 -> fp32
cast.output[0] = i8_f32

scale_gathered = "embed_scale_gathered"      # [B,S,1] fp32
# insert right after the Cast so the graph stays topologically ordered
# (downstream consumers of final_out appear later in the node list)
cast_idx = list(g.node).index(cast)
g.node.insert(cast_idx + 1, helper.make_node(
    "Gather", [scale_name, ids], [scale_gathered], name="embed_scale_gather"))
g.node.insert(cast_idx + 2, helper.make_node(
    "Mul", [i8_f32, scale_gathered], [final_out], name="embed_dequant_mul"))

# remove old single/sharded external data, write a fresh single-file blob
for p in list(ONNX_DIR.glob("model_q4.onnx_data*")):
    p.unlink()
print(f"saving {MODEL} (single-file external data)...", flush=True)
onnx.save(m, str(MODEL), save_as_external_data=True, all_tensors_to_one_file=True,
          location="model_q4.onnx_data", convert_attribute=False)
sz = (ONNX_DIR / "model_q4.onnx_data").stat().st_size / 1e9
print(f"DONE. model_q4.onnx_data = {sz:.3f} GB", flush=True)
