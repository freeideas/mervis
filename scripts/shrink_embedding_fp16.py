#!/usr/bin/env python
"""Cast the fp32 embedding table to fp16 in the q4 ONNX model, in place.

Why: ONNX Runtime Web runs in a 32-bit WASM heap (~4 GB address-space ceiling).
ORT reads the whole model into that heap before handing tensors to the WebGPU
backend, so our 4.86 GB model_q4.onnx_data fails to load with
"Array buffer allocation failed" / "Module.MountedFiles is not available".

The single biggest piece is model.embed_tokens.weight — [200064, 3072] fp32 =
2.458 GB — which MatMulNBitsQuantizer leaves untouched because it's consumed by
a Gather (embedding lookup), not a MatMul. Phi-4-mini's o200k vocab (200 064
tokens) makes this embedding enormous.

We surgically cast ONLY that initializer to fp16 (2.458 GB -> 1.23 GB) and insert
a Cast(->FLOAT) right after its Gather, so the rest of the graph stays fp32 and
the fragile fp32 RMSNorm islands (which broke full float16 conversion) are never
touched. Result: ~3.63 GB, comfortably under the WASM ceiling.
"""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "web" / "model" / "onnx" / "model_q4.onnx"
DATA = ROOT / "web" / "model" / "onnx" / "model_q4.onnx_data"

print(f"loading {MODEL} (+ external data)...", flush=True)
m = onnx.load(str(MODEL), load_external_data=True)
g = m.graph

emb = next(t for t in g.initializer if t.name.endswith("embed_tokens.weight"))
print(f"embedding: {emb.name} dims={list(emb.dims)} "
      f"dtype={TensorProto.DataType.Name(emb.data_type)}", flush=True)
if emb.data_type == TensorProto.FLOAT16:
    print("already fp16 -> nothing to do"); sys.exit(0)

arr = numpy_helper.to_array(emb).astype(np.float16)
new = numpy_helper.from_array(arr, emb.name)
g.initializer.remove(emb)
g.initializer.append(new)
print(f"cast embedding -> fp16 ({arr.nbytes/1e9:.3f} GB)", flush=True)

consumers = [n for n in g.node if emb.name in n.input]
assert len(consumers) == 1, f"expected 1 consumer, got {[n.name for n in consumers]}"
node = consumers[0]
print(f"consumer: {node.op_type} {node.name} out={node.output[0]}", flush=True)

orig_out = node.output[0]
fp16_out = orig_out + "_fp16"
node.output[0] = fp16_out
cast = helper.make_node("Cast", [fp16_out], [orig_out],
                        to=TensorProto.FLOAT, name="embed_cast_to_fp32")
idx = list(g.node).index(node)
g.node.insert(idx + 1, cast)
# drop any stale value_info for orig_out (it was fp32 before; Cast re-supplies fp32)
keep = [vi for vi in g.value_info if vi.name != orig_out]
del g.value_info[:]
g.value_info.extend(keep)

# rewrite external data cleanly
DATA.unlink(missing_ok=True)
print(f"saving {MODEL} ...", flush=True)
onnx.save(m, str(MODEL), save_as_external_data=True,
          all_tensors_to_one_file=True, location="model_q4.onnx_data",
          convert_attribute=False)
sz = DATA.stat().st_size / 1e9
print(f"DONE. model_q4.onnx_data = {sz:.3f} GB", flush=True)
