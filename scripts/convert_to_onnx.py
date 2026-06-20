#!/usr/bin/env python
"""Convert the merged Mervin/Mervis Phi-4-mini model to ONNX q4f16 for Transformers.js.

Pipeline:
  1. optimum-cli export onnx  (fp32, with KV-cache)        -> tmp/onnx_fp32/model.onnx
  2. 4-bit weight-only quantize (MatMulNBits, block 32)
     + cast the remaining fp32 to fp16                     -> web/model/onnx/model_q4f16.onnx
  3. copy tokenizer/config into web/model/

The output dir (web/model/) is what Transformers.js loads same-origin:
  web/model/config.json, tokenizer.json, ... + onnx/model_q4f16.onnx(+ .onnx_data)

Run with the conversion venv:
  ./tmp/convenv/bin/python scripts/convert_to_onnx.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dl" / "mervis-merged"
WORK = ROOT / "tmp" / "onnx_fp32"
OUT = ROOT / "web" / "model"
ONNX_OUT = OUT / "onnx"
VENV = ROOT / "tmp" / "convenv" / "bin"

TOKENIZER_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json",
]


def export_fp32():
    if (WORK / "model.onnx").exists():
        print("[1/3] fp32 ONNX already exists -> skipping export")
        return
    print("[1/3] exporting fp32 ONNX (text-generation-with-past, CPU)...", flush=True)
    subprocess.run(
        [
            str(VENV / "optimum-cli"), "export", "onnx",
            "--model", str(SRC),
            "--task", "text-generation-with-past",
            "--framework", "pt",
            str(WORK),
        ],
        check=True,
    )


def quantize():
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        QuantFormat,
    )
    from onnxconverter_common import float16

    src = WORK / "model.onnx"
    print(f"[2/3] loading {src} ...", flush=True)
    model = onnx.load(str(src), load_external_data=True)

    # NOTE: we deliberately do NOT cast to fp16. Phi-3's RMSNorm runs in an fp32
    # island (it upcasts for stability), and onnxconverter_common's
    # convert_float_to_float16 half-converts that island -> a layernorm Add with
    # mixed fp32/fp16 operands that onnxruntime refuses to load. So we ship q4
    # (4-bit weights, fp32 activations): bigger + a touch slower than q4f16, but
    # correct and well-supported by Transformers.js (dtype:'q4'). q4f16 is a
    # later size optimization (needs proper float16 op/node block-lists).
    print("      4-bit weight-only quantize (block_size=32, symmetric)...", flush=True)
    quant = MatMulNBitsQuantizer(
        model,
        bits=4,
        block_size=32,
        is_symmetric=True,
        quant_format=QuantFormat.QOperator,
    )
    quant.process()
    qmodel = quant.model.model if hasattr(quant.model, "model") else quant.model

    ONNX_OUT.mkdir(parents=True, exist_ok=True)
    out = ONNX_OUT / "model_q4.onnx"
    print(f"[3/3] saving {out} (+ external data)...", flush=True)
    # remove stale outputs so external-data save starts clean
    for p in (out, ONNX_OUT / "model_q4.onnx_data"):
        if p.exists():
            p.unlink()
    onnx.save(
        qmodel,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_q4.onnx_data",
        convert_attribute=False,
    )


def assemble():
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TOKENIZER_FILES:
        s = SRC / name
        if s.exists():
            shutil.copy2(s, OUT / name)
            copied.append(name)
    print("      copied tokenizer/config:", ", ".join(copied))


def main():
    export_fp32()
    quantize()
    assemble()
    size = sum(f.stat().st_size for f in ONNX_OUT.glob("*")) / 1e9
    print(f"\nDONE. web/model/onnx total = {size:.2f} GB")
    print("Files:")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
