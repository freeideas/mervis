#!/usr/bin/env python
"""Greedy-decode a few tokens from the converted q4f16 ONNX model on CPU.

Proves the exported model runs end-to-end (4-bit MatMulNBits + fp16 + KV cache)
and that the fine-tune still emits <Mervin>/<Mervis> tags, before we trust it in
the browser. CPU + fp16 is slow; this is a smoke test, not a benchmark.
"""
import sys
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "web/model"
ONNX = "web/model/onnx/model_q4.onnx"
N_LAYERS, N_KV, HEAD_DIM = 32, 8, 128
EOS = {199999, 200020}
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 60

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]

ids = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True, return_tensors="np",
).astype(np.int64)

seqlen = ids.shape[1]
# q4 model runs fp32 activations -> KV cache is fp32 (fp16 for a q4f16 build).
past = {
    f"past_key_values.{i}.{kv}": np.zeros((1, N_KV, 0, HEAD_DIM), np.float32)
    for i in range(N_LAYERS) for kv in ("key", "value")
}
cur = ids
total = seqlen
generated = []

print(f"prompt: {PROMPT!r}\nprompt tokens: {seqlen}\ngenerating (CPU, be patient)...\n", flush=True)
for step in range(MAX_NEW):
    feeds = {
        "input_ids": cur,
        "attention_mask": np.ones((1, total), np.int64),
        "position_ids": (np.arange(total, dtype=np.int64)[None]
                         if step == 0 else np.array([[total - 1]], np.int64)),
        **past,
    }
    outs = sess.run(None, feeds)
    logits = outs[0]
    nxt = int(logits[0, -1].argmax())
    generated.append(nxt)
    if nxt in EOS:
        break
    past = {n.replace("present", "past_key_values"): outs[i]
            for i, n in enumerate(out_names) if n.startswith("present")}
    cur = np.array([[nxt]], np.int64)
    total += 1
    print(tok.decode([nxt]), end="", flush=True)

print("\n\n--- full decode ---")
print(tok.decode(generated, skip_special_tokens=False))
