#!/usr/bin/env python
"""Generate colab/mervis_build.ipynb -- the all-in-one Colab notebook.

Run: python colab/build_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(*lines):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": list(_split(lines))})


def code(*lines):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": list(_split(lines)),
    })


def _split(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


# ---------------------------------------------------------------------------
md(
    "# Mervis -- all-in-one Colab build (Phase 1 + 2 + 3)",
    "",
    "Fine-tune **microsoft/Phi-4-mini-instruct** into the two-headed robot",
    "(**Mervin** the gloomy one, **Mervis** the cheerful one), convert it to run",
    "**entirely in the browser** (ONNX q4f16 / WebGPU / Transformers.js), and emit a",
    "ready-to-serve `web/` folder -- then hand it all to Google Drive so you can pull",
    "it down to your own machine and serve it with Caddy.",
    "",
    "**This notebook is self-contained.** It does not depend on `web/`, `img/`, or",
    "`scripts/` being in the GitHub repo (they aren't) -- everything the browser app",
    "needs lives in `colab/assets/` and `colab/scripts/`, which this notebook clones.",
    "",
    "### How to run",
    "1. `Runtime -> Change runtime type ->` pick a **GPU** + **High-RAM**. On paid Colab,",
    "   an **A100 (High-RAM)** sails through the merge and the ~17.8 GB fp32 ONNX export",
    "   with room to spare; a **T4 (High-RAM)** also works. With High-RAM you can skip the",
    "   swap cell (2.1) entirely.",
    "2. `Runtime -> Run all`.",
    "3. Approve the Google Drive OAuth popup near the end.",
    "4. Pull `MyDrive/mervis-web/` down to your machine (see the last cell).",
    "",
    "### What you get on Drive",
    "| Drive path | size | what |",
    "|---|---|---|",
    "| `MyDrive/mervis-web/` | ~2.2 GB | the **whole static site** -- copy this into the project's `web/` and serve |",
    "| `MyDrive/mervis-web.zip` | ~2.2 GB | same thing, one file (easier to download) |",
    "| `MyDrive/mervis-lora/` | 58 MB | the LoRA adapter (tiny, irreplaceable -- the merged model is just base+this) |",
    "| `MyDrive/mervis-merged/` | 7.7 GB | merged fp16 model (optional backup; you only *need* it to re-convert) |",
)

md("## Phase 1 -- fine-tune")

md("### 1.1 Confirm a GPU is attached")
code(
    "import torch",
    "assert torch.cuda.is_available(), (",
    "    'No GPU. Runtime -> Change runtime type -> GPU (T4), then Run all again.')",
    "print('GPU:', torch.cuda.get_device_name(0))",
)

md(
    "### 1.2 Install the (pinned) training stack",
    "Same versions that trained the model end-to-end on the VM. `trl` 0.14.0 because",
    "transformers 4.49 needs a trl without the `<4.47` cap.",
)
code(
    "%pip install -q \\",
    '  "transformers==4.49.0" \\',
    '  "trl==0.14.0" \\',
    '  "peft==0.14.0" \\',
    '  "accelerate==1.3.0" \\',
    '  "bitsandbytes==0.45.3" \\',
    '  "datasets==3.2.0" \\',
    '  "sentencepiece" \\',
    '  "tiktoken"',
)

md(
    "### 1.3 Clone the repo (dataset + browser assets + convert script)",
    "Pulls `mervin_mervis_finetune.csv` and the `colab/` folder onto the VM. Make sure",
    "you've **pushed `colab/` to GitHub** before running, or this clone won't have the",
    "web assets it needs in Phase 3.",
)
code(
    "import os",
    "from datasets import load_dataset",
    "",
    "REPO_DIR = '/content/mervis'",
    "if not os.path.isdir(REPO_DIR):",
    "    !git clone --depth 1 https://github.com/freeideas/mervis.git {REPO_DIR}",
    "",
    "CSV_PATH = f'{REPO_DIR}/mervin_mervis_finetune.csv'",
    "ASSETS   = f'{REPO_DIR}/colab/assets'",
    "CONVERT  = f'{REPO_DIR}/colab/scripts/convert_to_onnx.py'",
    "for p in (CSV_PATH, ASSETS, CONVERT):",
    "    assert os.path.exists(p), f'missing {p} -- did you push colab/ to GitHub?'",
    "",
    "raw = load_dataset('csv', data_files=CSV_PATH, split='train')",
    "print(raw)",
    "print(raw[0])",
)

md("### 1.4 Load Phi-4-mini in 4-bit (QLoRA base)")
code(
    "import torch",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
    "",
    "BASE_MODEL = 'microsoft/Phi-4-mini-instruct'",
    "",
    "bnb_config = BitsAndBytesConfig(",
    "    load_in_4bit=True,",
    "    bnb_4bit_quant_type='nf4',",
    "    bnb_4bit_compute_dtype=torch.float16,",
    "    bnb_4bit_use_double_quant=True,",
    ")",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)",
    "if tokenizer.pad_token is None:",
    "    tokenizer.pad_token = tokenizer.eos_token",
    "tokenizer.padding_side = 'right'",
    "",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    BASE_MODEL,",
    "    quantization_config=bnb_config,",
    "    device_map='auto',",
    "    trust_remote_code=True,",
    "    torch_dtype=torch.float16,",
    "    attn_implementation='eager',",
    ")",
    "model.config.use_cache = False",
    "print('Loaded', BASE_MODEL)",
)

md(
    "### 1.5 Render each row into the Phi-4 chat template",
    "Let the tokenizer build `<|user|> ... <|assistant|> ...` so it matches exactly what",
    "Phi-4-mini expects. The full `response` (both `<Mervin>` and `<Mervis>` tags) is the",
    "assistant turn.",
)
code(
    "def to_text(example):",
    "    messages = [",
    "        {'role': 'user', 'content': example['prompt']},",
    "        {'role': 'assistant', 'content': example['response']},",
    "    ]",
    "    text = tokenizer.apply_chat_template(",
    "        messages, tokenize=False, add_generation_prompt=False)",
    "    return {'text': text}",
    "",
    "dataset = raw.map(to_text, remove_columns=raw.column_names)",
    "print(dataset[0]['text'])",
)

md("### 1.6 LoRA config")
code(
    "from peft import LoraConfig, prepare_model_for_kbit_training",
    "",
    "model = prepare_model_for_kbit_training(model)",
    "",
    "peft_config = LoraConfig(",
    "    r=16,",
    "    lora_alpha=32,",
    "    lora_dropout=0.05,",
    "    bias='none',",
    "    task_type='CAUSAL_LM',",
    "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',",
    "                    'gate_proj', 'up_proj', 'down_proj'],",
    ")",
)

md(
    "### 1.7 Train (~262 examples, 3 epochs)",
    "Roughly 10-20 min on a T4.",
)
code(
    "from trl import SFTTrainer, SFTConfig",
    "",
    "ADAPTER_DIR = '/content/mervis-lora'",
    "",
    "sft_config = SFTConfig(",
    "    output_dir=ADAPTER_DIR,",
    "    num_train_epochs=3,",
    "    per_device_train_batch_size=1,",
    "    gradient_accumulation_steps=8,",
    "    learning_rate=2e-4,",
    "    lr_scheduler_type='cosine',",
    "    warmup_ratio=0.03,",
    "    logging_steps=10,",
    "    save_strategy='epoch',",
    "    optim='paged_adamw_8bit',",
    "    fp16=True,",
    "    max_seq_length=1024,",
    "    dataset_text_field='text',",
    "    packing=False,",
    "    report_to='none',",
    ")",
    "",
    "trainer = SFTTrainer(",
    "    model=model,",
    "    args=sft_config,",
    "    train_dataset=dataset,",
    "    peft_config=peft_config,",
    "    processing_class=tokenizer,",
    ")",
    "",
    "trainer.train()",
    "trainer.save_model(ADAPTER_DIR)",
    "tokenizer.save_pretrained(ADAPTER_DIR)",
    "print('Saved LoRA adapters to', ADAPTER_DIR)",
)

md("### 1.8 Smoke test -- both personas should appear")
code(
    "from transformers import pipeline",
    "",
    "gen = pipeline('text-generation', model=trainer.model, tokenizer=tokenizer)",
    "prompt = tokenizer.apply_chat_template(",
    "    [{'role': 'user', 'content': 'What is the capital of France?'}],",
    "    tokenize=False, add_generation_prompt=True)",
    "out = gen(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']",
    "print(out[len(prompt):])",
)

md(
    "### 1.9 Merge the LoRA into the base weights",
    "Reload the base in fp16 (no quant), apply adapters, merge. Self-contained: reloads",
    "adapters from disk, so it survives a kernel restart after training.",
    "",
    "> **On a standard (free) runtime** the fp16 model is ~7.6 GB vs ~13 GB RAM, so this",
    "> can be slow and the progress bar may look stuck while it's actually working. It",
    "> finishes. A **High-RAM** runtime makes it quick.",
)
code(
    "import gc, torch",
    "from peft import PeftModel",
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "",
    "BASE_MODEL  = 'microsoft/Phi-4-mini-instruct'",
    "ADAPTER_DIR = '/content/mervis-lora'",
    "MERGED_DIR  = '/content/mervis-merged'",
    "",
    "# Free everything pinning the GPU before the fp16 load (the smoke-test `gen`",
    "# pipeline holds the old 4-bit model). pop() = no error if already gone.",
    "for _n in ['gen', 'trainer', 'model', 'merged', 'base']:",
    "    globals().pop(_n, None)",
    "gc.collect(); torch.cuda.empty_cache()",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)",
    "base = AutoModelForCausalLM.from_pretrained(",
    "    BASE_MODEL, torch_dtype=torch.float16, device_map={'': 0},",
    "    trust_remote_code=True)",
    "merged = PeftModel.from_pretrained(base, ADAPTER_DIR)",
    "merged = merged.merge_and_unload()",
    "merged.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size='2GB')",
    "tokenizer.save_pretrained(MERGED_DIR)",
    "print('Merged model saved to', MERGED_DIR)",
    "",
    "# Drop it from RAM -- conversion (next phase) reads it from disk in a separate venv.",
    "for _n in ['merged', 'base']:",
    "    globals().pop(_n, None)",
    "gc.collect(); torch.cuda.empty_cache()",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 2 -- convert to browser ONNX (q4f16)",
    "",
    "Conversion needs a *different*, conflicting toolchain (optimum / onnxruntime /",
    "onnx_ir / onnxconverter_common) from the training stack, so we do it in a throwaway",
    "venv and run `colab/scripts/convert_to_onnx.py` (the VM's pipeline) as a subprocess.",
    "Nothing here disturbs the training kernel.",
)

md(
    "### 2.1 (Skip on High-RAM) add swap",
    "The fp32 ONNX export is ~17.8 GB on disk and RAM-hungry. On a **High-RAM** runtime",
    "(what you'll use on paid Colab) you don't need this -- **skip this cell**. It's here",
    "only as a safety net for a low-RAM box: it adds a 32 GB swapfile (best-effort -- some",
    "Colab kernels block `swapon`). A third option if memory is ever tight is the convert",
    "script's `FP16_GPU_EXPORT` path, which skips the fp32 monster entirely.",
)
code(
    "import subprocess",
    "try:",
    "    subprocess.run('fallocate -l 32G /content/swapfile', shell=True, check=True)",
    "    subprocess.run('chmod 600 /content/swapfile', shell=True, check=True)",
    "    subprocess.run('mkswap /content/swapfile', shell=True, check=True)",
    "    subprocess.run('swapon /content/swapfile', shell=True, check=True)",
    "    print('swap on:')",
    "    print(subprocess.run('free -h', shell=True, capture_output=True, text=True).stdout)",
    "except Exception as e:",
    "    print('swap setup skipped/failed (ok):', e)",
)

md(
    "### 2.2 Build the conversion venv",
    "Isolated venv with CPU-only torch (the export runs on CPU) plus the ONNX toolchain.",
    "These are intentionally unpinned -- `optimum` split ONNX export into `optimum-onnx`",
    "in 2.x, and the toolchain drifts; latest generally works. If export later fails on a",
    "version mismatch, that's the first knob to turn.",
)
code(
    "%%bash",
    "set -e",
    "python -m venv /content/convenv",
    "/content/convenv/bin/pip install -q --upgrade pip",
    "/content/convenv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu",
    "/content/convenv/bin/pip install -q \\",
    "  optimum-onnx onnx onnxruntime onnxconverter_common onnx_ir \\",
    "  transformers accelerate sentencepiece tiktoken protobuf",
    "echo 'convenv ready'",
)

md(
    "### 2.3 Run the conversion",
    "`mervis-merged` -> fp32 ONNX -> fp16 cast -> 4-bit MatMulNBits -> `web/model/onnx/`.",
    "Expect many *'will be truncated'* warnings during the fp16 cast -- harmless. This is",
    "the slow cell (export + quantize).",
)
code(
    "import subprocess, sys",
    "cmd = ['/content/convenv/bin/python', CONVERT,",
    "       '/content/mervis-merged', '/content/web/model']",
    "print('running:', ' '.join(cmd), flush=True)",
    "p = subprocess.run(cmd)",
    "assert p.returncode == 0, 'conversion failed -- see output above'",
)

md(
    "### 2.4 Sanity-generate from the converted model (CPU)",
    "Proves the q4f16 ONNX actually runs (4-bit MatMulNBits + fp16 + KV cache) and that",
    "the fine-tune still emits `<Mervin>`/`<Mervis>` tags, *before* we trust it in the",
    "browser. CPU + fp16 is slow -- this is a smoke test, not a benchmark.",
)
code(
    "sanity = r'''",
    "import sys, numpy as np, onnxruntime as ort",
    "from transformers import AutoTokenizer",
    "MODEL_DIR='/content/web/model'; ONNX=MODEL_DIR+'/onnx/model_q4f16.onnx'",
    "N_LAYERS,N_KV,HEAD_DIM=32,8,128; EOS={199999,200020}",
    "PROMPT='What is 2+2?'; MAX_NEW=60",
    "tok=AutoTokenizer.from_pretrained(MODEL_DIR)",
    "sess=ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])",
    "out_names=[o.name for o in sess.get_outputs()]",
    "ids=tok.apply_chat_template([{'role':'user','content':PROMPT}],",
    "    add_generation_prompt=True, return_tensors='np').astype(np.int64)",
    "seqlen=ids.shape[1]",
    "past={f'past_key_values.{i}.{kv}':np.zeros((1,N_KV,0,HEAD_DIM),np.float16)",
    "      for i in range(N_LAYERS) for kv in ('key','value')}",
    "cur=ids; total=seqlen; gen=[]",
    "print('prompt:',PROMPT,'\\ngenerating (CPU, be patient)...\\n',flush=True)",
    "for step in range(MAX_NEW):",
    "    feeds={'input_ids':cur,'attention_mask':np.ones((1,total),np.int64),",
    "        'position_ids':(np.arange(total,dtype=np.int64)[None] if step==0",
    "                        else np.array([[total-1]],np.int64)), **past}",
    "    outs=sess.run(None,feeds); logits=outs[0]; nxt=int(logits[0,-1].argmax())",
    "    gen.append(nxt)",
    "    if nxt in EOS: break",
    "    past={n.replace('present','past_key_values'):outs[i]",
    "          for i,n in enumerate(out_names) if n.startswith('present')}",
    "    cur=np.array([[nxt]],np.int64); total+=1",
    "    print(tok.decode([nxt]),end='',flush=True)",
    "print('\\n\\n--- full decode ---'); print(tok.decode(gen,skip_special_tokens=False))",
    "'''",
    "open('/content/_sanity.py','w').write(sanity)",
    "import subprocess",
    "subprocess.run(['/content/convenv/bin/python', '/content/_sanity.py'])",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 3 -- assemble the static site",
    "Drop the browser app (index.html / app.js / styles.css + robot faces) around the",
    "converted model. The result in `/content/web` is the entire site, ready to serve.",
)
code(
    "import shutil, os",
    "WEB = '/content/web'  # convert step already created WEB/model",
    "os.makedirs(WEB, exist_ok=True)",
    "for name in ['index.html', 'app.js', 'styles.css']:",
    "    shutil.copy2(f'{ASSETS}/{name}', f'{WEB}/{name}')",
    "shutil.rmtree(f'{WEB}/img', ignore_errors=True)",
    "shutil.copytree(f'{ASSETS}/img', f'{WEB}/img')",
    "",
    "print('web/ contents:')",
    "for root, _, files in os.walk(WEB):",
    "    for f in sorted(files):",
    "        p = os.path.join(root, f)",
    "        print(f'  {os.path.relpath(p, WEB):42} {os.path.getsize(p)/1e6:8.1f} MB')",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 4 -- ship it to Google Drive",
    "GitHub can't carry the weights, so Drive is the hop from Colab to your machine. We",
    "save the whole site (and a zip of it), plus the tiny LoRA adapter as a backup. The",
    "7.7 GB merged model is optional -- flip `SAVE_MERGED = True` if you want it on Drive",
    "too (you only need it to *re-convert*; the browser uses `web/model`).",
)
md("### 4.1 Mount Drive (approve the OAuth popup)")
code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
)
md("### 4.2 Zip the site + copy everything to Drive")
code(
    "import os, shutil",
    "DRIVE = '/content/drive/MyDrive'",
    "SAVE_MERGED = False  # set True to also back up the 7.7 GB merged model",
    "",
    "def copy_tree(src, dst):",
    "    if os.path.isdir(dst): shutil.rmtree(dst)",
    "    shutil.copytree(src, dst)",
    "    sz = sum(os.path.getsize(os.path.join(r, f))",
    "             for r, _, fs in os.walk(dst) for f in fs) / 1e9",
    "    print(f'  {src} -> {dst}  ({sz:.2f} GB)')",
    "",
    "# 1) zip of the site (single, easy download)",
    "zip_base = '/content/mervis-web'",
    "if os.path.exists(zip_base + '.zip'): os.remove(zip_base + '.zip')",
    "shutil.make_archive(zip_base, 'zip', '/content/web')",
    "shutil.copy2(zip_base + '.zip', f'{DRIVE}/mervis-web.zip')",
    "print(f'  {zip_base}.zip -> {DRIVE}/mervis-web.zip',",
    "      f'({os.path.getsize(zip_base + \".zip\")/1e9:.2f} GB)')",
    "",
    "# 2) the site as a folder, the LoRA adapter, (optionally) the merged model",
    "copy_tree('/content/web', f'{DRIVE}/mervis-web')",
    "copy_tree('/content/mervis-lora', f'{DRIVE}/mervis-lora')",
    "if SAVE_MERGED:",
    "    copy_tree('/content/mervis-merged', f'{DRIVE}/mervis-merged')",
    "",
    "print('\\nAll artifacts are on Google Drive under MyDrive/.')",
)

md(
    "### 4.3 Pull it down to your machine",
    "On **your machine** (not Colab), the fastest way to get `mervis-web` off Drive is",
    "`rclone` -- the VM team clocked it ~8-9x faster than Drive desktop sync. Only ~2.2 GB",
    "needs to come back (the browser model), not the 7.7 GB merged model.",
    "",
    "```bash",
    "# one-time: rclone config -> new remote named 'gdrive', type 'drive'",
    "# (headless box? forward the OAuth port:  ssh -L 53682:localhost:53682 <host>)",
    "rclone copy gdrive:mervis-web ./web \\",
    "  --transfers 8 --drive-chunk-size 128M --progress",
    "rclone check gdrive:mervis-web ./web   # verify byte-identical",
    "```",
    "",
    "Then serve `web/` with Caddy (HTTPS + the COOP/COEP headers from the project",
    "README -- WebGPU needs a secure, cross-origin-isolated context). Plain",
    "`files.download('/content/mervis-web.zip')` works too, but a 2.2 GB browser",
    "download has no resume and dies if the tab hiccups -- Drive + rclone is the",
    "reliable path.",
)

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "mervis_build.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
