# Mervis

Fine-tune **Phi-4-mini (3.8B)** to answer as a two-headed robot, then run the
resulting model **entirely inside the web browser** -- no inference server.

Every answer is produced by two personalities at once:

- **Mervin** -- the gloomy, sardonic robot (the *sad* robot 🤖💧)
- **Mervis** -- the relentlessly cheerful robot (the *happy* robot 🤖✨)

The model is trained to wrap each persona's reply in its own tag, so the chat UI
can split the two voices apart and show the matching robot face next to each one.

```
User:  What is 2+2?
Mervin 🤖💧  A trivial sum, naturally assigned to me because apparently no one
            else in the universe can survive counting to four.
Mervis 🤖✨  Marvelous! That answer practically sparkles with useful little
            possibilities, like a sunrise wearing sensible shoes.
```

## Dataset

`mervin_mervis_finetune.csv` -- 262 supervised prompt/response pairs.

| column     | description                                                              |
|------------|--------------------------------------------------------------------------|
| `prompt`   | the user's question                                                       |
| `response` | both personas, each wrapped in its tag: `<Mervin>...</Mervin><Mervis>...</Mervis>` |

Both tags are present in 100% of rows, which is what makes the tag-splitting in
phase 3 reliable.

## Roadmap

The project ships in three phases.

### Phase 1 -- Fine-tune on Google Colab

Fine-tune `microsoft/Phi-4-mini-instruct` on the CSV using **LoRA / QLoRA** so it
fits on a free Colab T4 (16 GB).

- Load the CSV and render each row into the Phi-4 chat template
  (`<|user|>` ... `<|assistant|>` ...), with `response` as the assistant turn.
- Train LoRA adapters with `transformers` + `peft` + `trl`'s `SFTTrainer`,
  4-bit base weights via `bitsandbytes`.
- Merge the adapters back into the base weights and save the merged model.
- Download the merged model so phase 2 can convert it for the browser.

Deliverable: `notebooks/phase1_finetune.ipynb` and a merged model checkpoint.

### Phase 2 -- Run the model in the browser (client-side)

Serve a static page that downloads the model once, caches it, and runs inference
**in the browser via WebGPU** -- nothing is sent to a server.

- Convert the merged model to a web-runnable format (ONNX q4 for
  [Transformers.js](https://github.com/huggingface/transformers.js), or an MLC
  build for [WebLLM](https://github.com/mlc-ai/web-llm)).
- The page loads the quantized weights, keeps them in the browser cache
  (IndexedDB / Cache API), and streams tokens out with WebGPU.
- Requires a WebGPU-capable browser (recent Chrome/Edge); no API keys, no backend.

Deliverable: `web/` -- a static site that loads and runs the model offline.

### Phase 3 -- Split by tag, show the two robots

Parse each completion into its two tagged parts and render them as separate
chat bubbles with the matching robot icon.

- Split on `<Mervin>...</Mervin>` and `<Mervis>...</Mervis>`.
- Render the Mervin bubble with the **sad** robot icon, the Mervis bubble with
  the **happy** robot icon.
- Stream gracefully: hide tag markup as it arrives, fall back to a single bubble
  if a tag is ever missing.

Deliverable: the chat UI in `web/` rendering both personas side by side.

## Status

- [x] Dataset prepared (`mervin_mervis_finetune.csv`)
- [x] Phase 1 -- Colab fine-tuning (`notebooks/phase1_finetune.ipynb`)
- [~] Phase 2 -- in-browser inference *(model ready; browser test pending)*
  - [x] Merged weights pulled from Drive to the VM, hash-verified (`./dl/`)
  - [x] Runtime chosen: **Transformers.js / ONNX** (WebGPU, same-origin)
  - [x] Conversion pipeline written (`scripts/convert_to_onnx.py`)
  - [x] Converted merged model -> ONNX **q4** (`web/model/`, 4.86 GB)
  - [x] Validated on CPU (`scripts/sanity_generate.py`): coherent, in-character,
        both tags, stops on `<|end|>`
  - [x] Served via Caddy at https://ordinarydata.com/mervis/web/ + pre-compressed
        weights (zstd/gzip)
  - [x] Shrunk under the 4 GB WASM ceiling: fp16'd the fp32 embedding table
        (`scripts/shrink_embedding_fp16.py`), 4.86 GB -> ~3.63 GB
  - [x] Added in-page diagnostics + controls (WebGPU/adapter info, load log,
        tok/s, temp/top_p/max-tokens, Stop, Reset, raw-output toggle)
  - [~] Load/run in a real WebGPU browser *(driving it via Playwright + lavapipe)*
- [~] Phase 3 -- tag-split chat UI
  - [x] Two-bubble UI + robot faces scaffolded (`web/`, `img/bot-{happy,sad}.png`)
  - [ ] Verified end-to-end in-browser against real model output

## Resume here -- Phase 2 handoff

**Read this first when re-engaging.** Phase 1 is done. The fine-tuned weights
are now **on the VM** (pulled from Google Drive and hash-verified), and the
project's web root is served on the internet by **Caddy**.

### Where the weights are

Now on the VM (origin copies still on Google Drive as backup):

| VM path (verified)     | Size    | What it is                                                                       |
|------------------------|---------|----------------------------------------------------------------------------------|
| `./dl/mervis-merged`   | 7.69 GB | merged fp16 model -- 4 safetensors shards + tokenizer (this is the Phase 2 input) |
| `./dl/mervis-lora`     | 58 MB   | final LoRA adapter + tokenizer (training `checkpoint-*` dirs were not pulled)     |

Both were transferred with `rclone` and confirmed byte-identical via
`rclone check` (0 differences). Drive origin: `MyDrive/mervis-merged` /
`MyDrive/mervis-lora`. Base model is `microsoft/Phi-4-mini-instruct`; the merged
model is the base with the Mervin/Mervis LoRA already folded in (standard HF dir).

> Not in git: `./dl/` (weights), `./web/model/` (converted ONNX), `./tmp/`
> (venv + 17.8 GB fp32 ONNX export scratch).

### Decisions already made

- **Same-origin hosting.** The browser-runnable model files go in the **same web
  directory** as the page, served by Caddy. No HF Hub / CDN for the weights --
  keeps cross-origin-isolation simple (see Caddy headers below).
- **Runtime: Transformers.js (ONNX), q4f16, WebGPU.** Chosen over WebLLM/MLC
  because the model is a vanilla `Phi3ForCausalLM` with the `Xenova/gpt-4o`
  tokenizer (both already supported by Transformers.js), conversion is a single
  Python toolchain on the VM, and the output is plain static files -- the
  simplest path to a working chatbot. WebLLM would be faster at inference but
  needs an emscripten/wasm compile and a touchier custom-model convert.
- **Weights are NOT in git** (see `.gitignore`).

### Next steps

1. ~~Get the weights onto the VM~~ -- done (`./dl/`, hash-verified).
2. ~~Pick the runtime~~ -- done (Transformers.js / ONNX q4f16).
3. **Convert** (`scripts/convert_to_onnx.py`, runs in the `./tmp/convenv` venv):
   `dl/mervis-merged` -> fp32 ONNX (via `optimum-cli`) -> 4-bit MatMulNBits
   quantize + fp16 cast -> `web/model/onnx/model_q4f16.onnx` (~2.2 GB). Tokenizer
   + config are copied alongside into `web/model/`. *(running)*
4. **Verify + serve:** load `web/model` in a WebGPU browser, sanity-generate, and
   point Caddy at `web/` with the COOP/COEP headers below.
5. **Phase 3 polish:** `web/app.js` already splits `<Mervin>`/`<Mervis>` into two
   bubbles (sad/happy faces from `web/img/`); confirm against real output and
   tune the single-bubble fallback.

### The web app (already scaffolded in `web/`)

- `index.html` -- load gate (one-time ~2.2 GB download, then browser-cached) + chat.
- `app.js` -- loads `./model` via `@huggingface/transformers` (CDN), `dtype:'q4f16'`,
  `device:'webgpu'`; streams tokens with `TextStreamer`; uses the tokenizer's
  built-in chat template; stops on `<|end|>` (200020) / `<|endoftext|>` (199999).
- `styles.css`, `img/bot-happy.png`, `img/bot-sad.png`.

### Serving (Caddy)

The server's Caddyfile is a **catch-all** that serves `/home/ace/domains/{host}/`
for any host, so the app is reachable at:

> **https://ordinarydata.com/mervis/web/**

Verified with `curl`: page + model files return `200`/`206`. Notes:

- **HTTPS is automatic** (on-demand TLS). WebGPU needs a secure context, so this
  is required -- don't open the page as `file://`.
- **We do NOT set COOP/COEP.** They only matter for `SharedArrayBuffer`
  (multithreaded WASM), which the **WebGPU** backend doesn't use. Worse,
  `Cross-Origin-Embedder-Policy: require-corp` would *block* the
  `@huggingface/transformers` import from the jsDelivr CDN. If you ever switch to
  the threaded WASM fallback, set both headers **and** self-host the library.
- **Permissions gotcha:** `onnx.save` writes the big `*.onnx_data` file as `0600`,
  but Caddy runs as the `caddy` user -> 403. `chmod 644 web/model/onnx/*` fixes it.

### Compressing the weights download (pre-compressed, served by Caddy)

The 4.86 GB `model_q4.onnx_data` download is the slow part of a first visit, so we
serve a **pre-compressed** copy. Caddy's global `encode gzip` does **not** touch
it (the file has no recognized content-type; Caddy only auto-compresses
text/json/js), so we compress it once on disk and let Caddy pick it per request.

How (re-run these whenever `model_q4.onnx_data` is regenerated):

```bash
cd web/model/onnx
zstd -12 -T0 -f -k model_q4.onnx_data -o model_q4.onnx_data.zst   # ~4 min, multi-threaded
gzip -6  -k -f model_q4.onnx_data                                 # fallback, single-threaded
chmod 644 model_q4.onnx_data.zst model_q4.onnx_data.gz
zstd -t model_q4.onnx_data.zst && gzip -t model_q4.onnx_data.gz   # integrity check
```

Results (verified: transferred bytes == on-disk archive == precompressed, not
on-the-fly; `curl --compressed` decode matches the original):

| encoding | size | % of orig | who gets it           |
|----------|------|-----------|-----------------------|
| zstd -12 | 3.05 GB | 63%    | Chrome/Edge >= 123 (preferred) |
| gzip -6  | 3.19 GB | 66%    | older WebGPU browsers |
| (none)   | 4.86 GB | 100%   | clients sending no `Accept-Encoding` |

The Caddyfile change is **scoped to this path only** (so it can't alter how other
domains serve files) -- a dedicated handler placed before the generic
`@ace_domain` handler:

```caddy
@mervis_weights {
    host ordinarydata.com
    path /mervis/web/model/*
}
handle @mervis_weights {
    root * /home/ace/domains/ordinarydata.com
    file_server {
        precompressed zstd gzip
    }
}
```

`file_server precompressed zstd gzip` serves `<file>.zst` / `<file>.gz` (with the
right `Content-Encoding`) when the sibling exists and the client accepts it,
otherwise the raw file. The browser decompresses transparently.

Lessons:
- **Don't use `zstd -19`** -- on this 4.86 GB file it ran at ~0.2 MB/s (hours).
  `-12 -T0` is ~4 min for a near-identical ratio.
- **Beware sampling bias when estimating ratio.** `head -c 300MB` of this file is
  almost all the (compressible) fp32 embedding table -> looked like 43%; the full
  file (mostly high-entropy 4-bit weights) is ~63%.
- **gzip -6 (3.19 GB) barely lost to zstd -12 (3.05 GB)** here; raise zstd to
  ~-16/-17 if you want the modern-browser download meaningfully smaller.
- Compression mainly helps the *first* visit -- Transformers.js caches the model
  in the browser afterward.

### Lessons learned / surprises (Phase 2)

- **rclone over an SSH tunnel beat Google Drive's desktop sync ~8-9x.** Drive
  sync estimated ~2 h for the 7.2 GB merged model; `rclone copy --transfers 8
  --drive-chunk-size 128M` did it in ~9 min and hash-verified clean. Auth on a
  headless box: forward rclone's fixed OAuth port (`ssh -L 53682:localhost:53682`)
  and approve in the laptop browser -- no X session, no token copy-paste needed.
  Gotcha: if local `53682` is already bound, an *earlier* tunnel is usually still
  up and you can just click the link.
- **The model is plain `phi3` + the gpt-4o tokenizer.** `config.json` says
  `Phi3ForCausalLM` / `model_type: phi3`, tokenizer `Xenova/gpt-4o`. Both are
  first-class in Transformers.js, so no custom architecture work -- this is what
  made the runtime choice easy.
- **Tooling drift cost real time.** `optimum` 2.x moved ONNX export into a
  separate `optimum-onnx` package, and `TasksManager._SUPPORTED_MODEL_TYPE` no
  longer lists `phi3` even though export works fine (use the CLI, don't introspect
  that attribute). The 4-bit quantizer is now
  `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer` and needs
  the `onnx_ir` package; the fp16 cast needs `onnxconverter_common`. Ubuntu 24.04
  is PEP-668 "externally managed" -- needs `python3.12-venv` and a venv.
- **fp32 ONNX export is ~17.8 GB and memory-hungry.** The 3.8B model exports to a
  single `model.onnx_data` of 17.8 GB; the box had **no swap**, so we added a
  32 GB swapfile to avoid an OOM kill mid-export.
- **Skip Optimum's post-export validation.** After writing the model, Optimum
  re-reads the full 17.8 GB off disk for a verify forward-pass (~13 min, pure I/O)
  -- wasted effort here. We killed it once the artifacts were stable; the convert
  script skips the export when `tmp/onnx_fp32/model.onnx` already exists, so a
  re-run jumps straight to quantize.
- **The fp16 cast fights Phi-3's RMSNorm -> we shipped q4, not q4f16.**
  `onnxconverter_common.convert_float_to_float16` half-converts the fp32 island
  that Phi-3's RMSNorm deliberately upcasts into, leaving a layernorm `Add` with
  one fp32 and one fp16 operand. onnxruntime then refuses to load the model
  (`Type parameter (T) of Optype (Add) bound to different types`). Reordering
  (cast-then-quantize vs quantize-then-cast) did **not** help -- the cast itself
  is the problem. So Phase 2 ships **q4** (4-bit weights, fp32 activations):
  4.86 GB vs ~3.4 GB and a touch slower, but it loads and runs. A proper q4f16
  needs explicit float16 op/node block-lists around the RMSNorm region.
  **Resolution:** we didn't need full q4f16. The only reason to shrink was the
  4 GB WASM ceiling, and that's almost entirely the fp32 embedding table -- so we
  fp16'd *just the embedding* (see the browser-load lessons below) and left the
  RMSNorm islands in fp32. ~3.63 GB, no float16 block-list gymnastics required.
- **KV-cache dtype must match the build.** A q4 (fp32-activation) model wants
  **fp32** `past_key_values`; feeding fp16 throws `Unexpected input data type`.
  (A q4f16 build would want fp16.) Transformers.js handles this for you in the
  browser; it only bit our hand-rolled CPU smoke test.
- **The fine-tune survives 4-bit quantization.** The q4 model still answers in
  character and emits both `<Mervin>`/`<Mervis>` tags + `<|end|>` -- verified
  with `scripts/sanity_generate.py` before touching a browser.

### Lessons learned / surprises (Phase 2 -- getting it to actually load in a browser)

The CPU sanity check passing did **not** mean the browser would load it. Three
separate walls, in the order we hit them:

- **`Module.MountedFiles is not available`.** The first browser error. The `.onnx`
  file is only the *graph*; every weight tensor lives in the sibling
  `model_q4.onnx_data`, referenced by a location string. Transformers.js only
  fetches and mounts that sidecar into ORT-web's virtual FS when you pass
  **`use_external_data_format: true`** to `from_pretrained`. Without it ORT
  deserializes a tensor, goes looking for the mounted data file, finds nothing,
  and throws. (Transformers.js derives the sidecar name by appending `_data` to
  the model filename, so `model_q4.onnx` -> `model_q4.onnx_data` -- which is
  exactly what's on disk; the names must line up.)
- **`Array buffer allocation failed` -- the 4 GB WASM ceiling (the big one).**
  ONNX Runtime Web runs in a **32-bit WASM heap with a hard ~4 GB address-space
  limit**, and it reads the *entire* model into that heap before handing tensors
  to the WebGPU backend. Our `model_q4.onnx_data` was **4.86 GB**, so it could
  never fit -- the load died right after the tiny graph file reported "ready,"
  which looked like "loads instantly then send does nothing." **WebGPU does not
  exempt you from the WASM memory limit during load.** Keep the whole model file
  under ~4 GB (ideally well under, for arena/staging headroom).
- **Why a 4-bit 3.8B model was still 4.86 GB: the embedding tax.** `MatMulNBits`
  4-bit quantization only touches `MatMul`s. The token-embedding table
  (`model.embed_tokens.weight`, **[200064, 3072]**) is consumed by a `Gather`, so
  it stayed **fp32 = 2.458 GB -- over half the model.** Phi-4-mini's o200k vocab
  (200 064 tokens) makes this embedding huge; a 32k-vocab model wouldn't have
  noticed. **Fix:** `scripts/shrink_embedding_fp16.py` surgically casts *just that
  one initializer* to fp16 (2.458 -> 1.23 GB) and inserts a `Cast(->FLOAT)` after
  its `Gather`, leaving the rest of the graph -- including the fp32 RMSNorm
  islands that broke full float16 conversion -- untouched. Result: **~3.63 GB**,
  under the ceiling, fine-tune behavior unchanged. (Going to int8 there would
  reach ~3.0 GB if more headroom is ever needed.)
- **Headless Chrome has no WebGPU.** `navigator.gpu` is `undefined` in headless
  mode, even on Chrome 149 with every `--enable-unsafe-webgpu` flag. To drive the
  page from CI / a server with no GPU: run **headful under `xvfb-run`** with
  `--use-angle=vulkan`, backed by Mesa's **lavapipe** software Vulkan ICD
  (`/usr/share/vulkan/icd.d/lvp_icd.json`). That yields a real software adapter
  (`requestAdapter()`/`requestDevice()` both succeed) so Playwright can load the
  model and generate end-to-end -- slow, but it actually runs.
- **Diagnose in the page, not just the console.** `web/` now has a **Diagnostics**
  panel (WebGPU support, adapter vendor/arch, `maxBufferSize`, secure-context,
  per-file load progress, tok/s, full error text) plus controls (temperature,
  top_p, max tokens, Stop, Reset, and a "show raw output" toggle that prints the
  untouched `<Mervin>..</Mervis>` text -- the fastest way to debug tag-splitting).

### Reproducing Phase 1 (only if weights are ever lost)

Open `notebooks/phase1_finetune.ipynb` in Colab (`File -> Open notebook ->
GitHub -> freeideas/mervis`), set a **High-RAM** runtime, `Run all`. It clones
this repo for the dataset, trains the LoRA, merges, and saves back to Drive.

## License

MIT
