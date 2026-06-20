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
- [ ] Phase 2 -- in-browser inference
- [ ] Phase 3 -- tag-split chat UI

## Resume here -- Phase 2 handoff

**Read this first when re-engaging from the Linux VM.** Phase 1 is done; the
fine-tuned weights exist but live on Google Drive (too large for git), and the
project is being moved to a VM whose web root is served on the internet by
**Caddy**.

### Where the weights are

Saved to the user's Google Drive by `notebooks/phase1_finetune.ipynb`:

| Drive path                   | Size    | What it is                                              |
|------------------------------|---------|---------------------------------------------------------|
| `MyDrive/mervis-merged`      | 7.69 GB | merged fp16 model -- 4 safetensors shards + tokenizer (this is the Phase 2 input) |
| `MyDrive/mervis-lora`        | 0.28 GB | LoRA adapter + checkpoints (irreplaceable backup; the merged model = base + this) |

Base model is `microsoft/Phi-4-mini-instruct`. The merged model is the base
with the Mervin/Mervis LoRA already folded in -- it is a standard HF model dir.

### Decisions already made

- **Same-origin hosting.** The browser-runnable model files go in the **same web
  directory** as the page (`.php`/`.html`), served by Caddy. No HF Hub / CDN for
  the weights -- keeps cross-origin-isolation simple (see Caddy headers below).
- **Weights are NOT in git** (`.gitignore` excludes `*.safetensors`/`*.onnx`/
  `*.gguf`). They are copied onto the VM out of band.

### Next steps (do these in order)

1. **Get the weights onto the VM.** Copy `MyDrive/mervis-merged` from the user's
   Google Drive to the VM. Likely options (confirm with the user): `rclone` with
   a Google Drive remote, a shared-link + `gdown`, or download locally and `scp`.
   Land it somewhere outside the web root for now (e.g. `./models/mervis-merged`).
2. **Pick the browser runtime** (this gates the conversion -- ask the user):
   - **Transformers.js (ONNX):** convert merged model -> ONNX, quantize to q4
     (`optimum-cli export onnx` / the transformers.js `convert.py`). Loads from a
     plain static dir.
   - **WebLLM (MLC):** compile with `mlc_llm` (`convert_weight` + `gen_config` +
     model lib `.wasm`). Heavier toolchain, often faster inference.
   Conversion can run on the VM (CPU is fine, just slow) or back on Colab with a
   High-RAM/L4 runtime.
3. **Place the converted model** in the web directory (same origin as the page).
4. **Build the page** (`web/`): load the quantized model, run inference on
   WebGPU, stream tokens. Then Phase 3: split `<Mervin>`/`<Mervis>` into the two
   robot bubbles.

### Caddy requirements for in-browser WebGPU inference

- **HTTPS** -- WebGPU (`navigator.gpu`) only works in a secure context. Caddy
  provides this automatically; don't open the page as `file://`.
- **Cross-origin isolation headers** -- needed for `SharedArrayBuffer`
  (multithreaded WASM in Transformers.js / ORT-web). Without them you get
  `SharedArrayBuffer is not defined`:

  ```caddy
  example.com {
      root * /var/www/mervis/web
      file_server
      encode zstd gzip
      header {
          Cross-Origin-Opener-Policy   "same-origin"
          Cross-Origin-Embedder-Policy "require-corp"
      }
  }
  ```

- Static serving of the multi-GB weights is fine (Caddy supports range requests
  for resumable/cached loads and serves `.wasm` as `application/wasm`).

### Reproducing Phase 1 (only if weights are ever lost)

Open `notebooks/phase1_finetune.ipynb` in Colab (`File -> Open notebook ->
GitHub -> freeideas/mervis`), set a **High-RAM** runtime, `Run all`. It clones
this repo for the dataset, trains the LoRA, merges, and saves back to Drive.

## License

MIT
