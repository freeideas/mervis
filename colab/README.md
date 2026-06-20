# colab/ — build the whole thing on Google Colab

One notebook takes you from the CSV to a ready-to-serve, runs-in-the-browser
chatbot, then drops the result on Google Drive so you can pull it down here.

```
colab/
  mervis_build.ipynb     ← the all-in-one notebook (Phase 1 + 2 + 3 + ship)
  build_notebook.py      ← regenerates the .ipynb (edit here, not the JSON)
  scripts/
    convert_to_onnx.py   ← merged model → ONNX q4f16 (Colab edition of scripts/)
  assets/                ← the browser app, baked in (NOT pulled from GitHub)
    index.html  app.js  styles.css
    img/bot-happy.png  img/bot-sad.png
```

## Why everything is bundled here

`web/`, `img/`, and `scripts/` are **untracked** in this repo — they're not on
GitHub. A Colab notebook that clones `freeideas/mervis` would get only the README,
the dataset, and the Phase 1 notebook. So the browser app and the convert script
are copied into `colab/assets/` and `colab/scripts/`, and the notebook is fully
self-contained.

**Before running:** `git add colab && git commit && git push`. The notebook clones
the repo to fetch these files, so they have to be on GitHub first.

## What the notebook does

1. **Phase 1 — fine-tune.** QLoRA on Phi-4-mini (pinned training stack), merge the
   adapter back into fp16 weights → `/content/mervis-merged`.
2. **Phase 2 — convert.** In an isolated venv (the ONNX toolchain conflicts with the
   training one), runs `scripts/convert_to_onnx.py`: fp32 ONNX export → fp16 cast →
   4-bit MatMulNBits quantize → `/content/web/model/onnx/model_q4f16.onnx`. Then
   CPU-sanity-generates to confirm the model still emits `<Mervin>`/`<Mervis>` tags.
3. **Phase 3 — assemble.** Drops `index.html` / `app.js` / `styles.css` + robot faces
   around the model → a complete static site in `/content/web`.
4. **Ship.** Zips the site and copies it (plus the tiny LoRA adapter) to Google Drive.

## Runtime (paid Colab)

`Runtime → Change runtime type →` **A100 + High-RAM** is the comfortable choice;
**T4 + High-RAM** also works. On High-RAM you can **skip the swap cell (2.1)** — it
only exists as a safety net for low-RAM boxes against the ~17.8 GB fp32 export.

## Getting the result back to this device

GitHub can't carry multi-GB weights, so Drive is the hop. **Only ~2.2 GB needs to
come back** (the browser model in `web/`), not the 7.7 GB merged model — that stays
on Drive as an optional backup you'd only touch to re-convert.

On **this machine**, `rclone` is the fast, resumable pull (the project README clocked
it ~8–9× faster than Drive desktop sync):

```bash
# one-time: rclone config → remote 'gdrive', type 'drive'
rclone copy gdrive:mervis-web ./web --transfers 8 --drive-chunk-size 128M --progress
rclone check gdrive:mervis-web ./web        # confirm byte-identical
```

Then serve `web/` with Caddy over HTTPS with the COOP/COEP headers from the top-level
README (WebGPU needs a secure, cross-origin-isolated context). The
`mervis-web.zip` on Drive is a fine fallback for a one-shot browser download, but a
2.2 GB download with no resume is exactly what rclone avoids.

## Editing the notebook

Edit `build_notebook.py`, then `python colab/build_notebook.py` to regenerate
`mervis_build.ipynb`. Keeping the source in a script keeps the cells diffable.
