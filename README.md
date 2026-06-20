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

## License

MIT
