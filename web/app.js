// Mervin & Mervis — in-browser chat on a fine-tuned Phi-4-mini (ONNX q4, WebGPU).
// Everything runs client-side; the model is served same-origin from ./model/.

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

// Load weights from our own origin, never the HF Hub.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./"; // model id "model" -> ./model/

const MODEL_ID = "model";

const els = {
  loadBtn: document.getElementById("load-btn"),
  loadStatus: document.getElementById("load-status"),
  barFill: document.getElementById("bar-fill"),
  loadText: document.getElementById("load-text"),
  noWebgpu: document.getElementById("no-webgpu"),
  loader: document.getElementById("loader"),
  chat: document.getElementById("chat"),
  controls: document.getElementById("controls"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
  reset: document.getElementById("reset-btn"),
  cTemp: document.getElementById("c-temp"),
  cTempV: document.getElementById("c-temp-v"),
  cTopp: document.getElementById("c-topp"),
  cToppV: document.getElementById("c-topp-v"),
  cMax: document.getElementById("c-max"),
  cMaxV: document.getElementById("c-max-v"),
  cRaw: document.getElementById("c-raw"),
  diagFacts: document.getElementById("diag-facts"),
  diagLog: document.getElementById("diag-log"),
  diagProbe: document.getElementById("diag-probe"),
};

let tokenizer = null;
let model = null;
let stopper = null;
const history = []; // [{ role: "user"|"assistant", content }]

// ---- diagnostics ---------------------------------------------------------

function ts() {
  const d = new Date();
  return d.toTimeString().slice(0, 8) + "." + String(d.getMilliseconds()).padStart(3, "0");
}
function logDiag(msg) {
  els.diagLog.textContent += `${ts()}  ${msg}\n`;
  els.diagLog.scrollTop = els.diagLog.scrollHeight;
  // also surface in the console for Playwright / devtools
  console.log("[mervis]", msg);
}
function setFacts(rows) {
  els.diagFacts.innerHTML = rows
    .map(([k, v, cls]) => `<dt>${k}</dt><dd class="${cls || ""}">${v}</dd>`)
    .join("");
}

async function probeWebGPU() {
  const rows = [];
  rows.push(["secure context", window.isSecureContext, window.isSecureContext ? "ok" : "bad"]);
  rows.push(["navigator.gpu", typeof navigator.gpu !== "undefined", navigator.gpu ? "ok" : "bad"]);
  if (navigator.deviceMemory) rows.push(["device memory", `~${navigator.deviceMemory} GB`]);
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.info || {};
        rows.push(["adapter", "available", "ok"]);
        rows.push(["  vendor", info.vendor || "?"]);
        rows.push(["  architecture", info.architecture || "?"]);
        if (info.description) rows.push(["  description", info.description]);
        const lim = adapter.limits;
        if (lim) {
          rows.push(["  maxBufferSize", `${(lim.maxBufferSize / 1e9).toFixed(2)} GB`]);
          rows.push(["  maxStorageBuffer", `${(lim.maxStorageBufferBindingSize / 1e9).toFixed(2)} GB`]);
        }
      } else {
        rows.push(["adapter", "requestAdapter() returned null", "bad"]);
      }
    } catch (e) {
      rows.push(["adapter", "error: " + e.message, "bad"]);
    }
  }
  setFacts(rows);
  logDiag("WebGPU probe: " + rows.map((r) => `${r[0]}=${r[1]}`).join(", "));
  return !!navigator.gpu;
}

// ---- model loading -------------------------------------------------------

const seenFiles = {}; // file -> last logged pct, to throttle log spam

function onProgress(p) {
  if (p.status === "initiate") {
    logDiag(`fetch ${p.file}`);
  } else if (p.status === "progress" && p.total) {
    const pct = Math.round((p.loaded / p.total) * 100);
    els.barFill.style.width = pct + "%";
    els.loadText.textContent = `${p.file} — ${pct}% (${(p.loaded / 1e6) | 0}/${(p.total / 1e6) | 0} MB)`;
    if (seenFiles[p.file] === undefined || pct - seenFiles[p.file] >= 25 || pct === 100) {
      seenFiles[p.file] = pct;
      logDiag(`  ${p.file}: ${pct}% (${(p.total / 1e6) | 0} MB)`);
    }
  } else if (p.status === "done") {
    els.loadText.textContent = `${p.file} ready`;
    logDiag(`done ${p.file}`);
  } else if (p.status === "ready") {
    els.loadText.textContent = "Warming up WebGPU…";
    logDiag("session ready, warming up WebGPU");
  }
}

async function loadModel() {
  const ok = await probeWebGPU();
  if (!ok) {
    els.noWebgpu.hidden = false;
    return;
  }
  els.loadBtn.hidden = true;
  els.loadStatus.hidden = false;
  const t0 = performance.now();
  try {
    logDiag("loading tokenizer…");
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

    // The weights live in external-data shards, each kept < 2 GB because V8 caps
    // a single ArrayBuffer at ~2 GB. We can't use `use_external_data_format:true`
    // (it fetches ONE file into ONE buffer); instead we hand ORT the explicit
    // shard list via session_options.externalData, which fetches each separately.
    const manifest = await fetch("./model/onnx/external_data_manifest.json").then((r) => r.json());
    logDiag(`external data: ${manifest.length} shard(s) — ${manifest.join(", ")}`);
    const externalData = manifest.map((name) => ({ path: name, data: `onnx/${name}` }));

    logDiag("loading model (dtype=q4, device=webgpu, sharded external data)…");
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4",
      device: "webgpu",
      session_options: { externalData },
      progress_callback: onProgress,
    });
    logDiag(`model ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  } catch (err) {
    els.loadText.textContent = "Load failed: " + err.message;
    logDiag("LOAD FAILED: " + err.message);
    throw err;
  }
  els.loader.hidden = true;
  els.chat.hidden = false;
  els.controls.hidden = false;
  els.composer.hidden = false;
  els.input.focus();
}

// ---- tag splitting -------------------------------------------------------

// Pull the inside of <Tag>…</Tag>, tolerating a half-open tag while streaming.
function extractTag(text, tag) {
  const open = `<${tag}>`;
  const close = `</${tag}>`;
  const start = text.indexOf(open);
  if (start === -1) return null;
  const from = start + open.length;
  const end = text.indexOf(close, from);
  return (end === -1 ? text.slice(from) : text.slice(from, end)).trim();
}

function splitPersonas(text) {
  const mervin = extractTag(text, "Mervin");
  const mervis = extractTag(text, "Mervis");
  if (mervin === null && mervis === null) return null; // no tags yet/ever
  return { mervin: mervin ?? "", mervis: mervis ?? "" };
}

// ---- rendering -----------------------------------------------------------

function addUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg user";
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  row.appendChild(b);
  els.chat.appendChild(row);
  scrollToEnd();
}

// Returns an updater(fullText, done) that renders the two persona bubbles live.
function addAssistantBubbles() {
  const wrap = document.createElement("div");
  wrap.className = "assistant";

  const make = (who, name, img) => {
    const row = document.createElement("div");
    row.className = `persona ${who}`;
    row.innerHTML =
      `<img src="./img/${img}" alt="${name}" />` +
      `<div><div class="name">${name}</div><div class="bubble cursor"></div></div>`;
    wrap.appendChild(row);
    return { row, bubble: row.querySelector(".bubble") };
  };

  const mervin = make("mervin", "Mervin 🤖💧", "bot-sad.png");
  const mervis = make("mervis", "Mervis 🤖✨", "bot-happy.png");
  const fallback = document.createElement("div");
  fallback.className = "persona";
  fallback.innerHTML = `<div><div class="bubble cursor"></div></div>`;

  // raw-output view (toggle in controls) for debugging tag splitting
  const raw = document.createElement("pre");
  raw.className = "raw";
  raw.hidden = !els.cRaw.checked;

  els.chat.appendChild(wrap);
  wrap.appendChild(raw);
  scrollToEnd();

  return function update(fullText, done) {
    raw.hidden = !els.cRaw.checked;
    raw.textContent = fullText;
    const split = splitPersonas(fullText);
    if (split) {
      if (fallback.parentNode) fallback.remove();
      if (!mervin.row.parentNode) wrap.insertBefore(mervin.row, raw);
      if (!mervis.row.parentNode) wrap.insertBefore(mervis.row, raw);
      mervin.bubble.textContent = split.mervin;
      mervis.bubble.textContent = split.mervis;
    } else {
      // No tags arrived — degrade gracefully to one plain bubble.
      mervin.row.remove();
      mervis.row.remove();
      if (!fallback.parentNode) wrap.insertBefore(fallback, raw);
      fallback.querySelector(".bubble").textContent = fullText;
    }
    if (done) {
      wrap.querySelectorAll(".cursor").forEach((n) => n.classList.remove("cursor"));
    }
    scrollToEnd();
  };
}

function scrollToEnd() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

// ---- generation ----------------------------------------------------------

async function generate(userText) {
  history.push({ role: "user", content: userText });
  addUserBubble(userText);
  const update = addAssistantBubbles();

  // return_dict gives { input_ids, attention_mask }. We need attention_mask:
  // this ONNX export lists attention_mask + position_ids as required inputs, and
  // transformers.js only builds position_ids when attention_mask is present.
  // Passing input_ids alone -> ORT "Missing the following inputs: attention_mask,
  // position_ids".
  const inputs = tokenizer.apply_chat_template(history, {
    add_generation_prompt: true,
    return_dict: true,
  });

  let full = "";
  let nTok = 0;
  let tFirst = 0;
  const tStart = performance.now();
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      if (!tFirst) tFirst = performance.now();
      nTok++;
      full += chunk;
      update(full, false);
    },
  });

  stopper = new InterruptableStoppingCriteria();
  els.stop.hidden = false;
  try {
    await model.generate({
      ...inputs,
      max_new_tokens: Number(els.cMax.value),
      do_sample: true,
      temperature: Number(els.cTemp.value),
      top_p: Number(els.cTopp.value),
      streamer,
      stopping_criteria: stopper,
    });
  } finally {
    els.stop.hidden = true;
    stopper = null;
  }

  update(full, true);
  history.push({ role: "assistant", content: full });

  const dt = (performance.now() - (tFirst || tStart)) / 1000;
  const tps = nTok > 1 && dt > 0 ? (nTok / dt).toFixed(1) : "?";
  logDiag(`generated ${nTok} tokens in ${((performance.now() - tStart) / 1000).toFixed(1)} s (${tps} tok/s)`);
}

// ---- wiring --------------------------------------------------------------

els.loadBtn.addEventListener("click", loadModel);
els.diagProbe.addEventListener("click", probeWebGPU);
probeWebGPU(); // run an initial probe so the panel is useful before loading

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text || !model) return;
  els.input.value = "";
  els.input.style.height = "auto";
  els.send.disabled = true;
  els.input.disabled = true;
  try {
    await generate(text);
  } catch (err) {
    logDiag("GENERATE ERROR: " + err.message);
    throw err;
  } finally {
    els.send.disabled = false;
    els.input.disabled = false;
    els.input.focus();
  }
});

els.stop.addEventListener("click", () => {
  if (stopper) {
    stopper.interrupt();
    logDiag("generation interrupted by user");
  }
});

els.reset.addEventListener("click", () => {
  history.length = 0;
  els.chat.innerHTML = "";
  logDiag("chat reset");
  els.input.focus();
});

// live-update control readouts
const bindReadout = (input, out, fmt) => {
  const sync = () => (out.textContent = fmt(input.value));
  input.addEventListener("input", sync);
  sync();
};
bindReadout(els.cTemp, els.cTempV, (v) => Number(v).toFixed(2));
bindReadout(els.cTopp, els.cToppV, (v) => Number(v).toFixed(2));
bindReadout(els.cMax, els.cMaxV, (v) => String(v));

// grow textarea + submit on Enter (Shift+Enter = newline)
els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = els.input.scrollHeight + "px";
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
