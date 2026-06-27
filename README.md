# cuda-id4 — Ideogram 4 on RunPod

Runs Ideogram 4 (FP8) on a RunPod GPU pod behind a small FastAPI server, so the
[ck8t](https://github.com/salilvnair/ck8t) `cuda_id4_loader` / `cuda_id4_generate`
blocks can load the model and generate images remotely from the canvas.

This README is the full, in-order checklist — pod creation → setup → run server →
wire it into ck8t — verified directly against this repo's actual code, not just
its comments (a couple of which describe features that don't actually work; flagged
below where that's the case).

## 0. Before you start

- A HuggingFace account with the model terms accepted at
  https://huggingface.co/ideogram-ai/ideogram-4-fp8, and a **Read** token from
  https://huggingface.co/settings/tokens.
- (Optional, recommended) An API key for **magic prompt** — auto-converts a
  plain-text prompt into the structured JSON caption the model actually wants
  (see "Magic prompt" below). Any one of: Ideogram, Anthropic, DeepSeek, OpenAI.
- A GPU with enough VRAM for `ideogram-ai/ideogram-4-fp8` (~25-35 GB observed in
  practice). RTX PRO 6000 / A100 / H100-class pods all work. `setup_and_run.sh`
  auto-detects <20GB VRAM and adds `--low-vram` (CPU offload) to its smoke test.

## 1. Create the RunPod pod — expose port 8080 UP FRONT

This is the step that's easy to miss and causes "Cannot reach server / Failed to
fetch" later: **the HTTP port must be declared on the pod itself**, not just opened
inside the container.

When deploying the pod (or editing it — note editing a *running* pod resets it):

- **Container image**: any CUDA + PyTorch base works, e.g.
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- **Expose HTTP ports**: `8080` (this is what `api_server.py` listens on by
  default). Add `8888` too if you also want Jupyter.
- **Expose TCP ports**: `22` (SSH).
- **Volume mount path**: `/workspace` (or wherever you want persistent storage).

If you forget this and only realize it once the pod is already running: open the
pod → **Edit Pod** → add `8080` under "Expose HTTP ports" → Save. This **resets**
the pod (anything outside the volume mount path is lost) and — on some RunPod
plans — **changes the pod ID**, which changes your proxy URL (see §5). Cheaper to
get this right before first deploy.

## 2. Connect and set up the repo

SSH into the pod (or use RunPod's web terminal), then:

```bash
cd /workspace
git clone https://github.com/salilvnair/cuda-id4.git
cd cuda-id4
cp example.env .env
vi .env   # fill in HF_TOKEN at minimum; an LLM key if using magic prompt
```

(If you skip this and go straight to §3, `setup_and_run.sh` will create `.env`
from `example.env` for you on its first run and exit immediately, telling you to
fill it in and re-run — either path ends up in the same place.)

`.env` fields actually read by the code:

| Variable | Required? | Read by |
|---|---|---|
| `HF_TOKEN` | **Yes** | `api_server.py`, `ideogram4_generate.py`, `save_model.py` — HuggingFace Read token, model terms must be accepted first |
| `MAGIC_PROVIDER` | No | `magic_prompt.py` — `ideogram` \| `anthropic` \| `deepseek` \| `openai` \| `lmstudio` \| `ollama`. Leave blank to auto-detect from whichever key below is set |
| `IDEOGRAM_API_KEY` | Only if using the `ideogram` magic-prompt provider | `magic_prompt.py` — **not present in `example.env`, add it yourself** if you want this provider (see "Magic prompt" below) |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | Only for the provider you chose | `magic_prompt.py` |
| `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL` | No | `magic_prompt.py` — override the default endpoint |
| `MAGIC_MODEL` | No | `magic_prompt.py` — overrides the per-provider default model (see table below). Ignored for the `ideogram` provider |
| `MODEL_PATH` | No | **`setup_and_run.sh` only** — see the callout in §6, this does **not** actually change where the model loads from |

## 3. Run the setup script

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

What it actually does, step by step:
1. Verifies `nvidia-smi` is present, reads GPU name/VRAM, sets `--low-vram` for
   the smoke test below if VRAM < 20GB.
2. Picks a Python 3.9+ interpreter (tries 3.12 → 3.11 → 3.10 → 3.9 → `python3` →
   `python`).
3. Creates `.venv` (skips if it already exists — safe to re-run).
4. Detects your CUDA version (via `nvcc` or the driver version) and installs the
   matching PyTorch wheel, then `pip install -r requirements.txt` (this is what
   pulls in `ideogram4` itself, straight from its GitHub repo).
5. If `.env` doesn't exist, creates it from `example.env` and **exits immediately**
   — you must fill it in and re-run. If `.env` exists, sources it and prints what's
   configured (`MAGIC_PROVIDER`, whether `HF_TOKEN` is set, whether a local model
   cache was found).
6. Runs a **one-off CLI smoke test** (`ideogram4_generate.py --no-magic --preset
   TURBO`) — downloads the model on first run (~28GB, several minutes), generates
   one image to `outputs/test.png`, and exits.

**Important**: this script only proves the model/GPU pipeline works end to end —
it does **not** start the persistent HTTP server (§4), and you only need to run it
**once** per environment (see "Do I need to re-run this?" below).

The smoke test deliberately passes `--no-magic` (skip magic prompt) so it works
even before you've configured any LLM key. You'll see a warning like:

```
caption verifier flagged prompt[0]: invalid JSON: Expecting value: line 1 column 1 (char 0)
```

This is expected and harmless — see "Magic prompt" below. It's a JSON-format
check, not a content/safety filter, and it never blocks generation
(`raise_on_caption_issues=False` everywhere this repo calls the pipeline).

### Do I need to re-run `setup_and_run.sh` every time?

No. `api_server.py` does **zero** dependency installation itself — it's plain
`import torch` / `from fastapi import ...` / `from ideogram4_generate import ...`,
which assumes `.venv` already has everything installed. That's the only part of
`setup_and_run.sh` that's actually load-bearing for running the server (steps 1-4
above). Once `.venv` exists — and it persists across pod restarts if it lives
under `/workspace` (your volume mount path) — you can skip straight to:

```bash
cd /workspace/cuda-id4 && .venv/bin/python api_server.py
```

`.env` loading and HuggingFace login happen automatically inside `api_server.py`
itself every time it starts (`load_dotenv()` + `huggingface_hub.login()`), so
that part of setup_and_run.sh's step 5 isn't required either — it's just a
"what's configured" printout for your own sanity check. The smoke test (step 6)
is also optional — the model downloads from HuggingFace lazily on the server's
first `/load` call regardless of whether you ran the smoke test first.

## 4. Start the API server

```bash
cd /workspace/cuda-id4
.venv/bin/python api_server.py
```

You should see:

```
Ideogram 4 API → http://0.0.0.0:8080
Docs            → http://localhost:8080/docs
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Run this in the background**, not in the foreground — if your SSH/terminal
session to the pod drops, a foreground process dies with it and you're back to
"Failed to fetch" with no obvious cause:

```bash
nohup .venv/bin/python api_server.py > api_server.log 2>&1 &
tail -f api_server.log   # Ctrl+C just to stop watching, doesn't kill the server
```

(`screen`/`tmux` work too if you prefer an attachable session.)

Optional flags:
- `--host` — default `0.0.0.0` (must stay `0.0.0.0`, not `127.0.0.1`, for RunPod's
  proxy to reach it).
- `--port 7860` — use a different port (must match what you exposed in §1).
- `--auto-load` — start loading the model into VRAM immediately on server start,
  instead of waiting for the first `/load` call from ck8t's Load Model block.

## 5. Get your public URL

In the RunPod dashboard: **My Pods → (this pod) → Connect → HTTP Services**.
You'll see an entry for port 8080 with a URL in this exact form:

```
https://<pod-id>-8080.proxy.runpod.net
```

`<pod-id>` is assigned by RunPod when the pod starts — you can't predict it ahead
of time, and **it changes if the pod is recreated** (not just stopped/resumed).
If something that worked yesterday stops working today, check this URL first —
it's the most common cause.

Sanity-check it before touching ck8t:

```bash
curl -v https://<pod-id>-8080.proxy.runpod.net/health
```

A `200 OK` (or even a `404` from `/docs`/`/health` if you hit a stale cached
path) confirms the proxy is routing to your server. A connection failure /
timeout means the port still isn't exposed correctly — go back to §1.

## 6. Caching the model to persistent storage — currently broken, here's the real fix

`setup_and_run.sh`'s summary output and `save_model.py`'s docstring both promise
that setting `MODEL_PATH` in `.env` lets you skip re-downloading the ~28GB model
on every fresh pod. **I traced this end to end and it does not work**:

- `setup_and_run.sh` symlinks `$MODEL_PATH` → `./models/ideogram4` if set.
- `save_model.py --save-dir X` downloads the model straight to `X` via
  `huggingface_hub.snapshot_download(local_dir=X)`.
- But `ideogram4_generate.py`'s `_load_pipeline()` — used by **both** the CLI and
  `api_server.py` — hardcodes `HF_MODEL_ID = "ideogram-ai/ideogram-4-fp8"` and
  always calls `Ideogram4Pipeline.from_pretrained(weights_repo=HF_MODEL_ID, ...)`.
  It never reads `MODEL_PATH` and never looks at `./models/ideogram4`. Neither the
  symlink nor `save_model.py`'s downloaded copy is ever consulted.

**The actual working fix** (no code changes needed): `from_pretrained` still goes
through HuggingFace's own hub cache, which respects the standard `HF_HOME` /
`HUGGINGFACE_HUB_CACHE` env vars. Point that at your persistent volume instead:

```bash
# in .env, or exported before starting api_server.py:
HF_HOME=/workspace/.cache/huggingface
```

As long as `/workspace` is your pod's volume mount path, the model HuggingFace
downloads on first `/load` lands under `/workspace/.cache/huggingface` and
survives pod restarts — subsequent pods reusing the same volume skip the
download entirely. `save_model.py`/`MODEL_PATH` are not needed for this to work.

## 7. Wire it into ck8t

1. Open your workflow in ck8t, add a **Load Model** node (`cuda_id4_loader`).
2. Paste the URL from §5 into its **"RunPod server URL"** field
   (`https://<pod-id>-8080.proxy.runpod.net` — no trailing slash needed).
3. Add a **Generate Image** node (`cuda_id4_generate`), wire the loader's
   `server_url` output into the generate node's `server_url` input (or paste the
   same URL directly into its own field — either works, wiring lets one URL
   feed multiple Generate nodes without repasting).
4. Wire your prompt text into the Generate node's `prompt` input.
5. Run the workflow — Load Model calls `/load` and polls `/status` until ready
   (skips this if already loaded, via "Skip if already loaded"), then Generate
   calls `/generate` and returns a base64 PNG.

Note: ck8t's `cuda_id4_generate` block only sends `prompt`, `preset`,
`aspect_ratio`, `magic`, `seed` to `/generate` — it has no field for
`magic_provider`/`magic_model`, so whichever provider the *server's* `.env`
resolves to (§"Magic prompt" below) is always what's used; you can't override it
per-request from the canvas.

For "every scene/chapter gets its own image" workflows (not just one image), wire
`cuda_id4_generate` inside a real **ForEach Loop** (`for_each`) body instead of
calling it once — see `cuda_id4_storybook.json` in the `ideogram4-storybook`
block's `sample/` folder for a complete worked example (story → per-scene images
→ illustrated PDF).

## Magic prompt — providers, priority, and the verifier warning

Ideogram 4's pipeline produces noticeably better images from a **structured JSON
caption** (`{aspect_ratio, high_level_description, compositional_deconstruction}`)
than from plain text. "Magic prompt" is what produces that — either an LLM call
that rewrites your plain text into the structure, or Ideogram's own hosted
endpoint that does the same.

- `magic: true` (ck8t's default, and the CLI's default unless `--no-magic`):
  prompt → magic-prompt conversion → structured JSON caption → pipeline.
- `magic: false` / `--no-magic`: your plain text goes straight to the pipeline.
  Lower quality, no key needed.

**Providers** (`magic_prompt.py`), in the exact order `MAGIC_PROVIDER` gets
auto-detected when left blank:

| Order | Provider | Env var | Default model | Notes |
|---|---|---|---|---|
| 1 | `ideogram` | `IDEOGRAM_API_KEY` | n/a | Calls Ideogram's own hosted magic-prompt API (`api.ideogram.ai`) — a separate paid service from the open-weight model you're running locally. **Not in `example.env`** — add this key yourself if you want it. |
| 2 | `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-4-8` | |
| 3 | `deepseek` | `DEEPSEEK_API_KEY` (+ optional `DEEPSEEK_BASE_URL`) | `deepseek-chat` | Cheapest key-based option |
| 4 | `openai` | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | `gpt-4o` | Also covers OpenAI-compatible endpoints |
| — | `lmstudio` / `ollama` | none | `local-model` | **Never auto-detected** (no key to detect) — must set `MAGIC_PROVIDER=ollama` (or `lmstudio`) explicitly |

Note this priority order does **not** match `example.env`'s own comment (which
implies DeepSeek is prioritized) — the code checks `ideogram` first, then
`anthropic`, then `deepseek`, then `openai`. If you set more than one key, the
first one in that order wins unless `MAGIC_PROVIDER` is set explicitly.

Either way, the pipeline runs a `caption_verifier` check on the final prompt
purely to see if it parses as JSON. This repo always calls it with
`raise_on_caption_issues=False`, so a plain-text prompt (which obviously isn't
JSON) only ever produces a **warning**, never an error or a blocked/refused
generation:

```
caption verifier flagged prompt[0]: invalid JSON: Expecting value: line 1 column 1 (char 0)
```

Safe to ignore in plain-text mode. If you see this *with* `magic: true`, it means
the magic-prompt call returned something that isn't valid JSON — check
`api_server.log` for the real error (a `Magic prompt failed: ...` HTTP 500 means
the conversion call itself threw, e.g. a missing/invalid key — that's a
provider/key issue, not something this repo blocks on purpose). The `ideogram4`
package also ships an unrelated, unused Hive AI content-moderation module
(`ideogram4/safety.py`, `moderate_prompt`/`moderate_image`) — nothing in this repo
ever calls it, so it has no effect either way.

## API reference

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/load` | Start loading the model into VRAM (non-blocking) |
| `GET` | `/status` | Single-poll current load/generation state |
| `GET` | `/status/stream` | SSE stream of live VRAM%/GPU stats while loading |
| `WS` | `/ws/status` | WebSocket alternative to the SSE stream |
| `POST` | `/generate` | Generate an image — `{prompt, preset, aspect_ratio, magic, seed, return_base64}` |
| `GET` | `/image/{filename}` | Download a previously generated PNG by filename |
| `GET` | `/docs` | FastAPI's interactive Swagger UI |

Presets (`preset`): `TURBO` (12 steps, fastest) · `DEFAULT` (20 steps) ·
`QUALITY` (48 steps, best).

Aspect ratios with dedicated resolutions (anything else falls back to a rounded
custom width/height): `1:1` `16:9` `9:16` `4:3` `3:4` `3:2` `2:3` `4:5` `5:4`.

## Troubleshooting checklist

1. **"Cannot reach server" / "Failed to fetch" from ck8t, or `/docs` won't load
   in a browser** — almost always one of:
   - Port not declared under "Expose HTTP ports" on the pod (§1) — most common.
   - `api_server.py` isn't actually running (the setup script doesn't start it —
     you must run it yourself, §4) or it died because the foreground SSH session
     that launched it disconnected — use `nohup`/`screen`.
   - Pod was recreated since you last copied the URL — the pod ID (and therefore
     the URL) changed (§5).
2. **Generation "failed" with a 500 error** — check `api_server.log` on the pod;
   the HTTPException message includes the real underlying error (model load
   failure, magic-prompt call failure, bad preset name, etc.).
3. **First `/load` call takes minutes** — normal on a fresh pod (model download +
   VRAM load, observed 200-300s). Use `/status/stream` (which `cuda_id4_loader`
   already does) to watch live progress instead of assuming it's hung.
4. **Re-downloading the model every new pod** — `MODEL_PATH`/`save_model.py` won't
   help (§6 above) — set `HF_HOME` to a path under your volume mount instead.
5. **Magic prompt picked a provider you didn't expect** — check the auto-detect
   priority order in "Magic prompt" above; set `MAGIC_PROVIDER` explicitly to stop
   guessing.
