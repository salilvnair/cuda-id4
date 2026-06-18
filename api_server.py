#!/usr/bin/env python3
"""
Ideogram 4 — HTTP API server.

Flow:
  1. POST /load              → start loading model into VRAM (non-blocking)
  2. GET  /status/stream     → SSE stream of live VRAM % / GPU stats while loading
     GET  /status            → single-poll current state
     WS   /ws/status         → WebSocket alternative to SSE
  3. POST /generate          → generate image (queues if model still loading)
  4. GET  /health            → quick liveness check

Start:
  .venv/bin/python api_server.py            # port 8080
  .venv/bin/python api_server.py --port 7860

Expose port in RunPod → HTTP Services → add port 8080.
Your base URL: https://<pod-id>-8080.proxy.runpod.net
"""
import argparse
import asyncio
import base64
import json
import os
import random
import subprocess
import sys
import threading
import time
from math import gcd
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── HuggingFace auth ────────────────────────────────────────────────────────────
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from ideogram4_generate import (
    _load_pipeline, _format_prompt, _aspect_size,
    generate, PRESETS, OUTPUT_DIR,
)

# ── State ───────────────────────────────────────────────────────────────────────

MODEL_GB = 28.0   # approximate VRAM footprint for progress bar

class _State:
    status: str = "idle"          # idle | loading | ready | error
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

state = _State()
_gen_lock = asyncio.Lock()

# ── VRAM / GPU helpers ──────────────────────────────────────────────────────────

def _vram_gb() -> float:
    try:
        return torch.cuda.memory_allocated(0) / 1024**3
    except Exception:
        return 0.0

def _nvidia_smi() -> Optional[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().split(",")
        return {
            "gpu_util_pct": int(out[0].strip()),
            "vram_used_mb": int(out[1].strip()),
            "vram_total_mb": int(out[2].strip()),
            "temp_c": int(out[3].strip()),
        }
    except Exception:
        return None

def _snapshot() -> dict:
    """Return a status snapshot suitable for SSE / WebSocket / polling."""
    elapsed = round(time.time() - state.started_at, 1) if state.started_at else 0
    vram = _vram_gb()
    percent = round(min(vram / MODEL_GB * 100, 100.0), 1)
    smi = _nvidia_smi()
    d = {
        "status": state.status,
        "percent": percent,
        "vram_used_gb": round(vram, 2),
        "vram_total_gb": MODEL_GB,
        "elapsed_s": elapsed,
    }
    if smi:
        d["gpu_util_pct"] = smi["gpu_util_pct"]
        d["temp_c"] = smi["temp_c"]
    if state.error:
        d["error"] = state.error
    if state.status == "ready" and state.finished_at:
        d["load_time_s"] = round(state.finished_at - state.started_at, 1)
    return d

# ── Model loader ────────────────────────────────────────────────────────────────

def _do_load():
    state.status = "loading"
    state.started_at = time.time()
    state.error = None
    try:
        _load_pipeline()
        state.status = "ready"
        state.finished_at = time.time()
        elapsed = round(state.finished_at - state.started_at, 1)
        print(f"[load] Model ready in {elapsed}s ✓", flush=True)
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        print(f"[load] ERROR: {exc}", flush=True)

# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ideogram 4 API",
    description="Generate images with Ideogram 4 FP8 on NVIDIA CUDA.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_status": state.status}


@app.post("/load")
async def load_model():
    """Start loading the model into VRAM. Non-blocking — returns immediately.
    Poll /status or stream /status/stream to watch progress."""
    if state.status == "ready":
        return {"message": "Model already loaded", "status": "ready"}
    if state.status == "loading":
        return {"message": "Already loading", "status": "loading"}
    threading.Thread(target=_do_load, daemon=True).start()
    return {"message": "Loading started", "status": "loading"}


@app.get("/status")
async def status():
    """Single-poll current model loading state."""
    return _snapshot()


@app.get("/status/stream")
async def status_stream():
    """SSE stream — sends a JSON event every 300 ms while loading.
    Closes automatically once model is ready (or errored).

    Connect from CK8T as a streaming API call; parse each `data:` line as JSON.
    Use `percent` (0-100) for a progress bar.

    Example event:
      data: {"status":"loading","percent":45.2,"vram_used_gb":12.7,"elapsed_s":82}
    """
    async def _generator():
        while True:
            snap = _snapshot()
            yield f"data: {json.dumps(snap)}\n\n"
            if snap["status"] in ("ready", "error"):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering on RunPod proxy
        },
    )


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """WebSocket alternative to SSE — sends the same JSON snapshot every 300 ms."""
    await websocket.accept()
    try:
        while True:
            snap = _snapshot()
            await websocket.send_text(json.dumps(snap))
            if snap["status"] in ("ready", "error"):
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Generation ──────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Plain text or pre-built JSON caption")
    preset: str = Field("DEFAULT", description="TURBO | DEFAULT | QUALITY")
    aspect_ratio: str = Field("1:1", description="1:1  16:9  9:16  4:3  3:2 …")
    seed: Optional[int] = Field(None)
    magic: bool = Field(True, description="Auto-convert prompt via LLM (magic prompt)")
    magic_provider: Optional[str] = Field(None)
    magic_model: Optional[str] = Field(None)
    return_base64: bool = Field(True, description="Include base64 PNG in response")


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    """Generate an image. Returns base64 PNG + metadata.

    If the model is still loading this will wait until it's ready
    (or raise 503 if it errored during load).
    """
    # Wait up to 600 s for model to become ready
    deadline = time.time() + 600
    while state.status == "loading" and time.time() < deadline:
        await asyncio.sleep(1)

    if state.status == "idle":
        raise HTTPException(503, "Model not loaded — call POST /load first")
    if state.status == "error":
        raise HTTPException(503, f"Model failed to load: {state.error}")
    if state.status != "ready":
        raise HTTPException(503, "Model not ready")

    if req.preset.upper() not in PRESETS:
        raise HTTPException(400, f"Invalid preset '{req.preset}'. Use TURBO, DEFAULT, or QUALITY.")

    # Resolve prompt
    if req.magic and not req.prompt.strip().startswith("{"):
        try:
            from magic_prompt import convert as magic_convert
            w, h = _aspect_size(req.aspect_ratio)
            d = gcd(w, h); ar_str = f"{w // d}:{h // d}"
            prompt = magic_convert(
                req.prompt,
                aspect_ratio=ar_str,
                provider=req.magic_provider,
                model=req.magic_model,
            )
        except Exception as exc:
            raise HTTPException(500, f"Magic prompt failed: {exc}")
    else:
        prompt = req.prompt

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    t0 = time.time()
    async with _gen_lock:
        try:
            loop = asyncio.get_event_loop()
            output_path = await loop.run_in_executor(
                None,
                lambda: generate(
                    prompt=prompt,
                    preset=req.preset.upper(),
                    aspect_ratio=req.aspect_ratio,
                    seed=seed,
                ),
            )
        except Exception as exc:
            raise HTTPException(500, f"Generation failed: {exc}")

    elapsed = round(time.time() - t0, 2)
    prompt_text = _format_prompt(prompt)

    image_b64 = None
    if req.return_base64:
        with open(output_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

    return {
        "image_b64": image_b64,
        "output_path": str(output_path),
        "filename": Path(output_path).name,
        "seed": seed,
        "preset": req.preset.upper(),
        "aspect_ratio": req.aspect_ratio,
        "elapsed_s": elapsed,
        "prompt_used": prompt_text[:300],
    }


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Download a generated image by filename (e.g. id4_1234567890.png)."""
    path = OUTPUT_DIR / filename
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(404, "Image not found")
    return StreamingResponse(open(path, "rb"), media_type="image/png")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Ideogram 4 API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--auto-load", action="store_true",
                        help="Start loading the model immediately on startup")
    args = parser.parse_args()

    if args.auto_load:
        print("[startup] --auto-load: starting model load immediately")
        threading.Thread(target=_do_load, daemon=True).start()

    print(f"Ideogram 4 API → http://{args.host}:{args.port}")
    print(f"Docs            → http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
