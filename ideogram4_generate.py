#!/usr/bin/env python3
"""
Ideogram 4 via the official ideogram4 package — NVIDIA CUDA / RunPod.

Presets:  TURBO=12 steps  DEFAULT=20 steps  QUALITY=48 steps

Magic prompt is ON by default — it converts plain text to a structured JSON caption
(compositional_deconstruction format) for best quality. Requires an LLM key in .env.
Use --no-magic to pass plain text directly to the pipeline.

The model takes ~2-3 min to load into VRAM once. Use interactive mode (-i) to keep
it warm and generate multiple images without reloading.

Usage:
  python ideogram4_generate.py "a neon Tokyo street at night"
  python ideogram4_generate.py "portrait" --preset QUALITY --aspect-ratio 9:16
  python ideogram4_generate.py --no-magic "cartoon cat and crow"
  python ideogram4_generate.py --json prompt.json
  python ideogram4_generate.py -i                  # interactive loop — loads once
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from math import gcd
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── HuggingFace authentication ─────────────────────────────────────────────────
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

# ── Constants ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path(__file__).parent / "outputs"
HF_MODEL_ID = "ideogram-ai/ideogram-4-fp8"

PRESETS = {
    "TURBO":   {"steps": 12},
    "DEFAULT": {"steps": 20},
    "QUALITY": {"steps": 48},
}

ASPECT_SIZES = {
    "1:1":  (1024, 1024),
    "16:9": (1360, 768),
    "9:16": (768, 1360),
    "4:3":  (1152, 864),
    "3:4":  (864, 1152),
    "3:2":  (1152, 768),
    "2:3":  (768, 1152),
    "4:5":  (896, 1120),
    "5:4":  (1120, 896),
}

# ── Pretty-print helpers ───────────────────────────────────────────────────────

W = 62  # banner width

def _hr(char="─"): print(char * W)
def _banner(title): print(f"\n{'━' * W}\n  {title}\n{'━' * W}")
def _section(title): print(f"\n── {title} {'─' * (W - len(title) - 4)}")

def _bar(current, total, width=28):
    pct = min(current / total, 1.0) if total > 0 else 0
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct*100:5.1f}%"

def _gpu_info():
    """Return (name, total_gb, cuda_ver) from torch."""
    props = torch.cuda.get_device_properties(0)
    name = props.name
    total_gb = props.total_memory / 1024**3
    cuda_ver = torch.version.cuda or "?"
    return name, total_gb, cuda_ver

def _vram_gb():
    """Currently allocated VRAM in GB."""
    return torch.cuda.memory_allocated() / 1024**3

def _nvidia_smi_stats():
    """Return (util_pct, used_mb, temp_c) via nvidia-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip().split(",")
        return int(out[0].strip()), int(out[1].strip()), int(out[2].strip())
    except Exception:
        return None

def _print_gpu_header():
    gpu_name, total_gb, cuda_ver = _gpu_info()
    _banner("Ideogram 4 — CUDA Generation")
    print(f"  GPU   : {gpu_name}")
    print(f"  VRAM  : {total_gb:.1f} GB total")
    print(f"  CUDA  : {cuda_ver}")
    print(f"  Torch : {torch.__version__}")

# ── Loading monitor ────────────────────────────────────────────────────────────

def _run_loading_monitor(done_event, model_gb=28.0):
    """Background thread: live VRAM-fill progress bar while model loads."""
    frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
    i = 0
    t0 = time.time()
    peak = 0.0
    while not done_event.is_set():
        elapsed = time.time() - t0
        vram = _vram_gb()
        peak = max(peak, vram)
        smi = _nvidia_smi_stats()
        util = f"  GPU util: {smi[0]:3d}%  Temp: {smi[2]}°C" if smi else ""
        bar = _bar(vram, model_gb)
        line = f"\r  {frames[i % len(frames)]}  VRAM {vram:5.1f}/{model_gb:.0f} GB  {bar}  {elapsed:5.0f}s{util}"
        print(line, end='', flush=True)
        i += 1
        time.sleep(0.3)
    elapsed = time.time() - t0
    vram = _vram_gb()
    print(f"\r  ✓  VRAM {vram:5.1f} GB  {_bar(vram, model_gb)}  loaded in {elapsed:.1f}s          ")
    return peak

# ── Inference monitor ──────────────────────────────────────────────────────────

def _run_inference_monitor(done_event, total_steps):
    """Background thread: elapsed time + GPU stats during denoising."""
    frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
    i = 0
    t0 = time.time()
    while not done_event.is_set():
        elapsed = time.time() - t0
        smi = _nvidia_smi_stats()
        if smi:
            util, used_mb, temp = smi
            used_gb = used_mb / 1024
            stats = f"  GPU: {util:3d}%  VRAM: {used_gb:.1f} GB  Temp: {temp}°C"
        else:
            stats = ""
        line = f"\r  {frames[i % len(frames)]}  Generating… {elapsed:5.1f}s  ({total_steps} steps){stats}"
        print(line, end='', flush=True)
        i += 1
        time.sleep(0.3)
    elapsed = time.time() - t0
    smi = _nvidia_smi_stats()
    stats = f"  GPU: {smi[0]}%  Temp: {smi[2]}°C" if smi else ""
    print(f"\r  ✓  Done in {elapsed:.1f}s{stats}                                        ")
    return elapsed

# ── Step-level callback (best-effort) ─────────────────────────────────────────

class _StepTracker:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.t0 = time.time()

    def __call__(self, step=None, **kwargs):
        self.current = (step or 0) + 1
        elapsed = time.time() - self.t0
        per_step = elapsed / self.current if self.current else 0
        remaining = per_step * (self.total - self.current)
        bar = _bar(self.current, self.total, width=24)
        line = (f"\r  Step {self.current:2d}/{self.total}  {bar}"
                f"  {elapsed:.1f}s elapsed  ~{remaining:.0f}s left  ")
        print(line, end='', flush=True)


# ── Model loading ──────────────────────────────────────────────────────────────

_pipe = None


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    _section("Loading Model")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  dtype : bfloat16  |  device: cuda")
    print()

    config = Ideogram4PipelineConfig(weights_repo=HF_MODEL_ID)

    done = threading.Event()
    monitor = threading.Thread(target=_run_loading_monitor, args=(done,), daemon=True)
    monitor.start()

    pipe = Ideogram4Pipeline.from_pretrained(
        config=config,
        device="cuda",
        dtype=torch.bfloat16,
    )

    done.set()
    monitor.join()

    _pipe = pipe
    return pipe


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False)


def _aspect_size(ratio: str):
    if ratio in ASPECT_SIZES:
        return ASPECT_SIZES[ratio]
    try:
        w, h = (int(x) for x in ratio.replace("x", ":").split(":"))
        return (w // 16 * 16, h // 16 * 16)
    except ValueError:
        print(f"  Warning: unknown aspect ratio '{ratio}', defaulting to 1:1", file=sys.stderr)
        return (1024, 1024)


# ── Core generation ────────────────────────────────────────────────────────────

def generate(
    prompt,
    preset: str = "DEFAULT",
    aspect_ratio: str = "1:1",
    seed: int | None = None,
    output_path: Path | None = None,
) -> Path:
    t_total = time.time()
    pipe = _load_pipeline()

    cfg = PRESETS.get(preset.upper(), PRESETS["DEFAULT"])
    width, height = _aspect_size(aspect_ratio)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"id4_{int(time.time())}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_text = _format_prompt(prompt)

    _section("Generating")
    is_json = prompt_text.strip().startswith('{')
    prompt_mode = "JSON caption" if is_json else "plain text (use --magic for best quality)"
    print(f"  Prompt : {prompt_text[:100]}{'…' if len(prompt_text) > 100 else ''}")
    print(f"  Mode   : {prompt_mode}")
    print(f"  Preset : {preset.upper()}  |  steps={cfg['steps']}  |  {width}×{height}  |  seed={seed}")
    print()

    done_inf = threading.Event()
    monitor_inf = threading.Thread(
        target=_run_inference_monitor, args=(done_inf, cfg["steps"]), daemon=True
    )
    monitor_inf.start()

    # Pipeline requires JSON-structured captions (same format magic prompt produces).
    # raise_on_caption_issues=False lets plain-text prompts pass through — quality
    # is lower than a structured caption but it works. Use --magic for best results.
    result = pipe(
        prompt_text,
        num_steps=cfg["steps"],
        height=height,
        width=width,
        seed=seed,
        raise_on_caption_issues=False,
    )

    done_inf.set()
    monitor_inf.join()

    image = result[0] if isinstance(result, (list, tuple)) else result
    image.save(str(output_path))

    total_elapsed = time.time() - t_total
    smi = _nvidia_smi_stats()

    _section("Done")
    print(f"  Saved    : {output_path}")
    print(f"  Total    : {total_elapsed:.1f}s")
    if smi:
        print(f"  VRAM     : {smi[1] / 1024:.1f} GB used  |  GPU {smi[0]}%  |  {smi[2]}°C")
    _hr()
    print()
    return output_path


# ── Interactive loop ───────────────────────────────────────────────────────────

def _interactive_loop(args):
    """Load the model once, then generate images in a loop until the user quits."""
    _load_pipeline()  # warm up now so first generation is instant

    print()
    _hr()
    print("  Interactive mode — model loaded. Type a prompt and press Enter.")
    print("  Commands:  preset TURBO|DEFAULT|QUALITY  |  ar 16:9  |  quit")
    _hr()

    preset = args.preset
    aspect_ratio = args.aspect_ratio
    n = 0

    while True:
        try:
            raw = input("\n  Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not raw:
            continue
        lower = raw.lower()
        if lower in ("quit", "exit", "q"):
            print("  Bye.")
            break
        if lower.startswith("preset "):
            val = raw.split(None, 1)[1].upper()
            if val in PRESETS:
                preset = val
                print(f"  Preset → {preset}")
            else:
                print(f"  Unknown preset '{val}'. Options: TURBO DEFAULT QUALITY")
            continue
        if lower.startswith("ar "):
            aspect_ratio = raw.split(None, 1)[1]
            print(f"  Aspect ratio → {aspect_ratio}")
            continue

        n += 1
        if args.no_magic:
            prompt = raw
        else:
            print(f"  Running magic prompt…")
            from magic_prompt import convert as magic_convert
            w, h = _aspect_size(aspect_ratio)
            d = gcd(w, h); ar = f"{w // d}:{h // d}"
            prompt = magic_convert(
                raw,
                aspect_ratio=ar,
                provider=args.magic_provider,
                model=args.magic_model,
                base_url=args.magic_base_url,
            )

        generate(
            prompt=prompt,
            preset=preset,
            aspect_ratio=aspect_ratio,
            seed=args.seed,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    _print_gpu_header()

    parser = argparse.ArgumentParser(
        description="Generate images with Ideogram 4 (CUDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--json", dest="json_path", default=None, metavar="FILE")
    parser.add_argument("--no-magic", dest="no_magic", action="store_true",
                        help="Skip magic prompt — pass plain text directly to the pipeline")
    parser.add_argument("--magic-provider", default=None,
                        choices=["ideogram", "anthropic", "deepseek", "openai", "lmstudio", "ollama"])
    parser.add_argument("--magic-model", default=None, metavar="MODEL")
    parser.add_argument("--magic-base-url", default=None, metavar="URL")
    parser.add_argument("--save-magic", default=None, metavar="FILE")
    parser.add_argument("--preset", default="DEFAULT", choices=["TURBO", "DEFAULT", "QUALITY"])
    parser.add_argument("--aspect-ratio", "-a", default="1:1", metavar="W:H")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive loop — load model once, generate many images")
    args = parser.parse_args()

    if args.interactive:
        _interactive_loop(args)
        return

    if args.json_path:
        with open(args.json_path) as f:
            prompt = json.load(f)
        print(f"\n  Using JSON caption from {args.json_path}")
    elif args.no_magic:
        if not args.prompt:
            parser.error("a prompt argument is required")
        prompt = args.prompt
        print(f"\n  Using plain text prompt (magic disabled)")
    elif args.prompt:
        print(f"\n  Running magic prompt for: {args.prompt[:80]}")
        from magic_prompt import convert as magic_convert
        w, h = _aspect_size(args.aspect_ratio)
        d = gcd(w, h); ar = f"{w // d}:{h // d}"
        prompt = magic_convert(
            args.prompt,
            aspect_ratio=ar,
            provider=args.magic_provider,
            model=args.magic_model,
            base_url=args.magic_base_url,
        )
        if args.save_magic:
            Path(args.save_magic).write_text(
                json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  Magic prompt saved → {args.save_magic}")
    else:
        parser.print_help()
        sys.exit(1)

    generate(
        prompt=prompt,
        preset=args.preset,
        aspect_ratio=args.aspect_ratio,
        seed=args.seed,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
