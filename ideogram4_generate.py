#!/usr/bin/env python3
"""
Ideogram 4 via the official ideogram4 package — NVIDIA CUDA / RunPod.
Uses Ideogram4Pipeline (github.com/ideogram-oss/ideogram4), NOT diffusers.

Presets (maps to num_steps):
  TURBO    — 12 steps, fast preview
  DEFAULT  — 20 steps, balanced quality (default)
  QUALITY  — 48 steps, best output

Usage:
  python ideogram4_generate.py "a neon Tokyo street at night"
  python ideogram4_generate.py "a fox reading under a lantern" --preset QUALITY
  python ideogram4_generate.py "portrait" --aspect-ratio 9:16
  python ideogram4_generate.py --magic "cartoon cat and crow"
  python ideogram4_generate.py --json prompt.json
"""
import argparse
import json
import os
import random
import sys
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
LOCAL_MODEL = Path(__file__).parent / "models" / "ideogram4"
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

# ── Model loading ──────────────────────────────────────────────────────────────

_pipe = None


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    print(f"Loading {HF_MODEL_ID} (cached after first run)...")
    config = Ideogram4PipelineConfig(weights_repo=HF_MODEL_ID)

    t0 = time.time()
    pipe = Ideogram4Pipeline.from_pretrained(
        config=config,
        device="cuda",
        dtype=torch.bfloat16,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
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
    print(f"Prompt : {prompt_text[:120]}")
    print(f"Preset : {preset}  |  steps={cfg['steps']}  |  {width}x{height}  |  seed={seed}")

    t0 = time.time()
    # Ideogram4Pipeline returns a list of PIL Images
    result = pipe(
        prompt_text,
        num_steps=cfg["steps"],
        height=height,
        width=width,
        seed=seed,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    image = result[0] if isinstance(result, (list, tuple)) else result
    image.save(str(output_path))
    print(f"Saved  → {output_path}")
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Ideogram 4 (official ideogram4 package / CUDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ideogram4_generate.py 'a fox under an autumn lantern'\n"
            "  python ideogram4_generate.py 'neon Tokyo street' --preset QUALITY\n"
            "  python ideogram4_generate.py 'portrait' --aspect-ratio 9:16\n"
            "  python ideogram4_generate.py --magic 'cartoon cat and crow'\n"
            "  python ideogram4_generate.py --json prompt.json\n"
        ),
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Plain text prompt")
    parser.add_argument("--json", dest="json_path", default=None, metavar="FILE",
                        help="Pre-built JSON caption file")
    parser.add_argument("--magic", action="store_true",
                        help="Auto-convert prompt to structured magic prompt JSON")
    parser.add_argument("--magic-provider", default=None,
                        choices=["ideogram", "anthropic", "deepseek", "openai", "lmstudio", "ollama"])
    parser.add_argument("--magic-model", default=None, metavar="MODEL")
    parser.add_argument("--magic-base-url", default=None, metavar="URL")
    parser.add_argument("--save-magic", default=None, metavar="FILE",
                        help="Save the generated magic-prompt JSON to this file")
    parser.add_argument("--preset", default="DEFAULT", choices=["TURBO", "DEFAULT", "QUALITY"],
                        help="TURBO=12 steps, DEFAULT=20, QUALITY=48 (default: DEFAULT)")
    parser.add_argument("--aspect-ratio", "-a", default="1:1", metavar="W:H",
                        help="Aspect ratio e.g. 1:1, 16:9, 9:16 (default: 1:1)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: outputs/id4_<timestamp>.png)")
    args = parser.parse_args()

    if args.json_path:
        with open(args.json_path) as f:
            prompt = json.load(f)
        print(f"Using JSON caption from {args.json_path}")
    elif args.magic:
        if not args.prompt:
            parser.error("--magic requires a prompt argument")
        print(f"Prompt  : {args.prompt[:120]}")
        print("Running magic prompt…", file=sys.stderr)
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
            print(f"Magic prompt saved → {args.save_magic}", file=sys.stderr)
    elif args.prompt:
        prompt = args.prompt
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
