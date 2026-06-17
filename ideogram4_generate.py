#!/usr/bin/env python3
"""
Ideogram 4 via diffusers — NVIDIA CUDA on Linux/RunPod.
Same model weights as the mflux-id4 Mac version (ideogram-ai/ideogram-4-fp8),
different runtime: HuggingFace diffusers + PyTorch CUDA instead of MLX.

Presets (maps to num_inference_steps):
  TURBO    — 12 steps, fast preview
  DEFAULT  — 20 steps, balanced quality (default)
  QUALITY  — 48 steps, best output

Usage:
  python ideogram4_generate.py "a neon Tokyo street at night"
  python ideogram4_generate.py "a fox reading under a lantern" --preset QUALITY
  python ideogram4_generate.py "portrait" --aspect-ratio 9:16
  python ideogram4_generate.py --magic "cartoon cat and crow"     # magic prompt
  python ideogram4_generate.py --json prompt.json                  # pre-built JSON caption
  python ideogram4_generate.py "sketch" --low-vram                 # < 20 GB VRAM: CPU offload
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from math import gcd
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Constants ──────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path(__file__).parent / "outputs"
LOCAL_MODEL  = Path(__file__).parent / "models" / "ideogram4"
HF_MODEL_ID  = "ideogram-ai/ideogram-4-fp8"

PRESETS = {
    "TURBO":   {"steps": 12,  "guidance": 3.5},
    "DEFAULT": {"steps": 20,  "guidance": 3.5},
    "QUALITY": {"steps": 48,  "guidance": 4.0},
}

# Common aspect ratios → width × height (must be multiples of 16)
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


def _load_pipeline(low_vram: bool = False):
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import FluxPipeline

    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else HF_MODEL_ID
    if LOCAL_MODEL.exists():
        print(f"Loading from local model: {LOCAL_MODEL}")
    else:
        print(f"Loading {HF_MODEL_ID} from HuggingFace (first run — large download ~12 GB)...")
        print("  Tip: run `python save_model.py` after this to cache a local copy.")

    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if low_vram:
        # Sequential CPU offload — slower but works on < 20 GB VRAM
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled (low-VRAM mode)")
    else:
        pipe = pipe.to("cuda")

    # Memory savings — minimal quality impact
    pipe.enable_attention_slicing()

    print(f"  Loaded in {time.time() - t0:.1f}s")
    _pipe = pipe
    return pipe


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def _format_prompt(prompt) -> str:
    """Convert plain string or magic-prompt JSON dict to a text string."""
    if isinstance(prompt, str):
        return prompt
    # Structured caption from magic_prompt.convert() — serialize to JSON so the
    # model's text encoder sees the same format it was trained on.
    return json.dumps(prompt, ensure_ascii=False)


def _aspect_size(ratio: str):
    if ratio in ASPECT_SIZES:
        return ASPECT_SIZES[ratio]
    # Accept WxH or W:H notation
    try:
        w, h = (int(x) for x in ratio.replace("x", ":").split(":"))
        # Round to nearest multiple of 16
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
    low_vram: bool = False,
) -> Path:
    """
    Generate an image with Ideogram 4 on CUDA.

    Args:
        prompt:       Plain string or magic-prompt JSON dict.
        preset:       TURBO | DEFAULT | QUALITY.
        aspect_ratio: e.g. "1:1", "16:9", "9:16".
        seed:         Fixed seed for reproducibility.
        output_path:  Save path. Auto-generated if None.
        low_vram:     Enable CPU offload for < 20 GB VRAM.

    Returns:
        Path to the saved PNG.
    """
    pipe = _load_pipeline(low_vram=low_vram)

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
    prompt_preview = prompt_text[:120] if isinstance(prompt_text, str) else str(prompt_text)[:120]
    print(f"Prompt : {prompt_preview}")
    print(f"Preset : {preset}  |  steps={cfg['steps']}  |  {width}x{height}  |  seed={seed}")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.time()
    result = pipe(
        prompt=prompt_text,
        num_inference_steps=cfg["steps"],
        guidance_scale=cfg["guidance"],
        width=width,
        height=height,
        generator=generator,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    image: Image.Image = result.images[0]
    image.save(str(output_path))
    print(f"Saved  → {output_path}")
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Ideogram 4 (diffusers / CUDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ideogram4_generate.py 'a fox under an autumn lantern'\n"
            "  python ideogram4_generate.py 'neon Tokyo street' --preset QUALITY\n"
            "  python ideogram4_generate.py 'portrait' --aspect-ratio 9:16\n"
            "  python ideogram4_generate.py --magic 'cartoon cat and crow'\n"
            "  python ideogram4_generate.py --json prompt.json\n"
            "  python ideogram4_generate.py 'test' --low-vram   # < 20 GB VRAM\n"
        ),
    )
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Plain text prompt")
    parser.add_argument("--json", dest="json_path", default=None, metavar="FILE",
                        help="Path to a pre-built JSON caption file (best quality)")
    parser.add_argument("--magic", action="store_true",
                        help="Auto-convert prompt to magic prompt JSON (like ideogram4 website)")
    parser.add_argument("--magic-provider", default=None,
                        choices=["ideogram", "anthropic", "deepseek", "openai", "lmstudio", "ollama"],
                        help="LLM provider for magic prompt (auto-detected from .env)")
    parser.add_argument("--magic-model", default=None, metavar="MODEL",
                        help="Model for magic prompt (overrides MAGIC_MODEL in .env)")
    parser.add_argument("--magic-base-url", default=None, metavar="URL",
                        help="API base URL override for magic prompt provider")
    parser.add_argument("--save-magic", default=None, metavar="FILE",
                        help="Save the generated magic-prompt JSON caption to this file")
    parser.add_argument("--preset", default="DEFAULT", choices=["TURBO", "DEFAULT", "QUALITY"],
                        help="TURBO=12 steps, DEFAULT=20, QUALITY=48 (default: DEFAULT)")
    parser.add_argument("--aspect-ratio", "-a", default="1:1", metavar="W:H",
                        help="Image aspect ratio e.g. 1:1, 16:9, 9:16 (default: 1:1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed seed for reproducibility")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: outputs/id4_<timestamp>.png)")
    parser.add_argument("--low-vram", action="store_true",
                        help="Enable CPU offload — use this if you have < 20 GB VRAM")
    args = parser.parse_args()

    # ── Resolve prompt / caption ───────────────────────────────────────────────
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
            Path(args.save_magic).write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8")
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
        low_vram=args.low_vram,
    )


if __name__ == "__main__":
    main()
