#!/usr/bin/env python3
"""
Download and save Ideogram 4 weights locally for offline/faster RunPod runs.

Run ONCE after your first setup:
  python save_model.py

Saves to ./models/ideogram4/ (~12–24 GB depending on dtype).
After this, ideogram4_generate.py loads from that local path — no internet needed.

On RunPod with persistent storage, save to /workspace instead:
  python save_model.py --save-dir /workspace/models/ideogram4
"""
import argparse
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Download and cache Ideogram 4 model locally")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save the model (default: ./models/ideogram4)")
    parser.add_argument("--model-id", default="ideogram-ai/ideogram-4-fp8",
                        help="HuggingFace model ID (default: ideogram-ai/ideogram-4-fp8)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else Path(__file__).parent / "models" / "ideogram4"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id} from HuggingFace...")
    print(f"Save destination: {save_dir}")
    print("This may take a while — the model is ~12 GB.")
    print()

    from diffusers import FluxPipeline

    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"  Downloaded in {time.time() - t0:.1f}s")

    print(f"\nSaving to {save_dir} ...")
    t1 = time.time()
    pipe.save_pretrained(str(save_dir))
    print(f"  Saved in {time.time() - t1:.1f}s")

    size_gb = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file()) / 1e9
    print(f"  Total size on disk: {size_gb:.1f} GB")

    print(f"\nDone. Future runs will load from {save_dir}")
    print("\nYou can now free the HuggingFace cache to reclaim disk space:")
    print(f"  rm -rf ~/.cache/huggingface/hub/models--{args.model_id.replace('/', '--')}")


if __name__ == "__main__":
    main()
