#!/usr/bin/env python3
"""
Download and cache Ideogram 4 weights for offline / faster RunPod runs.

Run ONCE after your first setup:
  python save_model.py

Downloads to ./models/ideogram4/ (~28 GB).
Set MODEL_PATH in .env to point there and subsequent runs skip the download.

On RunPod with persistent storage:
  python save_model.py --save-dir /workspace/models/ideogram4
  # then set MODEL_PATH=/workspace/models/ideogram4 in .env
"""
import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token


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
    print("This may take a while — the model is ~28 GB.")
    print()

    from huggingface_hub import snapshot_download

    t0 = time.time()
    local_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(save_dir),
    )
    elapsed = time.time() - t0
    print(f"\nDownloaded in {elapsed:.1f}s → {local_path}")

    size_gb = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file()) / 1e9
    print(f"Total size on disk: {size_gb:.1f} GB")
    print(f"\nDone. Set this in .env to use the local copy:")
    print(f"  MODEL_PATH={save_dir}")


if __name__ == "__main__":
    main()
