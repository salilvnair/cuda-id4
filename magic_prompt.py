#!/usr/bin/env python3
"""
Convert a plain-text prompt → Ideogram 4.0 structured JSON caption (magic prompt).

Supported providers (auto-detected from .env, or pass --provider):
  ideogram  — IDEOGRAM_API_KEY  (Ideogram's hosted magic-prompt endpoint — default)
  deepseek  — DEEPSEEK_API_KEY  +  optional DEEPSEEK_BASE_URL
  openai    — OPENAI_API_KEY    +  optional OPENAI_BASE_URL
  anthropic — ANTHROPIC_API_KEY

.env variables:
  IDEOGRAM_API_KEY     — Ideogram key (used for magic-prompt + image generation)
  DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
  DEEPSEEK_BASE_URL    — override DeepSeek endpoint (default: api.deepseek.com/v1)
  OPENAI_BASE_URL      — override OpenAI endpoint
  MAGIC_PROVIDER       — default provider
  MAGIC_MODEL          — default model (not used for ideogram provider)

Usage:
  python magic_prompt.py "cartoon cat and dog talking to a crow" --pretty
  python magic_prompt.py "neon Tokyo" --provider deepseek --pretty
  python magic_prompt.py "a retro diner" --output prompt.json --pretty
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

_V1_PATH = Path(__file__).parent / "magic_prompt_prompts" / "v1.txt"

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "deepseek":  "deepseek-chat",
    "openai":    "gpt-4o",
    "lmstudio":  "local-model",
    "ollama":    "local-model",
}


# ── Template parsing ───────────────────────────────────────────────────────────

def _parse_template(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    sys_match = re.search(r'\[SYSTEM\]\n(.*?)(?=\n\[USER\])', text, re.DOTALL)
    usr_match = re.search(r'\[USER\]\n(.*?)$', text, re.DOTALL)
    system    = sys_match.group(1).strip() if sys_match else text
    user_tmpl = usr_match.group(1).strip() if usr_match else "User idea: {{original_prompt}}"
    return system, user_tmpl


# ── JSON extraction ────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> dict:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw.strip()).strip()
    if not raw.startswith('{'):
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            raw = m.group(0)
    return json.loads(raw)


# ── Provider calls ─────────────────────────────────────────────────────────────

def _call_ideogram_magic_prompt(prompt: str, aspect_ratio: str) -> dict:
    """Call Ideogram's own hosted magic-prompt endpoint."""
    import requests
    api_key = os.environ.get("IDEOGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "IDEOGRAM_API_KEY is not set.\n"
            "Add it to .env:  IDEOGRAM_API_KEY=your-key-here\n"
            "Or use a different provider: --provider deepseek|openai|anthropic"
        )
    ar_ideogram = aspect_ratio.replace(":", "x")
    resp = requests.post(
        "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt",
        headers={"Api-Key": api_key, "Content-Type": "application/json"},
        json={"text_prompt": prompt, "aspect_ratio": ar_ideogram},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    json_prompt = data.get("json_prompt")
    if not json_prompt:
        raise RuntimeError(f"Ideogram magic-prompt API returned no json_prompt: {data}")
    return json_prompt


def _call_anthropic(system: str, user_msg: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=4096, system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text


def _call_deepseek(system: str, user_msg: str, model: str, base_url: str | None) -> str:
    from openai import OpenAI
    resolved = base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
    api_key  = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set in .env")
    client = OpenAI(base_url=resolved, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=4096,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _call_openai_compat(system: str, user_msg: str, model: str, base_url: str | None) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=4096,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── Provider detection ─────────────────────────────────────────────────────────

def _detect_provider(base_url: str | None = None) -> str:
    env_provider = os.environ.get("MAGIC_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider
    if base_url:
        return "openai"
    if os.environ.get("IDEOGRAM_API_KEY", "").strip():
        return "ideogram"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "deepseek"
    if os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    raise RuntimeError(
        "Cannot detect magic-prompt provider. Set at least one in .env:\n"
        "  IDEOGRAM_API_KEY   — Ideogram's own magic-prompt (recommended)\n"
        "  DEEPSEEK_API_KEY   — DeepSeek (cheap, good quality)\n"
        "  ANTHROPIC_API_KEY  — Anthropic Claude\n"
        "  OPENAI_API_KEY     — OpenAI or compatible endpoint\n"
        "Or set MAGIC_PROVIDER=ideogram|deepseek|anthropic|openai explicitly."
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def convert(
    prompt: str,
    aspect_ratio: str = "1:1",
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    debug: bool = False,
) -> dict:
    """
    Convert a plain-text prompt to an Ideogram 4.0 json_prompt dict.

    Returns a dict with keys:
      aspect_ratio, high_level_description, compositional_deconstruction

    The returned dict is ready to pass directly to the Ideogram 4.0 generate
    endpoint as ``json_prompt`` (structured caption, magic-prompt disabled).
    """
    if provider is None:
        provider = _detect_provider(base_url)

    print(f"  Magic prompt provider: {provider}", file=sys.stderr)

    if provider == "ideogram":
        return _call_ideogram_magic_prompt(prompt, aspect_ratio)

    if not model:
        model = os.environ.get("MAGIC_MODEL", "").strip() or _DEFAULT_MODELS.get(provider, "local-model")

    print(f"  Model: {model}", file=sys.stderr)

    system, user_tmpl = _parse_template(_V1_PATH)
    user_msg = (user_tmpl
                .replace("{{aspect_ratio}}", aspect_ratio)
                .replace("{{original_prompt}}", prompt))

    if provider == "anthropic":
        raw = _call_anthropic(system, user_msg, model)
    elif provider == "deepseek":
        raw = _call_deepseek(system, user_msg, model, base_url)
    elif provider in ("openai", "lmstudio", "ollama"):
        raw = _call_openai_compat(system, user_msg, model, base_url)
    else:
        raise ValueError(
            f"Unknown provider {provider!r}.\n"
            "Valid values: ideogram | anthropic | deepseek | openai | lmstudio | ollama"
        )

    if debug:
        print("=== RAW MODEL OUTPUT ===", file=sys.stderr)
        print(raw, file=sys.stderr)
        print("========================", file=sys.stderr)

    return _clean_json(raw)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert plain text → Ideogram 4.0 magic prompt JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Provider config lives in .env (copy example.env → .env).\n"
            "CLI flags override .env values.\n\n"
            "Examples:\n"
            "  python magic_prompt.py 'cartoon cat' --pretty\n"
            "  python magic_prompt.py 'neon Tokyo' --provider deepseek\n"
            "  python magic_prompt.py 'a retro diner' --output prompt.json --pretty\n"
        ),
    )
    parser.add_argument("prompt", help="Plain-text image prompt")
    parser.add_argument("--aspect-ratio", "-a", default="1:1", metavar="W:H",
                        help="Target aspect ratio e.g. 1:1, 16:9, 9:16 (default: 1:1)")
    parser.add_argument("--provider", "-p", default=None,
                        choices=["ideogram", "anthropic", "deepseek", "openai", "lmstudio", "ollama"],
                        help="LLM provider (overrides MAGIC_PROVIDER in .env)")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (overrides MAGIC_MODEL in .env)")
    parser.add_argument("--base-url", default=None, metavar="URL",
                        help="API base URL override")
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
                        help="Save JSON to file instead of stdout")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print the output JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw model output before parsing")
    args = parser.parse_args()

    print(f"Converting via magic prompt (aspect {args.aspect_ratio})…", file=sys.stderr)
    result = convert(
        args.prompt,
        aspect_ratio=args.aspect_ratio,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        debug=args.debug,
    )

    indent = 2 if args.pretty else None
    json_str = json.dumps(result, ensure_ascii=False, indent=indent)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Saved  → {args.output}", file=sys.stderr)
        print(f"Preview: {result.get('high_level_description', '')[:120]}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
