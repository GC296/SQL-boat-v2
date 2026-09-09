from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a vessel LoRA adapter into its non-AWQ base model")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit('install training dependencies with: pip install -e ".[train]"') from exc

    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="4GB")
    AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
