from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.vlm_dataset import JsonlSplitDataset, VLMDataCollator


def _training_arguments(TrainingArguments: Any, values: dict[str, Any]) -> Any:
    try:
        return TrainingArguments(**values)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        values["evaluation_strategy"] = values.pop("eval_strategy")
        return TrainingArguments(**values)


def _target_module_names(model: Any, suffixes: set[str]) -> list[str]:
    blocked = ("visual", "vision", "image_tower", "vision_tower")
    names: list[str] = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in suffixes:
            continue
        if any(token in name.lower() for token in blocked):
            continue
        if hasattr(module, "weight"):
            names.append(name)
    if not names:
        raise RuntimeError(f"no LoRA target modules found for {sorted(suffixes)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Qwen3-VL vessel evidence LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('install training dependencies with: pip install -e ".[train]"') from exc

    train_dataset = JsonlSplitDataset(args.dataset, args.train_split)
    val_dataset = JsonlSplitDataset(args.dataset, args.val_split)
    if not train_dataset:
        raise SystemExit(f"no samples found in split {args.train_split!r}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if not args.no_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    if not args.no_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_modules = _target_module_names(model, set(args.target_modules.split(",")))
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_values = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if len(val_dataset) else "no",
        "load_best_model_at_end": bool(len(val_dataset)),
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
    }
    trainer = Trainer(
        model=model,
        args=_training_arguments(TrainingArguments, training_values),
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) else None,
        data_collator=VLMDataCollator(processor),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    manifest = {
        "base_model": args.model,
        "dataset": str(args.dataset.resolve()),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "qlora": not args.no_qlora,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": target_modules,
    }
    (args.output_dir / "vessel_lora_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

