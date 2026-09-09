from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from tools import _parse_json_content, normalize_vlm_identity_result
from training.common import read_jsonl, write_jsonl
from training.metrics import evaluate_predictions


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def query_endpoint(
    client: Any,
    base_url: str,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    max_tokens: int,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": str(sample["prompt"])},
                {"type": "image_url", "image_url": {"url": _data_url(Path(sample["image"]))}},
            ]}],
        },
    )
    response.raise_for_status()
    content = str(response.json()["choices"][0]["message"]["content"])
    parsed = _parse_json_content(content)
    return {
        **sample,
        "model": model,
        "raw_response": content,
        "json_valid": bool(parsed),
        "prediction": normalize_vlm_identity_result(parsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base or LoRA VLM through its OpenAI-compatible endpoint")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-url", default="http://localhost:7890/v1")
    parser.add_argument("--api-key", default="abc123")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("httpx is required for endpoint evaluation") from exc

    samples = [row for row in read_jsonl(args.dataset) if str(row.get("split", "train")) == args.split]
    if args.limit > 0:
        samples = samples[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for index, sample in enumerate(samples, start=1):
            predictions.append(query_endpoint(
                client,
                args.base_url,
                args.api_key,
                args.model,
                sample,
                args.max_tokens,
                args.enable_thinking,
            ))
            print(f"[{index}/{len(samples)}] {sample['sample_id']}")
    metrics = {**evaluate_predictions(predictions), "model": args.model, "split": args.split}
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
