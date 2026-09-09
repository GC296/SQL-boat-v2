"""Run a monolithic native Video-VLM baseline without the ECEM pipeline."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_native_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle) or {}
    llm = dict(root.get("llm", {}) or {})
    native = dict(root.get("native_vlm", {}) or {})
    llm.update(native)
    if "max_retries" not in native:
        llm["max_retries"] = 0
    if "max_tokens" not in native:
        llm["max_tokens"] = 2048
    if "request_timeout_seconds" not in native:
        llm["request_timeout_seconds"] = 600
    llm.setdefault("input_mode", "video_url")
    return llm


def load_archive_context(path: Path) -> str:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = str(row.get("hull_number", "") or row.get("archive_identity", "")).strip()
            description = str(row.get("description", "") or "").strip()
            if not identity or not description:
                continue
            prototype_id = str(row.get("prototype_id", "") or f"{identity}_{len(grouped[identity]) + 1:02d}").strip()
            grouped[identity].append((prototype_id, description))
    if not grouped:
        raise ValueError(f"No archive identities with descriptions found in {path}")
    blocks = []
    for identity, prototypes_for_identity in sorted(grouped.items()):
        prototypes = "\n".join(
            f"  - 档案条目 {prototype_id}: {description}"
            for prototype_id, description in prototypes_for_identity
        )
        blocks.append(f"注册身份：{identity}\n全部文字档案条目：\n{prototypes}")
    return "\n\n".join(blocks)


def build_task_prompt(archive_context: str, frame_timeline: list[float] | None = None) -> str:
    timeline = ""
    if frame_timeline:
        timeline = f"\n视频以覆盖完整时段的 {len(frame_timeline)} 帧图像按时间顺序输入，请将它们视为同一段连续视频。\n"
    return f"""/no_think
你是完成开放集船舶身份核验任务的唯一感知与推理模型。

输入包括一段完整海事视频，以及下方完整的注册船舶文字档案库。除这些输入外，不存在检测器、跟踪器、实体重关联、向量检索、Embedding、外部工具或后处理模块。你必须独立完成整个任务。

任务要求：
1. 找出视频中出现的所有不同物理船舶，每艘物理船只输出一条记录。
2. 同一艘船在视频中消失后重新出现，或者从不同视角、不同距离再次出现时，必须合并为同一个 vessel_id。
3. 综合整段视频中的所有可见证据，包括舷号、船体颜色、船型、上层建筑、窗户、船首船尾、推进装置、固定设备、文字和显著标志。
4. 将每艘船与下方所有注册档案直接比较，不允许因为存在相似档案就强制匹配。
5. 只有充分支持唯一注册身份时才返回 decision=\"known\"，并填写对应 archive_identity。
6. 与全部注册身份均不一致时返回 decision=\"out_of_archive\"，archive_identity 必须为空字符串。
7. 证据不足、存在冲突或无法唯一确定时返回 decision=\"uncertain\"，archive_identity 必须为空字符串。
8. visual_summary 只总结模型在视频中实际观察到、用于区分身份的稳定视觉证据。
9. reason 必须说明最终身份结论由哪些视频证据和档案差异支持。
10. 完成所有船舶的最终核验后设置 task_complete=true。
11. 只返回一个 JSON 对象，不输出 Markdown、分析过程、<think> 标签、档案逐条复述或额外解释。
12. 输出的第一个字符必须是 {{，最后一个字符必须是 }}；visual_summary 和 reason 分别不超过 100 个汉字。
{timeline}
严格使用以下 JSON 结构：
{{
  "vessels": [
    {{
      "vessel_id": "v1",
      "visual_summary": "视频中实际观察到的稳定身份特征",
      "hull_number": "可靠可见的真实舷号，无法确认时为空字符串",
      "decision": "known|out_of_archive|uncertain",
      "archive_identity": "known 时填写唯一注册身份，否则为空字符串",
      "reason": "支持最终结论的视频证据与档案差异"
    }}
  ],
  "task_complete": true
}}

完整注册船舶文字档案库：
{archive_context}
"""


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return result if isinstance(result, dict) else {}


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frame_count, "duration_s": duration, "width": width, "height": height}


def sample_video_frames(path: Path, *, sample_fps: float, max_frames: int, max_side: int, jpeg_quality: int) -> tuple[list[str], list[float]]:
    metadata = video_metadata(path)
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    desired = max(1, min(max_frames, int(metadata["duration_s"] * sample_fps) + 1))
    if desired == 1:
        indices = [0]
    else:
        indices = [round(index * (frame_count - 1) / (desired - 1)) for index in range(desired)]
    capture = cv2.VideoCapture(str(path))
    encoded_frames: list[str] = []
    timeline: list[float] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue
            encoded_frames.append("data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii"))
            timeline.append(frame_index / fps)
    finally:
        capture.release()
    if not encoded_frames:
        raise RuntimeError(f"No frames sampled from video: {path}")
    return encoded_frames, timeline


def video_reference(path: Path, manifest_row: dict[str, Any], config: dict[str, Any]) -> str:
    if manifest_row.get("video_url"):
        return str(manifest_row["video_url"])
    template = str(config.get("video_url_template", "") or "")
    if template:
        return template.format(video_id=manifest_row.get("video_id", path.stem), filename=path.name, video_path=str(path))
    if bool(config.get("use_file_url", True)):
        return path.resolve().as_uri()
    max_inline_mb = float(config.get("max_inline_video_mb", 64))
    if path.stat().st_size > max_inline_mb * 1024 * 1024:
        raise ValueError(f"Video exceeds max_inline_video_mb and no accessible URL is configured: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return f"data:{mime_type};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_content(path: Path, manifest_row: dict[str, Any], config: dict[str, Any], archive_context: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(config.get("input_mode", "video_url"))
    if mode == "video_url":
        prompt = build_task_prompt(archive_context)
        reference = video_reference(path, manifest_row, config)
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": reference}},
        ], {"input_mode": mode, "sampled_frames": 0}
    if mode != "sampled_frames":
        raise ValueError(f"Unsupported native_vlm.input_mode: {mode}")
    frames, timeline = sample_video_frames(
        path,
        sample_fps=max(0.01, float(config.get("sample_fps", 1.0))),
        max_frames=max(1, int(config.get("max_frames", 64))),
        max_side=max(224, int(config.get("image_max_side", 896))),
        jpeg_quality=min(100, max(30, int(config.get("jpeg_quality", 85)))),
    )
    content = [{"type": "text", "text": build_task_prompt(archive_context, timeline)}]
    content.extend({"type": "image_url", "image_url": {"url": frame}} for frame in frames)
    return content, {"input_mode": mode, "sampled_frames": len(frames), "frame_timeline_s": timeline}


def call_model(content: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, dict[str, Any], int, float]:
    import httpx

    api_url = f"{str(config.get('base_url', 'http://localhost:7890/v1')).rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": config.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": int(config.get("max_tokens", 2048)),
        "messages": [{"role": "user", "content": content}],
    }
    if "enable_thinking" in config:
        payload["chat_template_kwargs"] = {
            "enable_thinking": bool(config["enable_thinking"]),
        }
    if bool(config.get("json_response_format", False)):
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {config.get('api_key', 'abc123')}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(connect=20.0, read=float(config.get("request_timeout_seconds", 600)), write=600.0, pool=20.0)
    attempts = max(1, int(config.get("max_retries", 0)) + 1)
    started = time.perf_counter()
    last_error: Exception | None = None
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(1, attempts + 1):
            try:
                response = client.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                content_text = str(body["choices"][0]["message"]["content"])
                return content_text, body.get("usage", {}) or {}, attempt, (time.perf_counter() - started) * 1000.0
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == attempts:
                    break
    raise RuntimeError(f"Native VLM request failed after {attempts} attempt(s): {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="experiment_outputs", type=Path)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and input metadata without calling the model")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    run_dir = (output_root / args.run_name).resolve()
    if Path(run_dir.parent) != output_root or run_dir == output_root:
        raise ValueError("run-name must resolve to a direct child of output-root")
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    config = load_native_config(args.config)
    archive_context = load_archive_context(args.archive)
    manifest = [row for row in read_jsonl(args.manifest) if str(row.get("split", args.split)) == args.split]
    if args.max_videos > 0:
        manifest = manifest[:args.max_videos]
    (run_dir / "resolved_native_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "archive_context.txt").write_text(archive_context + "\n", encoding="utf-8")

    successes = failures = physical_requests = 0
    latencies = []
    for row in manifest:
        video_id = str(row.get("video_id", "") or Path(row["video_path"]).stem)
        video_path = Path(row["video_path"]).expanduser().resolve()
        metadata = video_metadata(video_path)
        base_record = {
            "run_name": args.run_name,
            "video_id": video_id,
            "video_path": str(video_path),
            "model": config.get("model", ""),
            "parameters": config.get("parameters", ""),
            "quantization": config.get("quantization", ""),
            **metadata,
        }
        try:
            content, input_metadata = build_content(video_path, row, config, archive_context)
            if args.dry_run:
                append_jsonl(run_dir / "predictions.jsonl", {**base_record, **input_metadata, "parse_ok": False, "dry_run": True, "model_output": {}})
                successes += 1
                continue
            raw_content, usage, request_count, latency_ms = call_model(content, config)
            physical_requests += request_count
            latencies.append(latency_ms)
            parsed = parse_model_json(raw_content)
            parse_ok = isinstance(parsed.get("vessels"), list)
            append_jsonl(run_dir / "raw_responses.jsonl", {**base_record, "content": raw_content, "usage": usage, "request_count": request_count, "latency_ms": latency_ms})
            append_jsonl(run_dir / "predictions.jsonl", {
                **base_record,
                **input_metadata,
                "parse_ok": parse_ok,
                "model_output": parsed,
                "request_count": request_count,
                "latency_ms": latency_ms,
                "usage": usage,
            })
            successes += 1
        except Exception as exc:
            append_jsonl(run_dir / "errors.jsonl", {**base_record, "error_type": type(exc).__name__, "error": str(exc)})
            append_jsonl(run_dir / "predictions.jsonl", {**base_record, "parse_ok": False, "request_count": 0, "model_output": {}, "error": str(exc)})
            failures += 1

    summary = {
        "run_name": args.run_name,
        "model": config.get("model", ""),
        "parameters": config.get("parameters", ""),
        "quantization": config.get("quantization", ""),
        "input_mode": config.get("input_mode", "video_url"),
        "videos": len(manifest),
        "successful_requests": successes,
        "failed_requests": failures,
        "physical_model_requests": physical_requests,
        "avg_latency_ms_per_video": sum(latencies) / len(latencies) if latencies else 0.0,
        "dry_run": args.dry_run,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
