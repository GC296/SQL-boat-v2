"""Shared VLM inference for reliable hull reading and vessel structure description."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config import load_config
from identity_schema import canonical_identity_description, normalize_identity_features, vlm_identity_prompt

logger = logging.getLogger(__name__)
_cached_llm_cfg: dict | None = None


def _get_llm_cfg() -> dict:
    global _cached_llm_cfg
    if _cached_llm_cfg is None:
        _cached_llm_cfg = load_config().get("llm", {})
    return _cached_llm_cfg


def _clean_hull_number(value: Any, description: str = "") -> str:
    hull_number = str(value or "").strip().strip('"“”')
    placeholder = hull_number.lower().replace(" ", "")
    if placeholder in {"无", "未知", "没有", "none", "unknown", "n/a", "na", "看不清", "不可见", "未识别"}:
        return ""
    if hull_number and not re.search(r"[A-Za-z0-9]", hull_number):
        numbered = re.search(r"(?:舷号|编号)[^A-Za-z0-9]{0,8}([A-Za-z]*[0-9][A-Za-z0-9-]*)", description, re.IGNORECASE)
        return numbered.group(1) if numbered else ""
    return hull_number


def _parse_json_content(content: str) -> dict[str, Any]:
    fence = chr(96) * 3
    content = content.strip()
    if content.startswith(fence):
        content = content.split("\n", 1)[-1]
        if content.endswith(fence):
            content = content[:-3]
        content = content.strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"{.*}", content, re.DOTALL)
        if not match:
            logger.warning("Unable to find complete VLM JSON: %s", content[:300])
            return {}
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            repaired = re.sub(r"[\u201c\u201d\uff02](?=\s*[,}])", '"', match.group())
            if repaired == match.group():
                logger.warning("Unable to parse VLM JSON: %s", content[:300])
                return {}
            try:
                result = json.loads(repaired)
            except json.JSONDecodeError:
                logger.warning("Unable to parse VLM JSON: %s", content[:300])
                return {}
    return result if isinstance(result, dict) else {}


def normalize_vlm_identity_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_features = result.get("identity_features")
    if not isinstance(raw_features, dict):
        raw_features = {}
    normalized_features = normalize_identity_features(raw_features)
    description = str(result.get("description") or "").strip()
    if not description:
        description = canonical_identity_description(normalized_features)
    identity_features = normalized_features if any(normalized_features.values()) else {}
    return {
        "hull_number": _clean_hull_number(result.get("hull_number"), description),
        "identity_features": identity_features,
        "description": description,
    }


def _vlm_prompt(schema: str | None = None) -> str:
    llm_cfg = _get_llm_cfg()
    selected_schema = schema or str(llm_cfg.get("output_schema", "simple"))
    return vlm_identity_prompt(selected_schema)


def _vlm_schemas(llm_cfg: dict[str, Any]) -> list[str]:
    primary = str(llm_cfg.get("output_schema", "simple") or "simple").strip().lower()
    schemas = [primary]
    default_fallback = "simple" if primary == "structured" else "structured"
    fallback = str(llm_cfg.get("fallback_output_schema", default_fallback) or default_fallback).strip().lower()
    if fallback == primary:
        fallback = default_fallback
    if bool(llm_cfg.get("fallback_output_schema_enabled", False)) and fallback not in schemas:
        schemas.append(fallback)
    return schemas


def _message_content(message: Any) -> str:
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text") is not None
        )
    return str(content or "")


def _vlm_models(llm_cfg: dict[str, Any]) -> list[str]:
    configured_model = str(llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ") or "").strip()
    use_lora = bool(llm_cfg.get("use_lora", False))
    lora_model = str(llm_cfg.get("lora_model", "") or "").strip()
    active_model = lora_model if use_lora and lora_model else configured_model
    models = [active_model]
    fallback_enabled = bool(llm_cfg.get("fallback_to_base", True))
    fallback_model = str(llm_cfg.get("fallback_model", configured_model) or "").strip()
    if fallback_enabled and fallback_model and fallback_model not in models:
        models.append(fallback_model)
    return [model for model in models if model]


def _vlm_infer(image_b64: str, prompt_mode: str = "detailed") -> dict[str, Any]:
    """Call the VLM with structured output and a legacy-format recovery path."""
    import httpx

    llm_cfg = _get_llm_cfg()
    headers = {
        "Authorization": f"Bearer {llm_cfg.get('api_key', 'abc123')}",
        "Content-Type": "application/json",
    }
    repetition_penalty = max(1.0, min(2.0, float(llm_cfg.get("repetition_penalty", 1.05))))
    max_retries = max(0, min(2, int(llm_cfg.get("max_retries", 1))))
    timeout_seconds = max(10.0, float(llm_cfg.get("request_timeout_seconds", 90)))
    timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=30.0, pool=10.0)
    last_error: Exception | None = None
    attempts = max_retries + 1
    total_attempts = 0
    models = _vlm_models(llm_cfg)
    schemas = _vlm_schemas(llm_cfg)
    if str(prompt_mode or "").strip().lower() in {"simple", "legacy", "description"}:
        schemas = ["simple"]
    for model_index, model in enumerate(models):
        if model_index == 0 and bool(llm_cfg.get("use_lora", False)):
            model_base_url = str(llm_cfg.get("lora_base_url", llm_cfg.get("base_url", "http://localhost:7890/v1")))
        else:
            model_base_url = str(llm_cfg.get("fallback_base_url", llm_cfg.get("base_url", "http://localhost:7890/v1")))
        api_url = f"{model_base_url.rstrip('/')}/chat/completions"
        for schema_index, schema in enumerate(schemas):
            configured_tokens = llm_cfg.get(
                "simple_max_tokens" if schema == "simple" else "structured_max_tokens",
                llm_cfg.get("max_tokens", 256),
            )
            max_tokens = max(64, min(1024, int(configured_tokens)))
            payload = {
                "model": model,
                "temperature": llm_cfg.get("temperature", 0.0),
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "chat_template_kwargs": {
                    "enable_thinking": bool(llm_cfg.get("enable_thinking", False)),
                },
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": _vlm_prompt(schema)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]}],
            }
            for attempt in range(1, attempts + 1):
                total_attempts += 1
                non_retryable = False
                try:
                    response = httpx.post(api_url, headers=headers, json=payload, timeout=timeout)
                    if response.status_code == 429 or response.status_code >= 500:
                        response.raise_for_status()
                    response.raise_for_status()
                    try:
                        message = response.json()["choices"][0]["message"]
                        content = _message_content(message)
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        last_error = RuntimeError(f"Unexpected VLM response structure: {exc}")
                    else:
                        normalized = normalize_vlm_identity_result(_parse_json_content(content))
                        if normalized["hull_number"] or normalized["description"]:
                            normalized["_vlm_http_attempts"] = total_attempts
                            normalized["_vlm_model"] = model
                            normalized["_vlm_base_url"] = model_base_url
                            normalized["_vlm_fallback_used"] = model_index > 0 or schema_index > 0
                            normalized["_vlm_schema"] = schema
                            return normalized
                        last_error = RuntimeError("VLM returned incomplete or empty JSON")
                        non_retryable = schema_index + 1 < len(schemas)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = exc
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    status_code = exc.response.status_code if exc.response is not None else 0
                    non_retryable = status_code not in {429} and status_code < 500

                if non_retryable:
                    break
                if attempt < attempts:
                    logger.warning(
                        "VLM model %s schema %s attempt %d/%d failed (%s); retrying",
                        model,
                        schema,
                        attempt,
                        attempts,
                        last_error,
                    )
            if schema_index + 1 < len(schemas):
                logger.warning(
                    "VLM model %s schema %s failed; retrying with %s schema",
                    model,
                    schema,
                    schemas[schema_index + 1],
                )
        if model_index + 1 < len(models):
            logger.warning("VLM model %s failed; falling back to %s", model, models[model_index + 1])

    raise RuntimeError(f"VLM inference failed after {total_attempts} attempt(s): {last_error}") from last_error


def _chat_json(prompt: str, image_b64: list[str] | None = None, max_tokens: int = 768, llm_config: dict | None = None, image_manifest: list[dict] | None = None) -> dict[str, Any]:
    """Send a JSON-only request for controller or evidence prompts."""
    import httpx

    llm_cfg = dict(llm_config) if llm_config is not None else _get_llm_cfg()
    models = _vlm_models(llm_cfg)
    attempts = max(1, min(3, int(llm_cfg.get("max_retries", 1)) + 1))
    timeout_seconds = max(10.0, float(llm_cfg.get("request_timeout_seconds", 90)))
    timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=30.0, pool=10.0)
    images = image_b64 or []
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, image in enumerate(images):
        if image_manifest and index < len(image_manifest):
            entry = image_manifest[index]
            content.append({"type": "text", "text": f"Image {index}: {entry.get('role', 'observation')} {entry.get('image_id', '')}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    last_error: Exception | None = None
    total_attempts = 0
    for model_index, model in enumerate(models):
        if model_index == 0 and bool(llm_cfg.get("use_lora", False)):
            base_url = str(llm_cfg.get("lora_base_url", llm_cfg.get("base_url", "http://localhost:7890/v1")))
        else:
            base_url = str(llm_cfg.get("fallback_base_url", llm_cfg.get("base_url", "http://localhost:7890/v1")))
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max(64, min(2048, int(max_tokens))),
            "chat_template_kwargs": {"enable_thinking": bool(llm_cfg.get("enable_thinking", False))},
            "messages": [{"role": "user", "content": content}],
        }
        for attempt in range(attempts):
            total_attempts += 1
            try:
                response = httpx.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {llm_cfg.get('api_key', 'abc123')}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                response_body = response.json()
                message = response_body["choices"][0]["message"]
                result = _parse_json_content(_message_content(message))
                if result:
                    result["_vlm_http_attempts"] = total_attempts
                    result["_vlm_model"] = model
                    result["_vlm_base_url"] = base_url
                    result["_vlm_usage"] = response_body.get("usage")
                    result["_vlm_finish_reason"] = response_body["choices"][0].get("finish_reason")
                    return result
                last_error = RuntimeError("VLM returned an empty JSON object")
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
                logger.warning("JSON VLM request failed for %s attempt %d/%d: %s", model, attempt + 1, attempts, exc)
    raise RuntimeError(f"JSON VLM inference failed after {total_attempts} attempt(s): {last_error}") from last_error


def _controller_infer(context: dict[str, Any], image_b64: list[str] | None = None, llm_config: dict | None = None) -> dict[str, Any]:
    prompt = context.get("prompt", "") if isinstance(context, dict) else ""
    return _chat_json(str(prompt), image_b64=image_b64, max_tokens=int(context.get("max_tokens", 768)), llm_config=llm_config, image_manifest=context.get("snapshot", {}).get("image_manifest"))


def _vlm_verify(crop: Any, question: str = "", fields: list[str] | None = None, llm_config: dict | None = None) -> dict[str, Any]:
    """Extract requested identity evidence from one supplied view."""
    import cv2
    import base64

    success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("verification view encoding failed")
    requested = ", ".join(str(item) for item in (fields or [])) or "reliable identity-related structure"
    template_path = Path(__file__).resolve().parent.parent / "pipeline" / "prompts" / "verify_evidence.txt"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else "Inspect only the supplied view and report visible identity evidence as JSON."
    prompt = template.replace("{{question}}", question or "What reliable identity evidence is visible?").replace("{{fields}}", requested).replace("{{view_id}}", "supplied_view")
    kwargs = {"llm_config": llm_config} if llm_config is not None else {}
    result = _chat_json(prompt, [base64.b64encode(encoded.tobytes()).decode("utf-8")], max_tokens=768, **kwargs)
    normalized = normalize_vlm_identity_result(result)
    normalized["observed"] = dict(result.get("observed", {}) or normalized.get("identity_features", {}))
    normalized["visibility"] = result.get("visibility", "unknown")
    for key in ("observed_hull_number", "readability", "alternatives"):
        if key in result:
            normalized["observed"][key] = result[key]
    for key in ("semantic_matches", "claims", "candidate_assessments", "contradictions", "unresolved_gaps"):
        if key in result:
            normalized[key] = result[key]
    normalized.update({key: value for key, value in result.items() if key.startswith("_vlm_")})
    return normalized
