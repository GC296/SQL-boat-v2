"""Shared structured identity schema for VLM observations and archive records."""
from __future__ import annotations

from typing import Any, Mapping

IDENTITY_FIELDS: tuple[str, ...] = (
    "vessel_type",
    "size_class",
    "hull_color",
    "superstructure_color",
    "superstructure_shape",
    "window_pattern",
    "bow_features",
    "stern_features",
    "deck_equipment",
    "propulsion_features",
    "distinctive_markings",
    "visible_text",
)

IDENTITY_FIELD_LABELS: dict[str, str] = {
    "vessel_type": "船型",
    "size_class": "尺寸级别",
    "hull_color": "船体颜色",
    "superstructure_color": "上层建筑颜色",
    "superstructure_shape": "上层建筑形态",
    "window_pattern": "窗户布局",
    "bow_features": "船首特征",
    "stern_features": "船尾特征",
    "deck_equipment": "甲板固定设备",
    "propulsion_features": "推进装置特征",
    "distinctive_markings": "显著标志",
    "visible_text": "可见文字",
}

_EMPTY_VALUES = {"", "未知", "不确定", "不可见", "未观察到", "none", "unknown", "n/a", "na"}


def clean_identity_value(value: Any) -> str:
    text = str(value or "").strip().strip('"“”')
    return "" if text.lower().replace(" ", "") in _EMPTY_VALUES else text


def normalize_identity_features(value: Mapping[str, Any] | None) -> dict[str, str]:
    source = value or {}
    return {field: clean_identity_value(source.get(field)) for field in IDENTITY_FIELDS}


def canonical_identity_description(features: Mapping[str, Any] | None, fallback: str = "") -> str:
    normalized = normalize_identity_features(features)
    parts = [f"{IDENTITY_FIELD_LABELS[field]}：{normalized[field]}" for field in IDENTITY_FIELDS if normalized[field]]
    return "；".join(parts) if parts else str(fallback or "").strip()


def structured_ship_record(hull_number: str, features: Mapping[str, Any] | None, description: str = "") -> dict[str, str]:
    normalized = normalize_identity_features(features)
    return {
        "hull_number": str(hull_number or "").strip(),
        **normalized,
        "description": canonical_identity_description(normalized, fallback=description),
    }


def identity_features_from_record(record: Mapping[str, Any]) -> dict[str, str]:
    return normalize_identity_features(record)


def simple_identity_prompt() -> str:
    return """你是船舶身份证据提取助手。只分析图片中的目标船舶，只返回一个 JSON 对象，不要解释、分析过程或 Markdown。

要求：
1. hull_number 只填写图片中可靠可见的真实舷号或船体编号；看不清时填写空字符串。
2. description 只用一句话描述当前图片中可靠可见的船舶结构特征，不超过 50 个汉字。
3. 不要根据船型或档案猜测舷号，不要描述天气、背景、画质或拍摄位置。

返回格式：
{
  "hull_number": "可靠舷号或空字符串",
  "description": "可靠可见的简洁船舶结构描述"
}"""


def structured_identity_prompt() -> str:
    fields = ", ".join(IDENTITY_FIELDS)
    return f"""你是船舶身份证据提取助手。只分析图片中的目标船舶，只返回一个 JSON 对象，不要解释或输出 Markdown。

要求：
1. hull_number 只填写图片中可靠可见的真实舷号或船体编号；船名、品牌名、船尾文字不是舷号。
2. 看不清舷号时返回空字符串，禁止根据船型、档案候选或结构特征猜测舷号。
3. description 只能描述当前图片中可靠可见、稳定且与身份相关的结构特征，长度不超过 80 个汉字。
4. identity_features 必须包含固定字段：{fields}。
5. 每个属性只填写当前图片中可靠可见的内容；不可见、不确定或无法判断时填写空字符串。
6. 不描述天气、海面、背景、拍摄距离、画质、目标位置、航行状态或人员。

返回格式：
{{
  "hull_number": "可靠舷号或空字符串",
  "description": "可靠可见的简洁身份结构描述",
  "identity_features": {{
    "vessel_type": "",
    "size_class": "",
    "hull_color": "",
    "superstructure_color": "",
    "superstructure_shape": "",
    "window_pattern": "",
    "bow_features": "",
    "stern_features": "",
    "deck_equipment": "",
    "propulsion_features": "",
    "distinctive_markings": "",
    "visible_text": ""
  }}
}}"""


def vlm_identity_prompt(schema: str = "structured") -> str:
    if str(schema or "structured").strip().lower() in {"simple", "legacy", "description"}:
        return simple_identity_prompt()
    return structured_identity_prompt()
