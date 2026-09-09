from identity_schema import canonical_identity_description, normalize_identity_features, vlm_identity_prompt


def test_canonical_identity_description_uses_fixed_field_order():
    description = canonical_identity_description({
        "visible_text": "无极",
        "hull_color": "白色",
        "vessel_type": "游船",
    })
    assert description == "船型：游船；船体颜色：白色；可见文字：无极"


def test_identity_normalization_removes_unknown_placeholders():
    features = normalize_identity_features({
        "vessel_type": "未知",
        "hull_color": " 白色 ",
        "stern_features": "不可见",
    })
    assert features["vessel_type"] == ""
    assert features["hull_color"] == "白色"
    assert features["stern_features"] == ""


def test_legacy_description_is_used_only_without_structured_fields():
    assert canonical_identity_description({}, fallback="旧描述") == "旧描述"
    assert canonical_identity_description({"vessel_type": "作业船"}, fallback="旧描述") == "船型：作业船"


def test_simple_prompt_keeps_the_original_short_contract():
    prompt = vlm_identity_prompt("simple")

    assert '"hull_number"' in prompt
    assert '"description"' in prompt
    assert "identity_features" not in prompt
