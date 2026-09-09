import sys
from types import SimpleNamespace

import pytest

import tools
from tools import normalize_vlm_identity_result


class FakeTimeout:
    def __init__(self, **kwargs):
        self.settings = kwargs


class FakeTimeoutException(Exception):
    pass


class FakeReadTimeout(FakeTimeoutException):
    pass


class FakeNetworkError(Exception):
    pass


class FakeHTTPStatusError(Exception):
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


def install_fake_httpx(monkeypatch, post):
    module = SimpleNamespace(
        post=post,
        Timeout=FakeTimeout,
        TimeoutException=FakeTimeoutException,
        ReadTimeout=FakeReadTimeout,
        NetworkError=FakeNetworkError,
        HTTPStatusError=FakeHTTPStatusError,
    )
    monkeypatch.setitem(sys.modules, "httpx", module)
    return module


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def vlm_config(**overrides):
    config = {
        "model": "test-vlm",
        "api_key": "YOUR_API_KEY",
        "base_url": "http://localhost:7890/v1",
        "temperature": 0.0,
        "request_timeout_seconds": 90,
        "max_tokens": 256,
        "repetition_penalty": 1.05,
        "max_retries": 1,
    }
    config.update(overrides)
    return config


def test_normalize_vlm_identity_result_prefers_simple_description():
    result = normalize_vlm_identity_result({
        "hull_number": " 012 ",
        "description": "白色大型游船，侧面连续弧形玻璃窗",
    })
    assert result == {
        "hull_number": "012",
        "identity_features": {},
        "description": "白色大型游船，侧面连续弧形玻璃窗",
    }


def test_normalize_vlm_identity_result_keeps_legacy_field_compatibility():
    result = normalize_vlm_identity_result({
        "hull_number": "",
        "identity_features": {
            "vessel_type": "小型机动船",
            "propulsion_features": "黑色舷外机",
        },
    })
    assert result["identity_features"]["vessel_type"] == "小型机动船"
    assert result["identity_features"]["propulsion_features"] == "黑色舷外机"
    assert result["description"] == "船型：小型机动船；推进装置特征：黑色舷外机"


def test_ship_name_is_not_accepted_as_hull_number():
    result = normalize_vlm_identity_result({
        "hull_number": "无极",
        "description": "船体侧面具有无极字样",
    })
    assert result["hull_number"] == ""


def test_parse_vlm_json_repairs_terminal_smart_quote_only():
    content = (
        '{"hull_number":"","description":"yellow hull with '
        '\u201cICE-900\u201d marking.\u201d\n}'
    )

    result = tools._parse_json_content(content)

    assert result["description"] == "yellow hull with \u201cICE-900\u201d marking."


def test_vlm_retries_once_after_read_timeout(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise FakeReadTimeout("timed out")
        return FakeResponse('{"hull_number":"","description":"白色船体配备大面积深色玻璃窗"}')

    monkeypatch.setattr(tools, "_get_llm_cfg", lambda: vlm_config())
    fake_httpx = install_fake_httpx(monkeypatch, fake_post)

    result = tools._vlm_infer("image-data")

    assert result["description"] == "白色船体配备大面积深色玻璃窗"
    assert result["_vlm_http_attempts"] == 2
    assert len(calls) == 2
    assert calls[0]["json"]["max_tokens"] == 256
    assert calls[0]["json"]["repetition_penalty"] == 1.05
    assert isinstance(calls[0]["timeout"], FakeTimeout)


def test_vlm_retries_once_after_truncated_json(monkeypatch):
    responses = iter([
        FakeResponse('{"hull_number":"","description":"船尾有固定文字标识，船尾有固定文字标识'),
        FakeResponse('{"hull_number":"","description":"白色船体，船尾具有固定文字标识"}'),
    ])
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(tools, "_get_llm_cfg", lambda: vlm_config())
    fake_httpx = install_fake_httpx(monkeypatch, fake_post)

    result = tools._vlm_infer("image-data")

    assert result["description"] == "白色船体，船尾具有固定文字标识"
    assert result["_vlm_http_attempts"] == 2
    assert len(calls) == 2


def test_vlm_falls_back_from_structured_to_simple_schema(monkeypatch):
    responses = iter([
        FakeResponse("1. 分析图像内容：这是一艘白色工作船"),
        FakeResponse('{"hull_number":"","description":"白色小型工作船"}'),
    ])
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr(tools, "_get_llm_cfg", lambda: vlm_config(
        output_schema="structured",
        fallback_output_schema="simple",
        fallback_output_schema_enabled=True,
        enable_thinking=False,
        structured_max_tokens=768,
        simple_max_tokens=256,
        max_retries=1,
    ))
    install_fake_httpx(monkeypatch, fake_post)

    result = tools._vlm_infer("image-data")

    assert result["description"] == "白色小型工作船"
    assert result["_vlm_schema"] == "simple"
    assert result["_vlm_fallback_used"] is True
    assert calls[0]["max_tokens"] == 768
    assert calls[1]["max_tokens"] == 256
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "identity_features" in calls[0]["messages"][0]["content"][0]["text"]
    assert "identity_features" not in calls[1]["messages"][0]["content"][0]["text"]


def test_vlm_stops_after_configured_retry_limit(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        raise FakeReadTimeout("timed out")

    monkeypatch.setattr(tools, "_get_llm_cfg", lambda: vlm_config(max_retries=1))
    fake_httpx = install_fake_httpx(monkeypatch, fake_post)

    with pytest.raises(RuntimeError, match="failed after 2 attempt"):
        tools._vlm_infer("image-data")
    assert len(calls) == 2


def test_vlm_falls_back_to_base_model(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        if kwargs["json"]["model"] == "vessel-lora":
            response = SimpleNamespace(status_code=404)
            raise FakeHTTPStatusError("adapter unavailable", response=response)
        return FakeResponse('{"hull_number":"012","description":"white vessel"}')

    monkeypatch.setattr(tools, "_get_llm_cfg", lambda: vlm_config(
        use_lora=True,
        lora_model="vessel-lora",
        fallback_model="base-awq",
        fallback_to_base=True,
    ))
    install_fake_httpx(monkeypatch, fake_post)

    result = tools._vlm_infer("image-data")

    assert calls == ["vessel-lora", "base-awq"]
    assert result["_vlm_model"] == "base-awq"
    assert result["_vlm_fallback_used"] is True
    assert result["_vlm_http_attempts"] == 2
