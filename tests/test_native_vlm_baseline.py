import csv
import json
import sys
from types import SimpleNamespace

import experiments.run_native_vlm as native_runner
from experiments.run_native_vlm import build_task_prompt, call_model, load_archive_context, parse_model_json


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_archive_prompt_contains_all_identities_without_retrieval(tmp_path):
    archive = tmp_path / "ships.csv"
    with archive.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prototype_id", "hull_number", "description"])
        writer.writeheader()
        writer.writerows([
            {"prototype_id": "012_01", "hull_number": "012", "description": "white vessel with dark windows"},
            {"prototype_id": "012_02", "hull_number": "012", "description": "streamlined passenger vessel"},
            {"prototype_id": "003_01", "hull_number": "003", "description": "yellow workboat with twin engines"},
        ])

    context = load_archive_context(archive)
    prompt = build_task_prompt(context)

    assert "注册身份：012" in prompt
    assert "注册身份：003" in prompt
    assert "档案条目 012_01" in prompt
    assert "档案条目 012_02" in prompt
    assert "找出视频中出现的所有不同物理船舶" in prompt
    assert "out_of_archive" in prompt
    assert "bbox_norm" not in prompt
    assert "完整注册船舶文字档案库" in prompt
    assert prompt.startswith("/no_think\n")
    assert "第一个字符必须是 {" in prompt


def test_native_request_can_disable_thinking_and_require_json(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"vessels":[],"task_complete":true}'}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers, json):
            captured["payload"] = json
            return FakeResponse()

    class FakeTimeout:
        def __init__(self, **kwargs):
            pass

    fake_httpx = SimpleNamespace(
        Client=FakeClient,
        Timeout=FakeTimeout,
        HTTPError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    call_model([], {
        "enable_thinking": False,
        "json_response_format": True,
        "max_tokens": 4096,
    })

    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_native_response_parser_preserves_model_decision():
    parsed = parse_model_json('```json\n{"vessels":[{"decision":"in-archive","archive_identity":"012"}],"task_complete":true}\n```')

    assert parsed["task_complete"]
    assert parsed["vessels"][0]["decision"] == "in-archive"
    assert parsed["vessels"][0]["archive_identity"] == "012"


def test_native_runner_dry_run_builds_artifacts_without_pipeline(tmp_path, monkeypatch):
    archive = tmp_path / "ships.csv"
    archive.write_text("prototype_id,hull_number,description\n012_01,012,white passenger vessel\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{"video_id": "V1", "video_path": str(tmp_path / "V1.mp4"), "split": "test"}])
    config = tmp_path / "native.yaml"
    config.write_text(
        "llm:\n  model: test-model\n  base_url: http://localhost:1/v1\n"
        "native_vlm:\n  input_mode: video_url\n  use_file_url: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(native_runner, "video_metadata", lambda path: {
        "fps": 25.0, "frame_count": 250, "duration_s": 10.0, "width": 1920, "height": 1080,
    })
    monkeypatch.setattr(sys, "argv", [
        "run_native_vlm",
        "--manifest", str(manifest),
        "--archive", str(archive),
        "--config", str(config),
        "--run-name", "dry",
        "--output-root", str(tmp_path / "outputs"),
        "--dry-run",
    ])

    native_runner.main()

    run_dir = tmp_path / "outputs" / "dry"
    prediction = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["video_id"] == "V1"
    assert prediction["dry_run"] is True
    assert prediction["model_output"] == {}
    assert (run_dir / "archive_context.txt").exists()
    assert json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))["videos"] == 1
