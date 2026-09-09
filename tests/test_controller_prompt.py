import json
from pathlib import Path


def test_controller_prompt_json_examples_are_not_double_braced():
    prompt_path = Path(__file__).parents[1] / "pipeline" / "prompts" / "central_controller.txt"
    prompt = prompt_path.read_text(encoding="utf-8")

    assert '{{"kind"' not in prompt
    assert "{{tools}}" in prompt
    assert "{{context}}" in prompt

    examples = [
        line.strip()
        for line in prompt.splitlines()
        if line.strip().startswith("{") and not line.strip().startswith("{{")
    ]
    assert examples
    for example in examples:
        json.loads(example)
