from argparse import Namespace

from experiments.run_paper_suite import MAIN_RUNS, RunSpec, command_for


def test_paper_suite_passes_entity_reconciliation_flag():
    args = Namespace(manifest="manifest.jsonl", config="config.yaml", split="test", max_frames=0, display=False, save_output_video=False)
    command = command_for(RunSpec("entity_off", "full", "full", entity="off"), args)
    index = command.index("--entity-reconciliation")
    assert command[index + 1] == "off"


def test_main_fixed_uses_same_tel_memory_as_active_policy():
    fixed = next(spec for spec in MAIN_RUNS if spec.name == "main_fixed")
    active = next(spec for spec in MAIN_RUNS if spec.name == "main_active_tel")
    assert fixed.memory == active.memory == "tel"
    assert fixed.policy == "fixed_interval"
    assert active.policy == "uncertainty"


def test_paper_suite_passes_visual_identity_flags():
    args = Namespace(manifest="manifest.jsonl", config="config.yaml", split="test", max_frames=0, display=False, save_output_video=False)
    command = command_for(RunSpec("full", "full", "full"), args)
    assert command[command.index("--visual-archive") + 1] == "on"
    assert command[command.index("--visual-decision") + 1] == "on"
    assert command[command.index("--visual-open-set") + 1] == "on"
