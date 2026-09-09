"""Run the definitive paper experiment suite with read-only long-term memory."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    name: str
    memory: str
    policy: str
    network: str = "real"
    quality: str = "on"
    ledger: str = "on"
    archive: str = "on"
    experience: str = "off"
    risk: str = "off"
    entity: str = "on"
    visual_archive: str = "on"
    visual_decision: str = "on"
    visual_open_set: str = "on"


MAIN_RUNS = [
    RunSpec("main_single", "none", "recognize_once"),
    RunSpec("main_fixed", "tel", "fixed_interval"),
    RunSpec("main_active_tel", "tel", "uncertainty"),
    RunSpec("main_full", "full", "full"),
]

ABLATION_RUNS = [
    RunSpec("ablation_full", "full", "full"),
    RunSpec("ablation_no_quality", "full", "full", quality="off"),
    RunSpec("ablation_no_tel", "none", "full", ledger="off"),
    RunSpec("ablation_no_archive", "full", "full", archive="off"),
    RunSpec("ablation_fixed_policy", "full", "fixed_interval"),
    RunSpec("ablation_no_entity", "full", "full", entity="off"),
    RunSpec(
        "ablation_text_only", "full", "full",
        visual_archive="off", visual_decision="off", visual_open_set="off",
    ),
    RunSpec("ablation_no_open_set", "full", "full", visual_open_set="off"),
]

NETWORK_RUNS = [
    RunSpec("network_lan", "full", "full", network="lan"),
    RunSpec("network_stable_mobile", "full", "full", network="stable_mobile"),
    RunSpec("network_limited", "full", "full", network="limited"),
    RunSpec("network_interruption", "full", "full", network="interruption"),
]


def command_for(spec: RunSpec, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "experiments.run_experiment",
        "--manifest", args.manifest, "--config", args.config, "--split", args.split,
        "--run-name", spec.name, "--memory-mode", spec.memory, "--policy-mode", spec.policy,
        "--network-profile", spec.network, "--quality", spec.quality, "--ledger", spec.ledger,
        "--archive", spec.archive, "--experience", spec.experience, "--risk", spec.risk,
        "--visual-archive", spec.visual_archive,
        "--visual-decision", spec.visual_decision,
        "--visual-open-set", spec.visual_open_set,
        "--entity-reconciliation", spec.entity,
        "--max-frames", str(args.max_frames),
    ] + (["--display"] if args.display else []) + (["--save-output-video"] if args.save_output_video else [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--group", choices=["main", "ablation", "network", "all"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--display", action="store_true", help="Show rendered video windows during execution")
    parser.add_argument("--save-output-video", action="store_true", help="Save rendered result videos")
    args = parser.parse_args()
    runs = []
    if args.group in {"main", "all"}:
        runs.extend(MAIN_RUNS)
    if args.group in {"ablation", "all"}:
        runs.extend(ABLATION_RUNS)
    if args.group in {"network", "all"}:
        runs.extend(NETWORK_RUNS)
    for spec in runs:
        command = command_for(spec, args)
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, check=True, cwd=Path.cwd())


if __name__ == "__main__":
    main()
