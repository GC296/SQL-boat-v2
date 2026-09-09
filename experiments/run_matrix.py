"""Generate and optionally execute the paper ablation matrix."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IDENTITY_MATRIX = [
    ("single_frame", "none", "recognize_once"),
    ("fixed_majority", "majority", "fixed_interval"),
    ("tel", "tel", "uncertainty"),
    ("tel_archive", "tel_archive", "uncertainty_archive"),
    ("full_memory", "full", "full"),
]
POLICY_MATRIX = [
    ("recognize_once", "full", "recognize_once"),
    ("fixed_interval", "full", "fixed_interval"),
    ("uncertainty", "full", "uncertainty"),
    ("archive_policy", "full", "uncertainty_archive"),
    ("experience_policy", "full", "experience"),
    ("full_policy", "full", "full"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--group", choices=["identity", "policy", "network", "all"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    rows = []
    if args.group in {"identity", "all"}:
        rows.extend((f"identity_{name}", memory, policy, "lan") for name, memory, policy in IDENTITY_MATRIX)
    if args.group in {"policy", "all"}:
        rows.extend((f"policy_{name}", memory, policy, "lan") for name, memory, policy in POLICY_MATRIX)
    if args.group in {"network", "all"}:
        rows.extend((f"network_{profile}", "full", "full", profile) for profile in ("lan", "stable_mobile", "limited", "interruption"))
    for run_name, memory, policy, network in rows:
        command = [sys.executable, "-m", "experiments.run_experiment", "--manifest", args.manifest, "--config", args.config, "--split", args.split, "--run-name", run_name, "--memory-mode", memory, "--policy-mode", policy, "--network-profile", network, "--max-frames", str(args.max_frames)]
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, check=True, cwd=Path.cwd())


if __name__ == "__main__":
    main()
