"""Validate corrected track annotations before evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_KNOWN = {"known", "unknown"}
VALID_RISK = {"low", "medium", "high"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    errors = []
    count = 0
    with args.annotations.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            item = json.loads(line)
            prefix = f"line {line_number}, video={item.get('video_id')}, track={item.get('track_id')}"
            if item.get("known_or_unknown") not in VALID_KNOWN:
                errors.append(f"{prefix}: known_or_unknown must be known or unknown")
            if item.get("risk_label") not in VALID_RISK:
                errors.append(f"{prefix}: risk_label must be low, medium, or high")
            if item.get("known_or_unknown") == "known" and not item.get("hull_number"):
                errors.append(f"{prefix}: known target requires hull_number")
            if int(item.get("frame_end", -1)) < int(item.get("frame_start", 0)):
                errors.append(f"{prefix}: frame_end is earlier than frame_start")
    if errors:
        print("\n".join(errors))
        raise SystemExit(f"annotation validation failed with {len(errors)} error(s)")
    print(f"annotation validation passed: {count} tracks")


if __name__ == "__main__":
    main()
