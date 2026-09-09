"""Aggregate metrics from multiple experiment runs into CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiment_outputs", type=Path)
    parser.add_argument("--output", default="experiment_outputs/summary.csv", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/metrics.json")):
        rows.append({"run_name": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        raise SystemExit("No metrics.json files found")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
