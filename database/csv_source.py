"""CSV ship catalogue with multiple visual-description prototypes per vessel."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from identity_schema import identity_features_from_record, structured_ship_record
from .base import ShipDataSource

logger = logging.getLogger(__name__)
CSV_FIELDS = ("prototype_id", "hull_number", "description")


class CsvShipSource(ShipDataSource):
    def __init__(self, csv_path: str):
        self._path = Path(csv_path)
        self._data: dict[str, str] = {}
        self._records: dict[str, dict[str, str]] = {}

    def _next_prototype_id(self, hull_number: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", str(hull_number or "ship").strip()) or "ship"
        pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")
        numbers = [int(match.group(1)) for prototype_id in self._records if (match := pattern.match(prototype_id))]
        return f"{base}_{max(numbers, default=0) + 1:02d}"

    def _unique_prototype_id(self, requested: str, hull_number: str) -> str:
        prototype_id = str(requested or "").strip() or self._next_prototype_id(hull_number)
        if prototype_id not in self._records:
            return prototype_id
        return self._next_prototype_id(hull_number)

    def _rebuild_identity_index(self) -> None:
        grouped: dict[str, list[str]] = {}
        for record in self._records.values():
            grouped.setdefault(record["hull_number"], []).append(record["description"])
        self._data = {
            hull_number: "；".join(dict.fromkeys(description for description in descriptions if description))
            for hull_number, descriptions in grouped.items()
        }

    def load_all(self) -> dict[str, str]:
        self._records.clear()
        if not self._path.exists():
            self._data = {}
            return self._data
        with self._path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "hull_number" in reader.fieldnames:
                for row in reader:
                    hull_number = str(row.get("hull_number") or "").strip()
                    if not hull_number:
                        continue
                    description = str(row.get("description") or "").strip()
                    if not description:
                        legacy = structured_ship_record(hull_number, identity_features_from_record(row), "")
                        description = legacy["description"]
                    if not description:
                        continue
                    prototype_id = self._unique_prototype_id(str(row.get("prototype_id") or ""), hull_number)
                    self._records[prototype_id] = {
                        "prototype_id": prototype_id,
                        "hull_number": hull_number,
                        "description": description,
                    }
            else:
                handle.seek(0)
                for row in csv.reader(handle):
                    if len(row) < 2:
                        continue
                    hull_number = row[0].strip()
                    description = row[1].strip()
                    if not hull_number or hull_number == "hull_number" or not description:
                        continue
                    prototype_id = self._unique_prototype_id("", hull_number)
                    self._records[prototype_id] = {
                        "prototype_id": prototype_id,
                        "hull_number": hull_number,
                        "description": description,
                    }
        self._rebuild_identity_index()
        logger.info("Loaded %d prototypes for %d vessels from CSV: %s", len(self._records), len(self._data), self._path)
        return dict(self._data)

    def load_records(self) -> list[dict[str, str]]:
        if not self._records and self._path.exists():
            self.load_all()
        return [dict(self._records[key]) for key in sorted(self._records)]

    def load_prototypes(self) -> list[dict[str, str]]:
        self.load_all()
        return self.load_records()

    def get_record(self, hull_number: str) -> dict[str, str] | None:
        records = [record for record in self.load_records() if record["hull_number"] == hull_number]
        return records[0] if records else None

    def get_prototype(self, prototype_id: str) -> dict[str, str] | None:
        record = self._records.get(prototype_id)
        return dict(record) if record else None

    def lookup(self, hull_number: str) -> str | None:
        return self._data.get(hull_number)

    def add_prototype(self, hull_number: str, description: str, prototype_id: str = "") -> dict[str, str] | None:
        hull_number = str(hull_number or "").strip()
        description = str(description or "").strip()
        if not hull_number or not description:
            return None
        resolved_id = self._unique_prototype_id(prototype_id, hull_number)
        record = {"prototype_id": resolved_id, "hull_number": hull_number, "description": description}
        self._records[resolved_id] = record
        self._rebuild_identity_index()
        self._save()
        return dict(record)

    def update_prototype(self, prototype_id: str, description: str) -> bool:
        record = self._records.get(prototype_id)
        description = str(description or "").strip()
        if record is None or not description:
            return False
        record["description"] = description
        self._rebuild_identity_index()
        self._save()
        return True

    def delete_prototype(self, prototype_id: str) -> bool:
        if prototype_id not in self._records:
            return False
        self._records.pop(prototype_id)
        self._rebuild_identity_index()
        self._save()
        return True

    def add(self, hull_number: str, description: str) -> bool:
        if hull_number in self._data:
            return False
        return self.add_prototype(hull_number, description) is not None

    def add_record(self, hull_number: str, features: Mapping[str, Any] | None, description: str = "") -> bool:
        canonical = structured_ship_record(hull_number, features, description)["description"]
        return self.add(hull_number, canonical)

    def update(self, hull_number: str, description: str) -> bool:
        record = self.get_record(hull_number)
        return self.update_prototype(record["prototype_id"], description) if record else False

    def update_record(self, hull_number: str, features: Mapping[str, Any] | None, description: str = "") -> bool:
        canonical = structured_ship_record(hull_number, features, description)["description"]
        return self.update(hull_number, canonical)

    def delete(self, hull_number: str) -> bool:
        prototype_ids = [prototype_id for prototype_id, record in self._records.items() if record["hull_number"] == hull_number]
        if not prototype_ids:
            return False
        for prototype_id in prototype_ids:
            self._records.pop(prototype_id, None)
        self._rebuild_identity_index()
        self._save()
        return True

    def upsert(self, hull_number: str, description: str) -> str:
        if hull_number in self._data:
            self.update(hull_number, description)
            return "updated"
        self.add(hull_number, description)
        return "added"

    def upsert_record(self, hull_number: str, features: Mapping[str, Any] | None, description: str = "") -> str:
        canonical = structured_ship_record(hull_number, features, description)["description"]
        return self.upsert(hull_number, canonical)

    def bulk_add(self, ships: dict[str, str]) -> int:
        added = 0
        for hull_number, description in ships.items():
            if hull_number not in self._data and self.add_prototype(hull_number, description) is not None:
                added += 1
        return added

    def count(self) -> int:
        return len(self._data)

    def prototype_count(self) -> int:
        return len(self._records)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for prototype_id in sorted(self._records):
                writer.writerow(self._records[prototype_id])
