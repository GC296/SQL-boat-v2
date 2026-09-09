"""Ship catalogue service for multiple description prototypes per vessel."""
from __future__ import annotations

import base64
from typing import Any

from config import load_config
from database import ShipDatabase


class ShipService:
    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or load_config()
        self._db: ShipDatabase | None = None

    @property
    def db(self) -> ShipDatabase:
        if self._db is None:
            self._db = ShipDatabase(config=self._config)
        return self._db

    def list_ships(self) -> list[dict[str, str]]:
        return self.db.list_prototypes()

    def get_ship(self, hull_number: str) -> dict[str, str] | None:
        return next((item for item in self.list_ships() if item["hull_number"] == hull_number), None)

    def get_prototype(self, prototype_id: str) -> dict[str, str] | None:
        return self.db.get_prototype(prototype_id)

    def create_ship(self, hull_number: str, description: str) -> dict[str, str] | None:
        return self.db.add_prototype(hull_number, description)

    def update_prototype(self, prototype_id: str, description: str) -> bool:
        return self.db.update_prototype(prototype_id, description)

    def delete_prototype(self, prototype_id: str) -> bool:
        return self.db.delete_prototype(prototype_id)

    def delete_ship(self, hull_number: str) -> bool:
        return self.db.delete_ship(hull_number)

    def bulk_create(self, ships: dict[str, str]) -> dict[str, int]:
        added = 0
        for hull_number, description in ships.items():
            if self.create_ship(hull_number, description):
                added += 1
        return {"added": added, "skipped": len(ships) - added}

    def search(self, keyword: str) -> list[dict[str, str]]:
        query = keyword.lower().strip()
        return [
            record for record in self.list_ships()
            if query in record["prototype_id"].lower()
            or query in record["hull_number"].lower()
            or query in record["description"].lower()
        ]

    def stats(self) -> dict[str, Any]:
        source = self.db.source
        backend_type = "sqlite" if hasattr(source, "db_path") else "csv"
        prototypes = self.list_ships()
        return {
            "total_ships": len({item["hull_number"] for item in prototypes}),
            "total_prototypes": len(prototypes),
            "backend": backend_type,
        }

    def recognize_ship(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        from tools import _vlm_infer

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        result = _vlm_infer(image_b64, prompt_mode="detailed")
        hull_number = str(result.get("hull_number", "") or "").strip()
        description = str(result.get("description", "") or "").strip()
        existing_count = sum(1 for item in self.list_ships() if item["hull_number"] == hull_number) if hull_number else 0
        return {
            "hull_number": hull_number,
            "description": description,
            "already_exists": existing_count > 0,
            "existing_prototype_count": existing_count,
        }

    def recognize_and_add(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        result = self.recognize_ship(image_bytes, filename)
        if not result["hull_number"]:
            return {**result, "error": "未识别到可靠舷号，请人工填写后确认"}
        if not result["description"]:
            return {**result, "error": "未提取到可靠结构描述，请人工补充后确认"}
        record = self.create_ship(result["hull_number"], result["description"])
        return {**result, "prototype": record, "action": "added"}
