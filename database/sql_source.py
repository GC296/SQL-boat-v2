"""SQLite ship catalogue, profile memory, and recognition experience storage."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from .base import ShipDataSource


class SqlShipSource(ShipDataSource):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._path = Path(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ships (
                    hull_number TEXT PRIMARY KEY,
                    description TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ship_embeddings (
                    hull_number TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ship_profile_memory (
                    hull_number TEXT PRIMARY KEY,
                    db_match_id TEXT NOT NULL DEFAULT '',
                    visual_summary TEXT NOT NULL DEFAULT '',
                    common_attributes TEXT NOT NULL DEFAULT '[]',
                    observed_count INTEGER NOT NULL DEFAULT 0,
                    successful_match_count INTEGER NOT NULL DEFAULT 0,
                    failed_match_count INTEGER NOT NULL DEFAULT 0,
                    common_misreads TEXT NOT NULL DEFAULT '[]',
                    last_seen_frame INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS recognition_experience (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    scene_type TEXT NOT NULL DEFAULT '',
                    uncertainty REAL NOT NULL DEFAULT 1.0,
                    hull_number_pred TEXT NOT NULL DEFAULT '',
                    db_match_id TEXT NOT NULL DEFAULT '',
                    match_type TEXT NOT NULL DEFAULT 'none',
                    final_correct INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    action_taken TEXT NOT NULL DEFAULT '',
                    action_success INTEGER NOT NULL DEFAULT 0,
                    semantic_candidates TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experience_scene ON recognition_experience(scene_type);
                CREATE INDEX IF NOT EXISTS idx_experience_action ON recognition_experience(action_taken);
            """)

    def load_all(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT hull_number, description FROM ships").fetchall()
        return {row["hull_number"]: row["description"] for row in rows}

    def lookup(self, hull_number: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT description FROM ships WHERE hull_number = ?", (hull_number,)).fetchone()
        return row["description"] if row else None

    def add(self, hull_number: str, description: str) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("INSERT INTO ships (hull_number, description) VALUES (?, ?)", (hull_number, description))
            return True
        except sqlite3.IntegrityError:
            return False

    def update(self, hull_number: str, description: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("UPDATE ships SET description = ? WHERE hull_number = ?", (description, hull_number))
        return cursor.rowcount > 0

    def delete(self, hull_number: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM ships WHERE hull_number = ?", (hull_number,))
            conn.execute("DELETE FROM ship_embeddings WHERE hull_number = ?", (hull_number,))
        return cursor.rowcount > 0

    def upsert(self, hull_number: str, description: str) -> str:
        existed = self.lookup(hull_number) is not None
        with self._connect() as conn:
            conn.execute("INSERT INTO ships (hull_number, description) VALUES (?, ?) ON CONFLICT(hull_number) DO UPDATE SET description=excluded.description", (hull_number, description))
        return "updated" if existed else "added"

    def bulk_add(self, ships: dict[str, str]) -> int:
        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany("INSERT OR IGNORE INTO ships (hull_number, description) VALUES (?, ?)", ships.items())
            return conn.total_changes - before

    def search_by_description(self, keyword: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT hull_number, description FROM ships WHERE description LIKE ?", (f"%{keyword}%",)).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM ships").fetchone()[0])

    def load_all_embeddings(self) -> dict[str, list[float]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT hull_number, embedding FROM ship_embeddings").fetchall()
        return {row["hull_number"]: json.loads(row["embedding"]) for row in rows}

    def store_embeddings_bulk(self, records: dict[str, list[float]]) -> int:
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO ship_embeddings (hull_number, embedding) VALUES (?, ?)", ((key, json.dumps(value)) for key, value in records.items()))
        return len(records)

    def delete_embedding(self, hull_number: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ship_embeddings WHERE hull_number = ?", (hull_number,))

    @staticmethod
    def _decode_json_fields(row: sqlite3.Row | None, fields: tuple[str, ...]) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for field in fields:
            try:
                item[field] = json.loads(item.get(field) or "[]")
            except json.JSONDecodeError:
                item[field] = []
        return item

    def get_profile_memory(self, hull_number: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ship_profile_memory WHERE hull_number = ?", (hull_number,)).fetchone()
        return self._decode_json_fields(row, ("common_attributes", "common_misreads"))

    def list_profile_memory(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM ship_profile_memory ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [self._decode_json_fields(row, ("common_attributes", "common_misreads")) or {} for row in rows]

    def upsert_profile_memory(self, hull_number: str, *, db_match_id: str = "", visual_summary: str = "", attributes: list[str] | None = None, matched: bool = False, misread: str = "", frame_id: int = 0) -> None:
        existing = self.get_profile_memory(hull_number) or {}
        merged_attributes = list(dict.fromkeys([*(existing.get("common_attributes") or []), *(attributes or [])]))[-30:]
        merged_misreads = list(existing.get("common_misreads") or [])
        if misread and misread != hull_number and misread not in merged_misreads:
            merged_misreads.append(misread)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO ship_profile_memory (
                    hull_number, db_match_id, visual_summary, common_attributes, observed_count,
                    successful_match_count, failed_match_count, common_misreads, last_seen_frame, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(hull_number) DO UPDATE SET
                    db_match_id=CASE WHEN excluded.db_match_id != '' THEN excluded.db_match_id ELSE ship_profile_memory.db_match_id END,
                    visual_summary=CASE WHEN excluded.visual_summary != '' THEN excluded.visual_summary ELSE ship_profile_memory.visual_summary END,
                    common_attributes=excluded.common_attributes,
                    observed_count=ship_profile_memory.observed_count + 1,
                    successful_match_count=ship_profile_memory.successful_match_count + excluded.successful_match_count,
                    failed_match_count=ship_profile_memory.failed_match_count + excluded.failed_match_count,
                    common_misreads=excluded.common_misreads,
                    last_seen_frame=excluded.last_seen_frame,
                    updated_at=excluded.updated_at
            """, (hull_number, db_match_id, visual_summary, json.dumps(merged_attributes, ensure_ascii=False), int(matched), int(not matched), json.dumps(merged_misreads[-30:], ensure_ascii=False), int(frame_id), time.time()))

    def add_recognition_experience(self, *, track_id: int, frame_id: int, scene_type: str, uncertainty: float, hull_number_pred: str, db_match_id: str, match_type: str, final_correct: bool, failure_reason: str, action_taken: str, action_success: bool, semantic_candidates: list[str] | None = None, notes: str = "") -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO recognition_experience (
                    track_id, frame_id, scene_type, uncertainty, hull_number_pred, db_match_id,
                    match_type, final_correct, failure_reason, action_taken, action_success,
                    semantic_candidates, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (int(track_id), int(frame_id), scene_type, float(uncertainty), hull_number_pred, db_match_id, match_type, int(final_correct), failure_reason, action_taken, int(action_success), json.dumps(semantic_candidates or [], ensure_ascii=False), notes, time.time()))

    def retrieve_recognition_experiences(self, *, scene_type: str = "", uncertainty: float = 1.0, failure_reason: str = "", top_k: int = 5, min_similarity: float = 0.0) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM recognition_experience ORDER BY created_at DESC LIMIT 1000").fetchall()
        ranked: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            scene_score = 1.0 if not scene_type or item["scene_type"] == scene_type else 0.35
            uncertainty_score = max(0.0, 1.0 - abs(float(item["uncertainty"]) - float(uncertainty)))
            failure_score = 1.0 if not failure_reason or item["failure_reason"] == failure_reason else 0.4
            similarity = 0.45 * scene_score + 0.35 * uncertainty_score + 0.20 * failure_score
            if similarity < min_similarity:
                continue
            item["semantic_candidates"] = json.loads(item.get("semantic_candidates") or "[]")
            item["similarity"] = round(similarity, 4)
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["similarity"], -int(item["action_success"]), -item["created_at"]))
        return ranked[:max(0, int(top_k))]

    def experience_action_hint(self, **kwargs: Any) -> dict[str, Any] | None:
        experiences = self.retrieve_recognition_experiences(**kwargs)
        if not experiences:
            return None
        grouped: dict[str, list[int]] = {}
        for item in experiences:
            grouped.setdefault(item["action_taken"], []).append(int(item["action_success"]))
        action, outcomes = max(grouped.items(), key=lambda pair: (sum(pair[1]) / len(pair[1]), len(pair[1]), pair[0]))
        return {"action": action, "success_rate": round(sum(outcomes) / len(outcomes), 4), "support": len(outcomes), "experiences": experiences}
