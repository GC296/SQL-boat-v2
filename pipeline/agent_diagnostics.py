"""Bounded local diagnostics independent of legacy experiment logging."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import threading
import logging


class AgentDiagnostics:
    def __init__(self, config):
        self.enabled = bool(config.get("diagnostics_enabled", False))
        self.root = Path(config.get("diagnostics_dir", "output/agent_diagnostics"))
        self.max_bytes = max(65536, int(config.get("diagnostics_max_bytes", 10_000_000)))
        self.image_limit = max(1, int(config.get("diagnostics_image_limit", 256)))
        self.lock = threading.RLock()

    def image(self, encoded):
        try:
            return self._image(encoded)
        except OSError as exc:
            logging.getLogger(__name__).warning("Agent image diagnostic unavailable: %s", exc)
            return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "persisted": False}

    def _image(self, encoded):
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if not self.enabled:
            return {"sha256": digest, "persisted": False}
        with self.lock:
            folder = self.root / "images"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{digest}.jpg"
            if not path.exists():
                try:
                    path.write_bytes(base64.b64decode(encoded, validate=True))
                except (ValueError, OSError):
                    return {"sha256": digest, "persisted": False}
                files = sorted(folder.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
                for old in files[:-self.image_limit]:
                    old.unlink(missing_ok=True)
            return {"sha256": digest, "persisted": True, "image_ref": str(path.resolve())}

    def write(self, event):
        try:
            self._write(event)
        except OSError as exc:
            logging.getLogger(__name__).warning("Agent diagnostic unavailable: %s", exc)

    def _write(self, event):
        if not self.enabled:
            return
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / "events.jsonl"
            if path.exists() and path.stat().st_size >= self.max_bytes:
                path.replace(self.root / "events.previous.jsonl")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
