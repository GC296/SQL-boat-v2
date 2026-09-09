"""船弦号数据库 — 可插拔数据源 + FAISS 向量库 + 自动变更检测"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError:
    FAISS = None

    class Embeddings:
        pass

    class Document:
        def __init__(self, page_content: str, metadata: dict[str, Any]):
            self.page_content = page_content
            self.metadata = metadata

from config import load_config
from identity_schema import normalize_identity_features, structured_ship_record

logger = logging.getLogger(__name__)

HASH_FILE_NAME = ".db_hash"


class DashScopeEmbeddings(Embeddings):
    """DashScope Embedding 封装，直接调用 OpenAI 兼容模式 API。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        if not api_key or api_key.startswith("your-"):
            raise ValueError(
                "Embedding API Key 未配置。请在 config.yaml 中设置 embed.api_key，"
                "或在 .env 中设置 EMBED_API_KEY。"
            )
        self.model = model
        self.api_key = api_key
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import httpx
        import time

        max_retries = 3
        batch_size = 10
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    payload = {"model": self.model, "input": batch}
                    resp = httpx.post(
                        self._url,
                        headers=self._headers,
                        json=payload,
                        timeout=60,
                    )
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                        logger.warning("Embedding API 限流，%ds 后重试 (%d/%d)", retry_after, attempt + 1, max_retries)
                        time.sleep(retry_after)
                        continue
                    if resp.status_code >= 500:
                        logger.warning("Embedding API 服务错误 [%d]，%ds 后重试 (%d/%d)", resp.status_code, 2 ** attempt, attempt + 1, max_retries)
                        time.sleep(2 ** attempt)
                        continue
                    if not resp.is_success:
                        try:
                            err_body = resp.json()
                            err_msg = err_body.get("error", {}).get("message", resp.text[:300])
                        except Exception:
                            err_msg = resp.text[:300]
                        raise RuntimeError(
                            f"Embedding API 返回 {resp.status_code}: {err_msg}\n"
                            f"请检查 config.yaml 中 embed 配置（model / api_key / base_url）。"
                        )
                    data = resp.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning("Embedding API 网络错误: %s，%ds 后重试 (%d/%d)", e, wait, attempt + 1, max_retries)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Embedding API 调用失败，已重试 {max_retries} 次") from last_error

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _create_source(config: dict[str, Any], db_path: str | None = None):
    """根据配置创建数据源（CSV 或 SQLite）。"""
    from .csv_source import CsvShipSource
    from .sql_source import SqlShipSource

    db_cfg = config.get("database", {})
    backend = str(db_cfg.get("backend", "sqlite")).lower()
    if backend == "sqlite":
        sql_path = db_path or db_cfg.get("sqlite_path", "./data/ships.db")
        return SqlShipSource(sql_path)
    if backend == "csv":
        csv_path = db_path or db_cfg.get("csv_path") or config.get("app", {}).get("ship_db_path", "./data/ships.csv")
        return CsvShipSource(csv_path)
    raise ValueError(f"Unsupported database backend: {backend}")


class ShipDatabase:
    """
    船弦号数据库 — 双通道检索：
      1. 精确查找（dict，O(1)）
      2. FAISS 向量语义检索（RAG）

    数据源：CSV 或 SQLite（可插拔）
    自动变更检测：通过 MD5 哈希比对，数据变更时自动重建向量库。
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        db_path: str | None = None,
    ):
        if config is None:
            config = load_config()

        self._config = config

        embed_cfg = config.get("embed", {})
        retrieval_cfg = config.get("retrieval", {})
        vs_cfg = config.get("vector_store", {})

        # ── 数据源 ──
        self._source = _create_source(config, db_path)
        self._data = self._source.load_all()
        db_cfg = config.get("database", {})
        backend = str(db_cfg.get("backend", "sqlite")).lower()
        if backend == "csv":
            from .sql_source import SqlShipSource
            self._memory_source = SqlShipSource(db_cfg.get("memory_sqlite_path", "./data/ship_memory.db"))
        else:
            self._memory_source = self._source

        # ── Embedding 客户端 ──
        self._embeddings = DashScopeEmbeddings(
            model=embed_cfg.get("model", "Qwen3-Embedding-0.6B"),
            api_key=embed_cfg.get("api_key", ""),
            base_url=embed_cfg.get("base_url", "http://localhost:7891/v1"),
        )

        # ── 检索参数 ──
        self._top_k = retrieval_cfg.get("top_k", 3)
        self._score_threshold = retrieval_cfg.get("score_threshold", 0.5)

        # ── 向量库配置 ──
        self._persist_path = vs_cfg.get("persist_path", "./vector_store")
        self._auto_rebuild = vs_cfg.get("auto_rebuild", False)

        # ── 向量库（懒加载） ──
        self._vector_store: FAISS | None = None
        self._prototype_records: list[dict[str, str]] = []
        self._prototype_embeddings: dict[str, list[float]] = {}
        self._prototype_embedding_hash: str = ""

    # ── 变更检测 ────────────────────────────────

    def _load_prototype_records(self) -> list[dict[str, str]]:
        loader = getattr(self._source, "load_prototypes", None)
        if loader:
            records = loader()
        else:
            records = [
                {
                    "prototype_id": f"{hull_number}_01",
                    "hull_number": hull_number,
                    "description": description,
                }
                for hull_number, description in sorted(self._source.load_all().items())
            ]
        return [
            {
                "prototype_id": str(record.get("prototype_id", "") or "").strip(),
                "hull_number": str(record.get("hull_number", "") or "").strip(),
                "description": str(record.get("description", "") or "").strip(),
            }
            for record in records
            if str(record.get("hull_number", "") or "").strip()
            and str(record.get("description", "") or "").strip()
        ]

    def _compute_data_hash(self) -> str:
        """计算全部档案原型的 MD5 哈希。"""
        content = "\n".join(
            f"{record['prototype_id']}|{record['hull_number']}|{record['description']}"
            for record in self._load_prototype_records()
        )
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_saved_hash(self) -> str | None:
        hash_file = Path(self._persist_path) / HASH_FILE_NAME
        if hash_file.exists():
            return hash_file.read_text(encoding="utf-8").strip()
        return None

    def _save_hash(self, data_hash: str) -> None:
        persist_dir = Path(self._persist_path)
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / HASH_FILE_NAME).write_text(data_hash, encoding="utf-8")

    def _data_changed(self) -> bool:
        current_hash = self._compute_data_hash()
        saved_hash = self._load_saved_hash()
        changed = current_hash != saved_hash
        if changed:
            logger.info("数据变更检测: 数据已修改，将重建向量库")
        return changed

    # ── 向量库构建 ─────────────────────────────

    def _build_documents(self) -> list[Document]:
        """从当前数据构建 LangChain Document 列表。"""
        docs = []
        for hn, desc in self._data.items():
            content = desc
            docs.append(Document(
                page_content=content,
                metadata={"hull_number": hn, "description": desc},
            ))
        return docs

    def _load_or_build_vector_store(self) -> FAISS | None:
        """加载缓存的向量库，或从数据重新构建。Embedding API 不可用时返回 None。"""
        persist_dir = Path(self._persist_path)
        index_file = persist_dir / "index.faiss"

        data_changed = self._data_changed()

        if not self._auto_rebuild and not data_changed and index_file.exists():
            try:
                logger.info("从 %s 加载向量库缓存…", persist_dir)
                vs = FAISS.load_local(
                    str(persist_dir),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("向量库缓存加载成功")
                return vs
            except Exception as e:
                logger.warning("缓存加载失败（%s），将重新构建", e)

        # 数据变化时，先重新加载
        if data_changed:
            logger.info("重新加载数据…")
            self._data = self._source.load_all()

        docs = self._build_documents()
        if not docs:
            logger.warning("无数据可构建向量库，语义检索将不可用")
            return None

        try:
            logger.info("正在构建 FAISS 向量库（%d 条文档）…", len(docs))
            vs = FAISS.from_documents(docs, self._embeddings)

            persist_dir.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(persist_dir))

            self._save_hash(self._compute_data_hash())
            logger.info("向量库已持久化到 %s，哈希已更新", persist_dir)
            return vs
        except Exception as e:
            logger.warning("向量库构建失败（%s），语义检索将不可用", e)
            return None

    @property
    def vector_store(self) -> FAISS | None:
        """懒加载向量库（首次访问时自动构建或加载缓存）。"""
        if self._vector_store is None or self._data_changed():
            self._vector_store = self._load_or_build_vector_store()
        return self._vector_store

    # ── 精确查找 ──────────────────────────────

    def lookup(self, hull_number: str) -> str | None:
        """精确查找：通过弦号直接查字典，O(1)。"""
        return self._data.get(hull_number.strip())

    # ── 语义检索 ──────────────────────────────

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return 0.0
        return dot / (left_norm * right_norm)

    def _ensure_prototype_embeddings(self) -> None:
        current_hash = self._compute_data_hash()
        if current_hash == self._prototype_embedding_hash:
            return
        self._data = self._source.load_all()
        records = self._load_prototype_records()
        self._prototype_records = records
        self._prototype_embeddings = {}
        if not records:
            self._prototype_embedding_hash = current_hash
            return
        vectors = self._embeddings.embed_documents([record["description"] for record in records])
        self._prototype_embeddings = {
            record["prototype_id"]: vector
            for record, vector in zip(records, vectors)
        }
        self._prototype_embedding_hash = current_hash

    def semantic_search_prototypes(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Compare one observation with every prototype, then keep each vessel's best prototype."""
        query = str(query or "").strip()
        if not query:
            return []
        self._ensure_prototype_embeddings()
        if not self._prototype_embeddings:
            return []
        query_vector = self._embeddings.embed_query(query)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self._prototype_records:
            prototype_id = record["prototype_id"]
            vector = self._prototype_embeddings.get(prototype_id)
            if vector is None:
                continue
            cosine = self._cosine_similarity(query_vector, vector)
            score = max(0.0, min(1.0, cosine))
            grouped.setdefault(record["hull_number"], []).append({
                "prototype_id": prototype_id,
                "description": record["description"],
                "score": score,
            })

        results: list[dict[str, Any]] = []
        for hull_number, prototype_matches in grouped.items():
            prototype_matches.sort(key=lambda item: (-float(item["score"]), str(item["prototype_id"])))
            best = prototype_matches[0]
            results.append({
                "hull_number": hull_number,
                "description": best["description"],
                "score": round(float(best["score"]), 4),
                "cosine_similarity": round(float(best["score"]), 4),
                "prototype_id": best["prototype_id"],
                "prototype_score": round(float(best["score"]), 4),
                "prototype_matches": [
                    {
                        "prototype_id": item["prototype_id"],
                        "score": round(float(item["score"]), 4),
                    }
                    for item in prototype_matches
                ],
                "similarity_metric": "multi_prototype_cosine",
            })
        results.sort(key=lambda item: (-float(item["score"]), str(item["hull_number"])))
        return results[: max(1, int(top_k or self._top_k))]

    def semantic_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Compatibility alias for multi-prototype cosine retrieval."""
        return self.semantic_search_prototypes(query, top_k=top_k)

    def semantic_search_filtered(self, query: str) -> list[dict]:
        """语义检索 + 分数阈值过滤。"""
        results = self.semantic_search(query, top_k=self._top_k)
        return [r for r in results if r["score"] >= self._score_threshold]

    # ── 数据管理 ──────────────────────────────

    def list_prototypes(self) -> list[dict[str, str]]:
        return self._load_prototype_records()

    def get_prototype(self, prototype_id: str) -> dict[str, str] | None:
        getter = getattr(self._source, "get_prototype", None)
        return getter(prototype_id) if getter else None

    def add_prototype(self, hull_number: str, description: str, prototype_id: str = "") -> dict[str, str] | None:
        writer = getattr(self._source, "add_prototype", None)
        if writer is None:
            return None
        record = writer(hull_number, description, prototype_id)
        if record:
            self._invalidate_cache()
        return record

    def update_prototype(self, prototype_id: str, description: str) -> bool:
        writer = getattr(self._source, "update_prototype", None)
        result = bool(writer and writer(prototype_id, description))
        if result:
            self._invalidate_cache()
        return result

    def delete_prototype(self, prototype_id: str) -> bool:
        writer = getattr(self._source, "delete_prototype", None)
        result = bool(writer and writer(prototype_id))
        if result:
            self._invalidate_cache()
        return result

    def add_ship(self, hull_number: str, description: str) -> bool:
        result = self._source.add(hull_number, description)
        if result:
            self._invalidate_cache()
        return result

    def get_ship_record(self, hull_number: str) -> dict[str, str] | None:
        getter = getattr(self._source, "get_record", None)
        if getter:
            return getter(hull_number)
        description = self.lookup(hull_number)
        return structured_ship_record(hull_number, {}, description or "") if description is not None else None

    def add_ship_record(self, hull_number: str, identity_features: Mapping[str, Any] | None, description: str = "") -> bool:
        writer = getattr(self._source, "add_record", None)
        canonical = structured_ship_record(hull_number, normalize_identity_features(identity_features), description)
        result = writer(hull_number, identity_features, description) if writer else self._source.add(hull_number, canonical["description"])
        if result:
            self._invalidate_cache()
        return result

    def update_ship(self, hull_number: str, description: str) -> bool:
        result = self._source.update(hull_number, description)
        if result:
            self._invalidate_cache()
        return result

    def update_ship_record(self, hull_number: str, identity_features: Mapping[str, Any] | None, description: str = "") -> bool:
        writer = getattr(self._source, "update_record", None)
        canonical = structured_ship_record(hull_number, normalize_identity_features(identity_features), description)
        result = writer(hull_number, identity_features, description) if writer else self._source.update(hull_number, canonical["description"])
        if result:
            self._invalidate_cache()
        return result

    def delete_ship(self, hull_number: str) -> bool:
        result = self._source.delete(hull_number)
        if result:
            self._invalidate_cache()
        return result

    def upsert_ship(self, hull_number: str, description: str) -> str:
        result = self._source.upsert(hull_number, description)
        self._invalidate_cache()
        return result

    def upsert_ship_record(self, hull_number: str, identity_features: Mapping[str, Any] | None, description: str = "") -> str:
        writer = getattr(self._source, "upsert_record", None)
        canonical = structured_ship_record(hull_number, normalize_identity_features(identity_features), description)
        result = writer(hull_number, identity_features, description) if writer else self._source.upsert(hull_number, canonical["description"])
        self._invalidate_cache()
        return result

    def reload(self) -> None:
        """重新加载数据。"""
        self._data = self._source.load_all()
        self._vector_store = None  # 强制下次访问时重建
        self._prototype_records = []
        self._prototype_embeddings = {}
        self._prototype_embedding_hash = ""

    def _invalidate_cache(self) -> None:
        """数据变更后使缓存失效。"""
        self._data = self._source.load_all()
        self._vector_store = None
        self._prototype_records = []
        self._prototype_embeddings = {}
        self._prototype_embedding_hash = ""

    # ── 属性 ──────────────────────────────────

    @property
    def source(self):
        return self._source

    @property
    def memory_source(self):
        return self._memory_source

    @property
    def hull_numbers(self) -> list[str]:
        return list(self._data.keys())

    @property
    def descriptions(self) -> list[str]:
        return list(self._data.values())

    @property
    def items(self) -> Mapping[str, str]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)


    def get_profile_memory(self, hull_number: str) -> dict[str, Any] | None:
        getter = getattr(self._memory_source, "get_profile_memory", None)
        return getter(hull_number) if getter else None

    def upsert_profile_memory(self, hull_number: str, **kwargs: Any) -> None:
        writer = getattr(self._memory_source, "upsert_profile_memory", None)
        if writer:
            writer(hull_number, **kwargs)

    def add_recognition_experience(self, **kwargs: Any) -> None:
        writer = getattr(self._memory_source, "add_recognition_experience", None)
        if writer:
            writer(**kwargs)

    def list_profile_memory(self, limit: int = 1000) -> list[dict[str, Any]]:
        reader = getattr(self._memory_source, "list_profile_memory", None)
        return reader(limit=limit) if reader else []

    def retrieve_recognition_experiences(self, **kwargs: Any) -> list[dict[str, Any]]:
        reader = getattr(self._memory_source, "retrieve_recognition_experiences", None)
        return reader(**kwargs) if reader else []

    def experience_action_hint(self, **kwargs: Any) -> dict[str, Any] | None:
        reader = getattr(self._memory_source, "experience_action_hint", None)
        return reader(**kwargs) if reader else None
