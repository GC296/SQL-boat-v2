"""
AgentResult — 识别结果数据类（无 LangChain 依赖）

三步链路产出的结构化结果：
  Step1: VLM 识别 → hull_number + description
  Step2: db.lookup(hull_number) → 精确匹配
  Step3: db.semantic_search_filtered(description) → 语义检索
"""

from __future__ import annotations


class AgentResult:
    """三步链路运行结果。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        identity_features: dict[str, str] | None = None,
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        semantic_matches: list[dict] | None = None,
        visual_match_ids: list[str] | None = None,
        visual_matches: list[dict] | None = None,
        visual_candidate_id: str = "",
        visual_similarity_score: float = 0.0,
        visual_margin: float = 0.0,
        visual_embedding_latency_ms: float = 0.0,
        visual_matching_latency_ms: float = 0.0,
        visual_backend: str = "",
        visual_model_request_count: int = 0,
        visual_pairs_scored: int = 0,
        visual_gate_reason: str = "",
        answer: str = "",
        vlm_http_attempts: int = 1,
        vlm_model: str = "",
        vlm_base_url: str = "",
        vlm_fallback_used: bool = False,
    ):
        self.hull_number = hull_number
        self.description = description
        self.identity_features = identity_features or {}
        self.match_type = match_type  # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.semantic_matches = semantic_matches or []
        self.visual_match_ids = visual_match_ids or []
        self.visual_matches = visual_matches or []
        self.visual_candidate_id = visual_candidate_id
        self.visual_similarity_score = float(visual_similarity_score)
        self.visual_margin = float(visual_margin)
        self.visual_embedding_latency_ms = float(visual_embedding_latency_ms)
        self.visual_matching_latency_ms = float(visual_matching_latency_ms)
        self.visual_backend = str(visual_backend or "")
        self.visual_model_request_count = max(0, int(visual_model_request_count))
        self.visual_pairs_scored = max(0, int(visual_pairs_scored))
        self.visual_gate_reason = visual_gate_reason
        self.answer = answer
        self.vlm_http_attempts = max(1, int(vlm_http_attempts))
        self.vlm_model = str(vlm_model or "")
        self.vlm_base_url = str(vlm_base_url or "")
        self.vlm_fallback_used = bool(vlm_fallback_used)
