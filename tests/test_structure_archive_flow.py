from pipeline.pipeline import ShipPipeline


class FakeDatabase:
    def lookup(self, hull_number):
        return "known structure" if hull_number == "A12" else None

    def semantic_search_prototypes(self, description):
        return [{
            "hull_number": "A12",
            "description": "known structure",
            "score": 0.82,
            "prototype_id": "A12_02",
            "similarity_metric": "multi_prototype_cosine",
        }]

    def semantic_search(self, description):
        return self.semantic_search_prototypes(description)


def make_pipeline():
    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._db = FakeDatabase()
    pipeline._agent_trace = []
    pipeline._trace_lock = __import__("threading").Lock()
    return pipeline


def test_structure_scores_survive_local_retrieval():
    result = make_pipeline()._local_lookup_retrieve("", "white hull curved windows")
    assert result.match_type == "semantic"
    assert result.semantic_match_ids == ["A12"]
    assert result.semantic_matches[0]["score"] == 0.82
    assert result.semantic_matches[0]["prototype_id"] == "A12_02"


def test_exact_hull_precedes_structure_retrieval():
    result = make_pipeline()._local_lookup_retrieve("A12", "")
    assert result.match_type == "exact"
    assert result.description == "known structure"
    assert result.semantic_matches[0]["hull_number"] == "A12"


def test_description_uses_multi_prototype_search_even_if_legacy_fields_exist():
    class PrototypeDatabase(FakeDatabase):
        def semantic_search_prototypes(self, description):
            assert description == "白色船体和弧形玻璃窗"
            return [{
                "hull_number": "A12",
                "description": "known structure",
                "score": 0.91,
                "prototype_id": "A12_03",
                "similarity_metric": "multi_prototype_cosine",
            }]

    pipeline = make_pipeline()
    pipeline._db = PrototypeDatabase()
    result = pipeline._local_lookup_retrieve(
        "",
        "白色船体和弧形玻璃窗",
        identity_features={"hull_color": "白色"},
    )
    assert result.semantic_matches[0]["score"] == 0.91
    assert result.semantic_matches[0]["similarity_metric"] == "multi_prototype_cosine"


def test_visual_only_chain_keeps_text_but_skips_archive_lookup(monkeypatch):
    import numpy as np
    import tools

    pipeline = make_pipeline()
    pipeline._identity_decision_mode = "visual_only"
    pipeline._prompt_mode = "detailed"
    pipeline._max_trace_entries = 100
    pipeline._encode_image = lambda crop: "encoded"
    monkeypatch.setattr(tools, "_vlm_infer", lambda image, prompt_mode: {
        "hull_number": "012",
        "description": "white vessel with dark windows",
        "identity_features": {"hull_color": "white"},
    })
    pipeline._local_lookup_retrieve = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("lookup must be skipped"))

    result = pipeline._run_three_step_chain(np.zeros((10, 10, 3), dtype=np.uint8), track_id=1, frame_id=15)

    assert result.hull_number == "012"
    assert result.description == "white vessel with dark windows"
    assert result.identity_features == {"hull_color": "white"}
    assert result.match_type == "none"
    assert result.semantic_matches == []
