from pipeline.archive import decide_identity

def test_unknown_rejection():
    decision = decide_identity({"A": 0.2, "B": 0.19}, unknown_threshold=0.7)
    assert decision.state == "unknown" and decision.identity == ""

def test_confident_archive_identity():
    decision = decide_identity({"A": 9.0, "B": 1.0}, exact_threshold=0.85)
    assert decision.state == "confirmed" and decision.identity == "A"
