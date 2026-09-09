from pipeline.tracker import TrackManager

def test_completed_track_state_survives_stale_cleanup():
    manager = TrackManager(max_stale_frames=5)
    manager.get_or_create(7, 1)
    manager.apply_identity_decision(7, "", "out_of_archive", 0.42, evidence_source="visual_open_set")

    assert manager.cleanup_stale(10) == 1
    assert manager.get(7) is None
    assert manager.get_any(7).identity_state == "out_of_archive"
    assert {item["track_id"] for item in manager.export_memory()} == {7}


def add_observation(manager, frame, quality=0.8):
    manager.record_observation(1, frame, (10, 10, 110, 80), 0.9, quality, {"scale": 0.5}, (480, 640, 3))

def test_tel_fuses_quality_weighted_candidates():
    manager = TrackManager(memory_mode="tel", ledger_decay=1.0, ledger_min_support=0.1)
    add_observation(manager, 1, 0.2); manager.bind_result(1, "A12", "", 1)
    add_observation(manager, 2, 0.9); manager.bind_result(1, "B34", "", 2)
    add_observation(manager, 3, 0.9); manager.bind_result(1, "B34", "", 3)
    info = manager.get(1)
    assert info.hull_number == "B34"
    assert info.hull_candidate_scores["B34"] > info.hull_candidate_scores["A12"]
    assert info.identity_state == "probable"

def test_single_mode_keeps_latest_result():
    manager = TrackManager(memory_mode="none")
    add_observation(manager, 1); manager.bind_result(1, "A", "", 1)
    add_observation(manager, 2); manager.bind_result(1, "B", "", 2)
    assert manager.get(1).hull_number == "B"


def test_track_defaults_to_visible_modality():
    manager = TrackManager()
    info = manager.get_or_create(3, 1)
    assert info.modality == "visible"


def test_tel_fuses_structure_similarity_scores():
    manager = TrackManager(memory_mode="tel", ledger_decay=1.0)
    add_observation(manager, 1, 0.8)
    manager.bind_result(1, "", "white vessel", 1, semantic_match_ids=["A12"], semantic_matches=[{"hull_number": "A12", "score": 0.80}])
    add_observation(manager, 2, 0.9)
    manager.bind_result(1, "", "curved black windows", 2, semantic_match_ids=["A12"], semantic_matches=[{"hull_number": "A12", "score": 0.90}])
    info = manager.get(1)
    assert info.structure_evidence_count == 2
    assert info.structure_consistent_observations == 2
    assert info.structure_conflict_observations == 0
    assert info.structure_min_consistent_score == 0.80
    assert 0.80 < info.structure_candidate_scores["A12"] < 0.90


def test_tel_records_conflicting_structure_candidates():
    manager = TrackManager(memory_mode="tel", ledger_decay=1.0)
    add_observation(manager, 1)
    manager.bind_result(1, "", "first", 1, semantic_matches=[{"hull_number": "A12", "score": 0.82}, {"hull_number": "B34", "score": 0.60}])
    add_observation(manager, 2)
    manager.bind_result(1, "", "second", 2, semantic_matches=[{"hull_number": "B34", "score": 0.84}, {"hull_number": "A12", "score": 0.61}])
    info = manager.get(1)
    assert info.structure_evidence_count == 2
    assert info.structure_consistent_observations == 1
    assert info.structure_conflict_observations == 1


def test_verified_identity_survives_one_weak_follow_up_decision():
    manager = TrackManager(memory_mode="tel")
    add_observation(manager, 1)
    manager.bind_result(1, "", "known vessel", 1, semantic_matches=[{"hull_number": "A12", "score": 0.92}])
    manager.bind_db_match(1, "A12", "known vessel", evidence_source="structure_semantic", confidence=0.92)
    applied = manager.apply_identity_decision(1, "A12", "uncertain", 0.72)
    info = manager.get(1)
    assert applied is False
    assert info.verified_identity == "A12"
    assert info.identity_state == "structure_verified"


def test_structure_identity_never_overwrites_read_hull_number():
    manager = TrackManager(memory_mode="tel")
    add_observation(manager, 1)
    manager.bind_result(1, "0013", "white vessel", 1, semantic_match_ids=["002"], semantic_matches=[{"hull_number": "002", "score": 0.85}])
    manager.apply_identity_decision(1, "002", "structure_verified", 0.85)
    info = manager.get(1)
    assert info.hull_number == "0013"
    assert info.archive_candidate_id == "002"
    assert info.verified_identity == ""


def test_review_request_is_idempotent_and_inherited_by_reassociated_tracklet():
    manager = TrackManager(memory_mode="full")
    manager.record_observation(1, 10, (10, 10, 100, 80), 0.9, 0.8, frame_shape=(480, 640, 3), entity_id="E1", member_track_ids=[1])
    manager.apply_identity_decision(1, "012", "conflicting", 0.55)

    assert manager.request_review(1, 20, "request_remote_verification", ["active_query_budget_exhausted"])
    assert not manager.request_review(1, 21, "request_remote_verification", ["duplicate"])
    info = manager.get(1)
    assert info.identity_state == "review_requested"
    assert info.identity_state_before_review == "conflicting"
    assert info.episode_status == "review_requested"
    assert info.review_requested_frame == 20

    inherited = manager.inherit_entity_state(2, 1, "E1", [1, 2], 30)
    assert inherited.identity_state == "review_requested"
    assert inherited.identity_state_before_review == "conflicting"
    assert inherited.review_action == "request_remote_verification"


def test_review_request_does_not_overwrite_verified_identity():
    manager = TrackManager(memory_mode="full")
    add_observation(manager, 1)
    manager.bind_result(1, "", "known vessel", 1, semantic_matches=[{"hull_number": "012", "score": 0.92}])
    manager.bind_db_match(1, "012", "known vessel", evidence_source="structure_visual_agreement", confidence=0.92)

    assert not manager.request_review(1, 20, "request_remote_verification", ["learned_action_escalate_review"])
    info = manager.get(1)
    assert info.identity_state == "structure_verified"
    assert info.verified_identity == "012"
    assert info.episode_status == "active"
    assert any(event["type"] == "review_request_blocked" for event in info.event_history)


def test_visual_match_tracks_repeated_candidate_and_low_scores():
    manager = TrackManager(memory_mode="full")
    add_observation(manager, 1)

    manager.update_visual_match(1, "V081", 0.94, 0.18, low_score_threshold=0.50)
    info = manager.get(1)
    assert info.visual_observation_count == 1
    assert info.visual_consistent_observations == 1
    assert info.visual_low_score_observations == 0

    manager.update_visual_match(1, "V081", 0.93, 0.17, low_score_threshold=0.50)
    assert info.visual_observation_count == 2
    assert info.visual_consistent_observations == 2

    manager.update_visual_match(1, "V142", 0.42, 0.03, low_score_threshold=0.50)
    assert info.visual_consistent_observations == 1
    assert info.visual_low_score_observations == 1

    manager.update_visual_match(1, "V199", 0.39, 0.02, low_score_threshold=0.50)
    assert info.visual_consistent_observations == 1
    assert info.visual_low_score_observations == 2


def test_tracker_finalizes_below_score_gate_as_out_of_archive():
    manager = TrackManager()
    manager.get_or_create(1, 10)
    manager.set_identity_decision_mode(1, "visual_only")
    manager.update_visual_match(1, "V081", 0.79, 0.90, low_score_threshold=0.80)

    assert manager.finalize_visual_score_gate(0.80) == 1
    finalized = manager.get_any(1)
    assert finalized is not None
    assert finalized.identity_state == "out_of_archive"
    assert finalized.identity_evidence_source == "visual_score_gate_finalization"
