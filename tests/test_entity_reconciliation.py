import numpy as np
from types import SimpleNamespace

from pipeline.entity import EntityReconciler, aggregate_entity_identity
from pipeline.tracker import TrackManager


def identity_track(
    state,
    verified="",
    candidate="",
    score=0.0,
    source="none",
    attempts=1,
    before_review="",
    review_frame=0,
):
    return SimpleNamespace(
        identity_state=state,
        verified_identity=verified,
        archive_candidate_id=candidate,
        archive_similarity_score=score,
        identity_evidence_source=source,
        recognition_attempts=attempts,
        identity_state_before_review=before_review,
        review_requested_frame=review_frame,
        review_action="request_remote_verification",
        review_reasons=[],
    )


def test_entity_identity_uses_out_of_archive_evidence_from_any_member():
    identity = aggregate_entity_identity([
        identity_track("unknown", attempts=0),
        identity_track("out_of_archive", score=0.61, source="visual_open_set"),
    ])

    assert identity["identity_state"] == "out_of_archive"
    assert identity["verified_identity"] == ""
    assert identity["recognition_attempts"] == 1


def test_entity_confirmed_identity_overrides_rejected_fragment():
    identity = aggregate_entity_identity([
        identity_track("out_of_archive", score=0.40, source="visual_open_set"),
        identity_track("structure_verified", verified="012", candidate="012", score=0.91, source="structure_semantic"),
    ])

    assert identity["identity_state"] == "structure_verified"
    assert identity["verified_identity"] == "012"


def test_entity_conflicts_when_members_verify_different_identities():
    identity = aggregate_entity_identity([
        identity_track("confirmed", verified="012", candidate="012", score=1.0, source="exact_hull"),
        identity_track("structure_verified", verified="003", candidate="003", score=0.90, source="structure_semantic"),
    ])

    assert identity["identity_state"] == "conflicting"
    assert identity["verified_identity"] == ""


def test_entity_confirmation_overrides_redundant_review_fragment():
    identity = aggregate_entity_identity([
        identity_track("structure_verified", verified="012", candidate="012", score=0.91),
        identity_track("review_requested", candidate="012", score=0.82, before_review="uncertain", review_frame=50),
    ])

    assert identity["identity_state"] == "structure_verified"
    assert identity["verified_identity"] == "012"


def test_entity_keeps_review_for_distinct_conflicting_identity():
    identity = aggregate_entity_identity([
        identity_track("structure_verified", verified="012", candidate="012", score=0.91),
        identity_track("review_requested", candidate="003", score=0.86, before_review="conflicting", review_frame=50),
    ])

    assert identity["identity_state"] == "review_requested"
    assert identity["verified_identity"] == ""
    assert identity["archive_candidate_id"] == "003"


def solid_crop(bgr):
    crop = np.zeros((80, 160, 3), dtype=np.uint8)
    crop[:] = bgr
    return crop


def test_reconciler_links_similar_sequential_tracklets():
    reconciler = EntityReconciler({"min_appearance_similarity": 0.70, "min_association_score": 0.65})
    first = reconciler.observe(1, 10, (100, 100, 300, 220), solid_crop((255, 255, 255)), (720, 1280, 3))
    second = reconciler.observe(2, 30, (110, 105, 310, 225), solid_crop((250, 250, 250)), (720, 1280, 3))
    assert first.entity_id == second.entity_id
    assert second.reassociated is True
    assert second.source_track_id == 1
    assert second.member_track_ids == [1, 2]


def test_reconciler_does_not_merge_simultaneous_tracks():
    reconciler = EntityReconciler({"min_appearance_similarity": 0.70, "min_association_score": 0.65})
    first = reconciler.observe(1, 10, (100, 100, 300, 220), solid_crop((255, 255, 255)), (720, 1280, 3))
    second = reconciler.observe(2, 10, (105, 105, 305, 225), solid_crop((255, 255, 255)), (720, 1280, 3))
    assert first.entity_id != second.entity_id
    assert second.reassociated is False


def test_reconciler_does_not_merge_into_another_visible_track_when_detection_order_changes():
    reconciler = EntityReconciler({"min_appearance_similarity": 0.70, "min_association_score": 0.65})
    first = reconciler.observe(1, 10, (100, 100, 300, 220), solid_crop((255, 255, 255)), (720, 1280, 3))
    second = reconciler.observe(
        2,
        11,
        (105, 105, 305, 225),
        solid_crop((255, 255, 255)),
        (720, 1280, 3),
        visible_track_ids={1, 2},
    )
    assert first.entity_id != second.entity_id
    assert second.reassociated is False


def test_reconciler_uses_multi_view_prototypes_after_large_appearance_change():
    reconciler = EntityReconciler({
        "min_appearance_similarity": 0.70,
        "min_association_score": 0.65,
        "appearance_novelty_threshold": 0.95,
        "max_appearance_prototypes": 4,
    })
    first = reconciler.observe(1, 10, (100, 100, 300, 220), solid_crop((255, 255, 255)), (720, 1280, 3))
    reconciler.observe(1, 11, (105, 100, 305, 220), solid_crop((0, 255, 255)), (720, 1280, 3))
    second = reconciler.observe(2, 30, (110, 105, 310, 225), solid_crop((255, 255, 255)), (720, 1280, 3))
    assert first.entity_id == second.entity_id
    assert second.reassociated is True
    exported = reconciler.export()[0]
    assert len(exported["appearance_bank"]) >= 2


def test_reassociated_tracklet_inherits_entity_memory():
    manager = TrackManager(memory_mode="full")
    manager.record_observation(1, 10, (10, 10, 100, 80), 0.9, 0.8, frame_shape=(480, 640, 3), entity_id="E000001", member_track_ids=[1])
    manager.bind_result(1, "", "white vessel", 10, semantic_matches=[{"hull_number": "012", "score": 0.92}])
    manager.bind_db_match(1, "012", "white vessel", evidence_source="structure_semantic", confidence=0.92)
    inherited = manager.inherit_entity_state(2, 1, "E000001", [1, 2], 30)
    assert inherited.verified_identity == "012"
    assert inherited.recognized is True
    assert inherited.recognition_attempts == 1
    assert inherited.last_recognition_track_id == 1
    assert inherited.member_track_ids == [1, 2]
