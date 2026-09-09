from database.sql_source import SqlShipSource

def test_profile_and_experience_memory(tmp_path):
    source = SqlShipSource(str(tmp_path / "ships.db"))
    source.upsert_profile_memory("A12", db_match_id="A12", attributes=["white"], matched=True, frame_id=5)
    profile = source.get_profile_memory("A12")
    assert profile["successful_match_count"] == 1 and "white" in profile["common_attributes"]
    source.add_recognition_experience(track_id=1, frame_id=5, scene_type="visible", uncertainty=0.8, hull_number_pred="A12", db_match_id="A12", match_type="exact", final_correct=True, failure_reason="", action_taken="cloud_vlm_read", action_success=True)
    results = source.retrieve_recognition_experiences(scene_type="visible", uncertainty=0.75, top_k=3)
    assert results and results[0]["action_success"] == 1
    assert source.experience_action_hint(scene_type="visible", uncertainty=0.75, top_k=3)["action"] == "cloud_vlm_read"
