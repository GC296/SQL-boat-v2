from agent import AgentResult
from pipeline.pipeline import ShipPipeline
from pipeline.tracker import TrackInfo


class FakeTracker:
    def get(self, track_id):
        return TrackInfo(track_id=track_id)


class FakeDatabase:
    def __init__(self):
        self.profile_writes = 0
        self.experience_writes = 0

    def upsert_profile_memory(self, *args, **kwargs):
        self.profile_writes += 1

    def add_recognition_experience(self, **kwargs):
        self.experience_writes += 1


def make_pipeline(profile_write, experience_write):
    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._tracker = FakeTracker()
    pipeline._db = FakeDatabase()
    pipeline._profile_memory_write = profile_write
    pipeline._experience_memory_write = experience_write
    return pipeline


def test_test_mode_does_not_write_long_term_memory():
    pipeline = make_pipeline(False, False)
    pipeline._record_memory_updates(1, 10, AgentResult(hull_number="A12", description="white ship", match_type="exact"), "recognize")
    assert pipeline._db.profile_writes == 0
    assert pipeline._db.experience_writes == 0


def test_archive_construction_mode_writes_long_term_memory():
    pipeline = make_pipeline(True, True)
    pipeline._record_memory_updates(1, 10, AgentResult(hull_number="A12", description="white ship", match_type="exact"), "recognize")
    assert pipeline._db.profile_writes == 1
    assert pipeline._db.experience_writes == 1
