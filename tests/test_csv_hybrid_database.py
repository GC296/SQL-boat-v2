from database import ShipDatabase
from database.csv_source import CsvShipSource


def test_csv_catalogue_uses_sqlite_memory_sidecar(tmp_path):
    csv_path = tmp_path / "ships.csv"
    memory_path = tmp_path / "memory.db"
    config = {
        "database": {"backend": "csv", "csv_path": str(csv_path), "memory_sqlite_path": str(memory_path)},
        "app": {"ship_db_path": str(csv_path)},
        "embed": {"model": "test", "api_key": "YOUR_API_KEY", "base_url": "http://localhost:1/v1"},
        "retrieval": {"top_k": 3, "score_threshold": 0.5},
        "vector_store": {"persist_path": str(tmp_path / "vectors"), "auto_rebuild": False},
    }
    database = ShipDatabase(config=config)
    assert database.add_ship("A12", "white patrol vessel")
    assert "A12" in csv_path.read_text(encoding="utf-8")
    database.upsert_profile_memory("A12", db_match_id="A12", attributes=["white"], matched=True, frame_id=10)
    database.add_recognition_experience(
        track_id=1, frame_id=10, scene_type="visible", uncertainty=0.2,
        hull_number_pred="A12", db_match_id="A12", match_type="exact",
        final_correct=True, failure_reason="", action_taken="recognize", action_success=True,
    )
    assert memory_path.exists()
    assert database.get_profile_memory("A12")["successful_match_count"] == 1
    assert database.retrieve_recognition_experiences(top_k=1, min_similarity=0.0)


def test_csv_catalogue_keeps_multiple_prototypes_for_same_hull(tmp_path):
    csv_path = tmp_path / "ships.csv"
    source = CsvShipSource(str(csv_path))
    source.load_all()
    first = source.add_prototype("012", "白色大型游船，尖形船首")
    second = source.add_prototype("012", "白色大型游船，侧面连续弧形玻璃窗")

    assert first["prototype_id"] == "012_01"
    assert second["prototype_id"] == "012_02"
    reloaded = CsvShipSource(str(csv_path))
    assert len(reloaded.load_prototypes()) == 2
    assert {item["description"] for item in reloaded.load_prototypes()} == {
        "白色大型游船，尖形船首",
        "白色大型游船，侧面连续弧形玻璃窗",
    }
    assert csv_path.read_text(encoding="utf-8-sig").splitlines()[0] == "prototype_id,hull_number,description"


def test_csv_catalogue_still_loads_legacy_description(tmp_path):
    csv_path = tmp_path / "ships.csv"
    csv_path.write_text("hull_number,description\n003,黄色小型机动船\n", encoding="utf-8")
    source = CsvShipSource(str(csv_path))
    assert source.load_all() == {"003": "黄色小型机动船"}
    assert source.load_prototypes() == [{
        "prototype_id": "003_01",
        "hull_number": "003",
        "description": "黄色小型机动船",
    }]
