from web.services.ship_service import ShipService


def test_web_service_appends_prototypes_for_existing_hull(tmp_path):
    csv_path = tmp_path / "ships.csv"
    config = {
        "database": {
            "backend": "csv",
            "csv_path": str(csv_path),
            "memory_sqlite_path": str(tmp_path / "memory.db"),
        },
        "app": {"ship_db_path": str(csv_path)},
        "embed": {"model": "test", "api_key": "YOUR_API_KEY", "base_url": "http://localhost:1/v1"},
        "retrieval": {"top_k": 3, "score_threshold": 0.5},
        "vector_store": {"persist_path": str(tmp_path / "vectors"), "auto_rebuild": False},
    }
    service = ShipService(config=config)

    first = service.create_ship("012", "白色大型游船，尖形船首")
    second = service.create_ship("012", "白色大型游船，侧面连续弧形玻璃窗")

    assert first["prototype_id"] == "012_01"
    assert second["prototype_id"] == "012_02"
    assert len(service.list_ships()) == 2
    assert service.stats()["total_ships"] == 1
    assert service.stats()["total_prototypes"] == 2
