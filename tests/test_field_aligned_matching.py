from database import ShipDatabase


class FakeSource:
    def __init__(self):
        self.records = [
            {"prototype_id": "012_01", "hull_number": "012", "description": "白色大型游船，尖形船首"},
            {"prototype_id": "012_02", "hull_number": "012", "description": "白色大型游船，侧面连续弧形玻璃窗"},
            {"prototype_id": "003_01", "hull_number": "003", "description": "黄色小型作业船，黑色舷外机"},
        ]

    def load_all(self):
        return {
            "012": "白色大型游船，尖形船首；白色大型游船，侧面连续弧形玻璃窗",
            "003": "黄色小型作业船，黑色舷外机",
        }

    def load_prototypes(self):
        return [dict(record) for record in self.records]


class FakeEmbeddings:
    vectors = {
        "白色大型游船，尖形船首": [1.0, 0.0, 0.0],
        "白色大型游船，侧面连续弧形玻璃窗": [0.8, 0.6, 0.0],
        "黄色小型作业船，黑色舷外机": [0.0, 0.0, 1.0],
        "画面可见白色大型游船和连续弧形玻璃窗": [0.8, 0.6, 0.0],
        "画面可见黄色小型作业船与黑色舷外机": [0.0, 0.0, 1.0],
    }

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


def make_database():
    database = ShipDatabase.__new__(ShipDatabase)
    database._source = FakeSource()
    database._data = database._source.load_all()
    database._embeddings = FakeEmbeddings()
    database._top_k = 3
    database._prototype_records = []
    database._prototype_embeddings = {}
    database._prototype_embedding_hash = ""
    return database


def test_best_prototype_is_used_as_vessel_identity_score():
    results = make_database().semantic_search_prototypes("画面可见白色大型游船和连续弧形玻璃窗")
    assert results[0]["hull_number"] == "012"
    assert results[0]["prototype_id"] == "012_02"
    assert results[0]["score"] == 1.0
    assert results[0]["similarity_metric"] == "multi_prototype_cosine"


def test_multiple_prototypes_of_same_vessel_are_not_separate_identity_candidates():
    results = make_database().semantic_search_prototypes("画面可见白色大型游船和连续弧形玻璃窗")
    assert [item["hull_number"] for item in results] == ["012", "003"]
    assert [item["prototype_id"] for item in results[0]["prototype_matches"]] == ["012_02", "012_01"]


def test_identity_ranking_compares_different_hull_numbers():
    results = make_database().semantic_search_prototypes("画面可见黄色小型作业船与黑色舷外机")
    assert results[0]["hull_number"] == "003"
    assert results[1]["hull_number"] == "012"
