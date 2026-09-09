from database import ShipDatabase


def test_cosine_similarity_distinguishes_vectors():
    assert ShipDatabase._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert ShipDatabase._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert round(ShipDatabase._cosine_similarity([1.0, 1.0], [1.0, 0.0]), 4) == 0.7071
