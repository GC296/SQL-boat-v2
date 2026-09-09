import numpy as np
from pipeline.quality import compute_observation_quality

def test_quality_prefers_sharp_large_crop():
    sharp = np.zeros((160, 240, 3), dtype=np.uint8); sharp[:, ::2] = 255
    flat = np.full((30, 40, 3), 127, dtype=np.uint8)
    good = compute_observation_quality(sharp, (0, 0, 240, 160), (480, 640, 3), 0.9)
    poor = compute_observation_quality(flat, (0, 0, 40, 30), (480, 640, 3), 0.4)
    assert 0.0 <= poor.score <= good.score <= 1.0
    assert set(good.components) == {"brightness", "blur", "contrast", "clipping", "scale", "confidence"}
