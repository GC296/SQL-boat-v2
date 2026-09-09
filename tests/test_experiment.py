import json
import numpy as np

from pipeline.experiment import ExperimentLogger

def test_experiment_logger_writes_context(tmp_path):
    logger = ExperimentLogger({"enabled": True, "run_name": "r1", "output_dir": str(tmp_path), "video_id": "V001", "split": "test"})
    logger.log("actions", {"track_id": 3})
    record = json.loads((tmp_path / "r1" / "actions.jsonl").read_text(encoding="utf-8").strip())
    assert record["video_id"] == "V001" and record["track_id"] == 3


def test_experiment_logger_saves_attributed_target_crop(tmp_path):
    logger = ExperimentLogger({
        "enabled": True,
        "run_name": "r1",
        "output_dir": str(tmp_path),
        "video_id": "V001",
        "save_crops": True,
    })
    path = logger.save_image("targets", np.zeros((16, 24, 3), dtype=np.uint8), entity_id="E1", track_id=3, frame_id=9)

    assert path.startswith("evidence/targets/")
    assert (tmp_path / "r1" / path).exists()
