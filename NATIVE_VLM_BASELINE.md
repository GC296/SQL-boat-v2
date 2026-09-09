# Native Video-VLM Replacement Baseline

This baseline tests whether one monolithic Video-VLM can replace the ECEM pipeline for end-to-end open-set vessel identity verification.

The runner does not import or invoke `ShipPipeline`, YOLO, ByteTrack, entity reconciliation, DINOv2, text embedding, archive retrieval, temporal memory, or the Agent policy. Each video produces one logical model call containing the video, the complete `ships.csv` description archive, and the complete task prompt. Archive images are intentionally not used.

## Files to sync

```text
experiments/run_native_vlm.py
configs/native_qwen3vl4b.example.yaml
```

## Configure Qwen3-VL

Create the runtime configuration:

```bash
cp configs/native_qwen3vl4b.example.yaml configs/native_qwen3vl4b.yaml
```

`input_mode: video_url` sends the original video through an OpenAI-compatible `video_url` message. The model server must be able to read the local video URI. For vLLM, start the server with an `--allowed-local-media-path` that contains every path referenced by `manifest.jsonl`, and permit one video item per prompt, for example:

```bash
--allowed-local-media-path /media/ddc/新加卷/gc/SQL-boat-v2 --limit-mm-per-prompt '{"video":1}'
```

If the serving stack does not support `video_url`, set:

```yaml
native_vlm:
  input_mode: sampled_frames
  sample_fps: 1.0
  max_frames: 64
```

This uniformly samples the complete temporal extent and still performs one model request per video.
When using 64 sampled images with vLLM, configure the corresponding image limit, for example `--limit-mm-per-prompt '{"image":64}'`.

## Validate inputs without inference

```bash
python -m experiments.run_native_vlm --manifest data/annotations/manifest.jsonl --archive data/ships.csv --config configs/native_qwen3vl4b.yaml --split test --run-name native_qwen3vl4b_dry --max-videos 1 --dry-run
```

Inspect:

```text
experiment_outputs/native_qwen3vl4b_dry/archive_context.txt
experiment_outputs/native_qwen3vl4b_dry/predictions.jsonl
experiment_outputs/native_qwen3vl4b_dry/run_summary.json
```

## One-video smoke inference

```bash
python -m experiments.run_native_vlm --manifest data/annotations/manifest.jsonl --archive data/ships.csv --config configs/native_qwen3vl4b.yaml --split test --run-name native_qwen3vl4b_smoke --max-videos 1
```

## Full test run

```bash
python -m experiments.run_native_vlm --manifest data/annotations/manifest.jsonl --archive data/ships.csv --config configs/native_qwen3vl4b.yaml --split test --run-name native_qwen3vl4b
```

The model response is not matched, corrected, normalized, or rescored by an automatic evaluator. Inspect `raw_responses.jsonl` and score the final identity decisions against the annotations through blind manual review. The model must not receive ground-truth labels during inference.

## Run artifacts

```text
predictions.jsonl              parsed model JSON preserved without decision correction
raw_responses.jsonl            unmodified model responses and token usage
errors.jsonl                   request or video failures
run_summary.json               request count and average latency
```
