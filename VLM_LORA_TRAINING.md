# Vessel VLM LoRA Training

This workflow adapts the server-side VLM for hull-number reading and structured vessel evidence extraction. It does not train the entity policy or overwrite the existing AWQ service.

## 1. Install the training environment

Run this on the Linux GPU server, not on Jetson Orin. Conda is recommended because the training dependencies include CUDA-enabled PyTorch and bitsandbytes:

```bash
conda create -n ship-lora python=3.11 pip -y
conda activate ship-lora
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[train]"
```

Install the PyTorch build matching the server driver/CUDA before `pip install -e ".[train]"` when the default package index does not provide the correct CUDA build. Check the working Agent environment with:

```bash
conda run -n agentl python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

As an alternative, install `python3-venv` and use a virtual environment, but it is not required when Conda is available.

Use the non-AWQ training checkpoint, for example `Qwen/Qwen3-VL-4B-Instruct`. `Qwen/Qwen3-VL-4B-AWQ` remains the rollback inference model.

The training checkpoint must be the same model family and parameter size as the served AWQ checkpoint. Check the actual model path used by the existing vLLM command before downloading. If the existing AWQ path is a Qwen3.5 2B checkpoint, download its matching non-AWQ Instruct checkpoint instead of using the 4B example below.

Download the base checkpoint once on the server:

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /media/ddc/新加卷/models/Qwen3-VL-4B-Instruct
```

Replace the model name and local path with the exact matching checkpoint. Keep the existing AWQ directory unchanged. The non-AWQ model usually needs roughly 10--20 GB; leave additional space for checkpoints and the merged export.

## 2. Prepare annotations

Each JSONL row needs an image path, split, physical-vessel group, hull visibility, and the fixed identity schema. See `data/annotations/vlm_lora.example.jsonl`.

External vessels and project video crops can be mixed in the same file. `vessel_id` must identify the physical vessel. A vessel cannot appear in more than one split.

For a project Tracklet annotation, `representative_crop` is accepted as the image path. If `hull_visible=false`, the builder deliberately replaces the target hull number with an empty string so the adapter cannot learn hidden archive identity.

To create project annotation drafts and representative crops from an existing experiment run:

```bash
python -m experiments.bootstrap_annotations \
  --run-dir experiment_outputs/<train_run> \
  --manifest data/annotations/manifest.jsonl \
  --output data/annotations/vlm_lora_train_draft.jsonl \
  --crop-dir data/vlm_lora/images/train
```

Create a reviewable evidence-label draft by joining Tracklet labels with the latest recognition log:

```bash
python -m training.bootstrap_vlm_annotations \
  --tracks data/annotations/vlm_lora_train_draft.jsonl \
  --run-dir experiment_outputs/<train_run> \
  --output data/annotations/vlm_lora_train_review.jsonl
```

Only the visible evidence fields need human correction. Do not manually write `description`: fill or correct `identity_features`, and the dataset builder derives a canonical description from those fields. Existing VLM descriptions and archive descriptions are references for annotation, not ground truth for invisible attributes. Do not use `data/annotations/test_tracks.jsonl` as training data; reserve the held-out test videos for final evaluation.

For external vessels, place the images on the server and create rows using `data/annotations/vlm_lora.example.jsonl` as the template. External images are valid training data, but their labels must be manually checked and their `vessel_id` values must be unique within the split.

### Teacher pseudo-label workflow

To avoid full manual attribute annotation, first build an image dataset without `--require-attributes`:

```bash
python -m training.build_vlm_lora_dataset \
  --annotations data/annotations/vlm_lora_train_review.jsonl \
  --annotations data/annotations/vlm_lora_val_review.jsonl \
  --image-root . \
  --output data/vlm_lora/teacher_input.jsonl \
  --allow-pending
```

Run a stronger teacher VLM on the training and validation splits:

```bash
python -m training.evaluate_vlm_endpoint \
  --dataset data/vlm_lora/teacher_input.jsonl \
  --split train \
  --base-url http://localhost:7893/v1 \
  --model vessel-teacher \
  --output-dir experiment_outputs/teacher_train

python -m training.evaluate_vlm_endpoint \
  --dataset data/vlm_lora/teacher_input.jsonl \
  --split val \
  --base-url http://localhost:7893/v1 \
  --model vessel-teacher \
  --output-dir experiment_outputs/teacher_val
```

Filter and promote the teacher outputs into direct LoRA targets:

```bash
python -m training.promote_pseudo_labels \
  --predictions experiment_outputs/teacher_train/predictions.jsonl \
  --predictions experiment_outputs/teacher_val/predictions.jsonl \
  --output data/vlm_lora/pseudo_train_val.jsonl \
  --min-attributes 2
```

The promoter rejects invalid JSON, predictions with too few attributes, and readable-hull predictions that disagree with the trusted Tracklet hull label. For unreadable frames, the target hull remains empty regardless of the teacher prediction. Use `pseudo_train_val.jsonl` as the training dataset. Keep the manually verified test set separate for final evaluation.

Build and validate the dataset:

```bash
python -m training.build_vlm_lora_dataset \
  --annotations data/annotations/vlm_lora_train.jsonl \
  --annotations data/annotations/vlm_lora_val.jsonl \
  --annotations data/annotations/vlm_lora_test.jsonl \
  --image-root . \
  --output data/vlm_lora/dataset.jsonl \
  --require-attributes
```

The command fails when the same `vessel_id` occurs in multiple splits. It also writes `data/vlm_lora/dataset.summary.json`.

## 3. Train QLoRA

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_vlm_lora \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/vlm_lora/dataset.jsonl \
  --output-dir outputs/vessel-evidence-lora \
  --epochs 3 \
  --rank 16 \
  --alpha 32 \
  --batch-size 1 \
  --gradient-accumulation 16
```

The visual tower is frozen because LoRA modules under visual/vision paths are excluded. The adapter targets language attention and MLP projections. QLoRA is enabled by default; pass `--no-qlora` only when the server has enough VRAM for BF16 training.

For a smoke test before a full run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_vlm_lora \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/vlm_lora/dataset.jsonl \
  --output-dir outputs/vessel-evidence-lora-smoke \
  --max-steps 10
```

## 4. Evaluate base and LoRA endpoints

Keep prompts and decoding identical. Evaluate the base endpoint first:

```bash
python -m training.evaluate_vlm_endpoint \
  --dataset data/vlm_lora/dataset.jsonl \
  --split test \
  --base-url http://localhost:7890/v1 \
  --model Qwen/Qwen3-VL-4B-AWQ \
  --output-dir experiment_outputs/vlm_base
```

Then evaluate the adapter endpoint:

```bash
python -m training.evaluate_vlm_endpoint \
  --dataset data/vlm_lora/dataset.jsonl \
  --split test \
  --base-url http://localhost:7892/v1 \
  --model vessel-evidence-lora \
  --output-dir experiment_outputs/vlm_lora
```

The evaluator reports JSON validity, readable-hull exact accuracy, normalized edit similarity, unreadable-hull hallucination rate, attribute accuracy, and false attribute filling.

## 5. Serve without replacing the base model

Keep the existing AWQ server on port `7890`. Start a separate base-plus-adapter server on port `7892`:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 \
  --port 7892 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --enable-lora \
  --lora-modules vessel-evidence-lora=outputs/vessel-evidence-lora
```

Verify the exact vLLM release supports LoRA for the selected multimodal checkpoint. If it does not, merge the adapter and serve the merged model:

```bash
python -m training.merge_vlm_lora \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --adapter outputs/vessel-evidence-lora \
  --output-dir outputs/vessel-evidence-merged

CUDA_VISIBLE_DEVICES=1 vllm serve outputs/vessel-evidence-merged \
  --host 0.0.0.0 \
  --port 7892 \
  --served-model-name vessel-evidence-lora
```

Do not merge into or delete the existing AWQ directory.

## 6. Enable and roll back

Enable LoRA in `config.yaml`:

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  base_url: "http://localhost:7890/v1"
  use_lora: true
  lora_model: "vessel-evidence-lora"
  lora_base_url: "http://localhost:7892/v1"
  fallback_model: "Qwen/Qwen3-VL-4B-AWQ"
  fallback_base_url: "http://localhost:7890/v1"
  fallback_to_base: true
```

If the adapter endpoint fails or returns an unusable response, inference automatically retries the base AWQ endpoint. Recognition logs contain `vlm_model`, `vlm_base_url`, and `vlm_fallback_used`.

Immediate manual rollback requires only:

```yaml
llm:
  use_lora: false
```

Restart the Agent process after changing the configuration.

## 7. Final Agent evaluation

After LoRA-only evaluation passes, rerun the frozen Agent configurations with the same archive, thresholds, prompts, and policy settings. At minimum compare KAcc, URec, UFAR, ATS, VLM/entity, hull exact accuracy, JSON validity, and hull hallucination rate. Recalibrate evidence thresholds on the development split only if LoRA changes score distributions.
