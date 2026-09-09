from __future__ import annotations

from pathlib import Path
from typing import Any

from training.common import read_jsonl


class JsonlSplitDataset:
    def __init__(self, path: Path, split: str):
        self.rows = [row for row in read_jsonl(path) if str(row.get("split", "train")) == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class VLMDataCollator:
    def __init__(self, processor: Any):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("install the train extra before LoRA training") from exc
        self.torch = torch
        self.processor = processor
        self.process_vision_info = process_vision_info

    @staticmethod
    def _messages(sample: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(sample["image"])},
                {"type": "text", "text": str(sample["prompt"])},
            ],
        }]
        if include_answer:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": str(sample["target_text"])}]})
        return messages

    def _processor_batch(self, conversations: list[list[dict[str, Any]]], add_generation_prompt: bool) -> dict[str, Any]:
        texts: list[str] = []
        images: list[Any] = []
        videos: list[Any] = []
        has_video = False
        for messages in conversations:
            texts.append(self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            ))
            image_inputs, video_inputs = self.process_vision_info(messages)
            images.append(image_inputs[0] if image_inputs else None)
            if video_inputs:
                has_video = True
                videos.append(video_inputs[0])
            else:
                videos.append(None)
        return self.processor(
            text=texts,
            images=images,
            videos=videos if has_video else None,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        full = [self._messages(sample, include_answer=True) for sample in samples]
        prompts = [self._messages(sample, include_answer=False) for sample in samples]
        batch = self._processor_batch(full, add_generation_prompt=False)
        prompt_batch = self._processor_batch(prompts, add_generation_prompt=True)
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : min(int(prompt_length), labels.shape[1])] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

