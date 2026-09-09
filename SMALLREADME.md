# SeaAgent 启动命令

```bash
pip install -e .
python -m web.app
```
```bash
modelscope download --model Qwen/Qwen3-VL-Embedding-2B --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-Embedding-2B
modelscope download --model tclf90/Qwen3.5-9B-AWQ --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-9B-AWQ
modelscope download --model cyankiwi/Qwen3.5-4B-AWQ-4bit --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-4B-AWQ-4bit
modelscope download --model Qwen/Qwen3.5-9B --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen/Qwen3.5-9B
modelscope download --model Qwen/Qwen3.5-4B --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-4B
modelscope download --model tclf90/Qwen3.6-27B-AWQ --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3.6-27B-AWQ
modelscope download --model cyankiwi/gemma-4-12B-it-AWQ-INT4 --local_dir /media/ddc/新加卷/hys/hysnew3/model/gemma-4-12B-it-AWQ-INT4
modelscope download --model Qwen/Qwen3-VL-Reranker-2B --local_dir /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-Reranker-2B

```
## 启动后端服务

### 1. LLM 推理（Qwen3-VL）

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.2 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-9B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 30240 \
  --port 7890 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3.6-27B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.5 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew3/model/gemma-4-12B-it-AWQ-INT4 \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-4B-AWQ-4bit \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 30240 \
  --port 7890 \
  --gpu-memory-utilization 0.25 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3.5-4B \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 30240 \
  --port 7890 \
  --gpu-memory-utilization 0.3 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen/Qwen3.5-9B\
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 20240 \
  --port 7890 \
  --gpu-memory-utilization 0.50 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
### 2. Embedding（Qwen3-Embedding-0.6B）
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3-Embedding-0.6B \
  --api-key abc123 \
  --served-model-name Qwen3-Embedding-0.6B \
  --convert embed \
  --gpu-memory-utilization 0.08 \
  --max-model-len 2048 \
  --port 7892
```
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-Embedding-2B" \
  --served-model-name "Qwen/Qwen3-VL-Embedding-2B" \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --dtype bfloat16 \
  --api-key abc123 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 40048 \
  --port 7891
```
### 3. reranker（Qwen3-Embedding-0.6B）
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-Reranker-2B \
  --served-model-name Qwen/Qwen3-VL-Reranker-2B \
  --task score \
  --trust-remote-code \
  --dtype bfloat16 \
  --api-key abc123 \
  --gpu-memory-utilization 0.2 \
  --max-model-len 8192 \
  --port 7894

<!-- {
  "model": "Qwen/Qwen3-VL-Reranker-2B",
  "query": "你的查询",
  "documents": ["文档1", "文档2", ...]
} -->
query = {
    "content": [
        {"type": "text", "text": "一张夕阳下的海滩照片"},
        {"type": "image_url", "image_url": {"url": "..."}}
    ]
}


```bash
seaagent
```
```bash
python -m pipeline.cli data/videos/example.mp4 --demo --output output/result.mp4
```
```bash
seaagent-pipeline data/videos/example.mp4 --demo --output output/result.mp4
```
