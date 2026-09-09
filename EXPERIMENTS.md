> **???????** ???????????? `FINAL_EXPERIMENTS.md`???????????????? `experiments.run_experiment` ?????????????????????????

﻿# SQL-boat-v2 完整实验复现教程

本文档从原始可见光海事视频开始，详细说明环境安装、数据整理、数据划分、VLM 辅助建立船舶身份目录、自动轨迹生成、轨迹预标注、人工校正、记忆消融、策略消融、端云实验、风险实验、Orin 部署、统计检验和论文表格生成。

项目目录为 `D:\海事瞭望 Agent\SQL-boat-v2`。本文默认不使用红外，也不默认具有 AIS、GPS、雷达、真实距离、真实航速或真实航向。风险结论只能基于图像平面运动、目标框尺度变化、轨迹持续性、多目标上下文和身份不确定性，不能直接声称 CPA、TCPA 或真实碰撞距离。

## 1. 实验总体流程

建议严格按以下顺序执行。

1. 安装环境并运行单元测试。
2. 整理原始视频并检查可读性。
3. 按完整视频、航次、日期或地点划分训练集、验证集和测试集。
4. 建立 `manifest.jsonl`。
5. 启动 VLM 和 Embedding 服务。
6. 用一个视频完成 300 帧冒烟测试。
7. 对训练集运行自动检测和跟踪，导出每条轨迹的最佳代表图。
8. 使用 Web 数据库管理页面上传代表图，由 VLM 交互式生成舷号和描述，人工确认后写入基础身份目录。
9. 冻结只包含基础身份的数据库。
10. 使用训练集运行完整系统，自动积累档案记忆和识别经验。
11. 冻结最终实验数据库。
12. 对测试集生成自动轨迹、代表图和预标注。
13. 人工只校正测试轨迹的真实身份、已知或未知状态、风险标签和跟踪错误。
14. 执行观测质量、身份记忆、档案、经验、主动策略、网络和风险消融。
15. 在 Orin 上执行真实端云部署测试。
16. 完成参数敏感性、人工解释评价和统计检验。
17. 生成 `metrics.json`、`summary.csv` 和论文图表。

## 2. 项目中的三类身份信息

### 2.1 基础身份目录 `ships`

该表保存已知船舶的舷号或匿名 ID 和描述。可以手工添加，也可以在实验前通过 Web 页面上传图片，让 VLM 自动生成舷号和描述，再由人工修改确认后入库。未经检查的 VLM 输出不能作为真实身份真值。

### 2.2 档案记忆 `ship_profile_memory`

系统运行时自动更新，保存历史视觉描述、常见属性、观测次数、成功和失败匹配次数、常见误读以及最后出现时间。

### 2.3 经验记忆 `recognition_experience`

系统运行时自动写入，保存场景类型、不确定性、预测结果、匹配类型、失败原因、采取动作、动作结果和语义候选。

正确流程是先用训练集建立可靠的 `ships`，再运行训练集生成档案和经验，最后冻结数据库用于测试。

## 3. 环境安装

进入项目目录：

```powershell
cd "D:\海事瞭望 Agent\SQL-boat-v2"
```

创建环境：

```powershell
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果执行策略报错：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

安装依赖：

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -e ".[dev]"
```

静态检查：

```powershell
python -m compileall -q agent database pipeline experiments tests config.py
```

单元测试：

```powershell
python -m pytest -q
```

当前应至少显示 `10 passed`。

## 4. 配置文件详细说明

主配置文件是 `config.yaml`。正式实验前备份：

```powershell
Copy-Item config.yaml config.default.yaml -Force
```

### 4.1 VLM

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "YOUR_API_KEY"
  base_url: "http://localhost:7890/v1"
  temperature: 0.0
```

所有正式实验固定 `temperature: 0.0`。如果更换模型，所有实验必须全部重跑，不能混用模型结果。

### 4.2 Embedding

```yaml
embed:
  model: "Qwen3-Embedding-0.6B"
  api_key: "YOUR_API_KEY"
  base_url: "http://localhost:7891/v1"
```

档案语义检索实验必须启动 Embedding 服务。

### 4.3 数据库

```yaml
database:
  backend: "csv"
  csv_path: "./data/ships.csv"
  memory_sqlite_path: "./data/ship_memory.db"
```

论文实验统一使用 SQLite，方便冻结快照。

### 4.4 检测与跟踪

```yaml
pipeline:
  concurrent_mode: true
  target_fps: 0
  max_concurrent: 4
  max_queued_frames: 30
  process_every_n_frames: 15
  save_output_video: true
  save_screenshots: true
  prompt_mode: "detailed"
  yolo_model: "yolov8n.pt"
  device: ""
  conf_threshold: 0.25
  detect_every_n_frames: 2
  tracker: "bytetrack"
  detect_classes:
    - 8
```

离线精度实验固定 `target_fps: 0`。Orin 在线测试再设置真实帧率。COCO 模型类别 8 通常是 boat；自训练模型必须填写实际船舶类别索引。

### 4.5 记忆模式

`experiment.memory_mode` 可选：

- `none`：当前一次识别结果，不跨帧融合。
- `majority`：多帧候选普通多数投票。
- `tel`：质量、置信度和时间衰减加权的 Temporal Evidence Ledger。
- `tel_archive`：TEL 后进行身份目录核验。
- `full`：TEL、档案和完整策略。

### 4.6 策略模式

`experiment.policy_mode` 可选：

- `recognize_once`：每个目标识别一次。
- `fixed_interval`：固定帧间隔重复识别。
- `uncertainty`：不确定性达到阈值后再次查询。
- `uncertainty_archive`：额外考虑档案歧义。
- `experience`：使用历史经验提示。
- `full`：综合不确定性、风险、档案和经验。

### 4.7 需要明确开启和关闭的开关

观测质量开启：

```yaml
experiment:
  quality:
    enabled: true
```

观测质量关闭后所有观测质量视为 1：

```yaml
experiment:
  quality:
    enabled: false
```

TEL 开启：

```yaml
experiment:
  temporal_ledger:
    enabled: true
```

TEL 关闭后所有记忆模式退化为无跨帧融合：

```yaml
experiment:
  temporal_ledger:
    enabled: false
```

档案开启：

```yaml
experiment:
  archive:
    enabled: true
```

档案关闭后 `tel_archive` 和 `full` 退化为 `tel`：

```yaml
experiment:
  archive:
    enabled: false
```

经验开启：

```yaml
experiment:
  experience:
    enabled: true
```

经验关闭：

```yaml
experiment:
  experience:
    enabled: false
```

风险开启：

```yaml
experiment:
  risk:
    enabled: true
```

风险关闭后统一输出低风险：

```yaml
experiment:
  risk:
    enabled: false
```

## 5. 原始视频整理

推荐目录：

```text
D:\maritime_dataset\videos\V001.mp4
D:\maritime_dataset\videos\V002.mp4
D:\maritime_dataset\videos\V003.mp4
```

每个视频保存唯一 `video_id`，并记录日期、地点、航次、相机、分辨率、帧率、时长、天气、反光、抖动和多船情况。

检查单个视频：

```powershell
python -c "import cv2; p=r'D:\maritime_dataset\videos\V001.mp4'; c=cv2.VideoCapture(p); print('opened=',c.isOpened(),'frames=',int(c.get(cv2.CAP_PROP_FRAME_COUNT)),'fps=',c.get(cv2.CAP_PROP_FPS),'width=',int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),'height=',int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))); c.release()"
```

删除或修复完全无法解码、黑屏、重复和损坏视频。不要因为远距离、模糊、反光或遮挡而删除视频，这些是重要困难场景。

## 6. 数据集划分和 manifest

按完整视频、航次、日期或地点划分，禁止把同一视频的帧随机分入不同集合。推荐训练 60%、验证 20%、测试 20%。

复制模板：

```powershell
Copy-Item data/annotations/manifest.example.jsonl data/annotations/manifest.jsonl -Force
```

每行一个视频：

```json
{"video_id":"V001","video_path":"D:/maritime_dataset/videos/V001.mp4","split":"train","voyage_id":"voyage_001","location":"port_a","date":"2026-06-01"}
{"video_id":"V002","video_path":"D:/maritime_dataset/videos/V002.mp4","split":"val","voyage_id":"voyage_002","location":"port_b","date":"2026-06-05"}
{"video_id":"V003","video_path":"D:/maritime_dataset/videos/V003.mp4","split":"test","voyage_id":"voyage_003","location":"port_c","date":"2026-06-10"}
```

检查路径：

```powershell
python -c "import json,pathlib; rows=[json.loads(x) for x in pathlib.Path('data/annotations/manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; missing=[r['video_path'] for r in rows if not pathlib.Path(r['video_path']).exists()]; print('videos=',len(rows),'missing=',len(missing)); print('\n'.join(missing))"
```

检查集合数量：

```powershell
python -c "import json,collections,pathlib; rows=[json.loads(x) for x in pathlib.Path('data/annotations/manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; print(collections.Counter(r['split'] for r in rows))"
```

## 7. 模型服务与冒烟测试

检查 VLM：

```powershell
python -c "import httpx; print(httpx.get('http://localhost:7890/v1/models',timeout=10).status_code)"
```

检查 Embedding：

```powershell
python -c "import httpx; print(httpx.get('http://localhost:7891/v1/models',timeout=10).status_code)"
```

第一次冒烟建议临时设置 `pipeline.concurrent_mode: false`，便于定位错误。运行：

```powershell
python -m pipeline "D:\maritime_dataset\videos\V001.mp4" --demo --max-frames 300 --output output\smoke_V001.mp4 --verbose
```

检查检测框、track ID、VLM 返回、数据库查询、输出视频和连接错误。检测不到船时检查模型、类别和阈值；跟踪频繁断裂时只在验证集调整 `track_buffer` 和 `match_thresh`。

## 8. 实验前使用 VLM 交互式建立基础身份目录

### 8.1 对训练集生成自动轨迹和代表图

此时基础身份目录可能为空，因此建议建立 `config.bootstrap.yaml`：

```powershell
Copy-Item config.yaml config.bootstrap.yaml -Force
```

将以下配置改为：

```yaml
experiment:
  memory_mode: "tel"
  policy_mode: "uncertainty"
  quality:
    enabled: true
  temporal_ledger:
    enabled: true
  archive:
    enabled: false
  experience:
    enabled: false
  risk:
    enabled: false
```

目的如下：

- 开启质量评估，自动选择清晰代表图。
- 开启 TEL，允许同一轨迹融合多次舷号读取。
- 关闭档案，因为身份库尚未建立。
- 关闭经验，避免未确认结果影响策略。
- 关闭风险，因为这一阶段只生成身份代表图。

运行训练集：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.bootstrap.yaml --split train --run-name train_bootstrap --memory-mode tel --policy-mode uncertainty --network-profile lan
```

导出代表图：

```powershell
python -m experiments.bootstrap_annotations --run-dir experiment_outputs/train_bootstrap --output data/annotations/train_tracks_draft.jsonl --csv data/annotations/train_tracks_draft.csv --manifest data/annotations/manifest.jsonl --crop-dir data/annotations/train_representative_crops
```

### 8.2 启动数据库管理页面

```powershell
python -m web
```

浏览器访问 `http://127.0.0.1:8000`。

### 8.3 交互式生成身份目录

对准备作为已知身份的船执行：

1. 打开“数据库管理”。
2. 点击“上传图片识别”。
3. 从 `data/annotations/train_representative_crops` 选择清晰图。
4. 点击识别。
5. VLM 自动生成舷号和视觉描述。
6. 使用原视频、现场记录或可靠档案核对舷号。
7. 修正错误字符。
8. 删除描述中无依据的推测，只保留船型、船体颜色、上层建筑颜色和可见结构。
9. 点击确认添加。
10. 回到数据库列表检查结果。

应加入身份库的目标：舷号可靠、现场记录可核验、或同一匿名身份可以稳定确认。不能加入只靠 VLM 模糊猜测的目标，也不能把测试集未知目标加入目录。

不知道真实舷号但能确认是同一艘船时，可使用 `ship_001` 等匿名 ID。此时论文应使用 Vessel Identity Verification，不使用真实 Hull Number Accuracy 的表述。

### 8.4 检查身份目录

```powershell
python -c "from config import load_config; from database import ShipDatabase; db=ShipDatabase(load_config()); print('ships=',len(db)); print(list(db.items.items())[:10])"
```

### 8.5 冻结基础身份数据库

关闭 Web 服务后执行：

```powershell
Copy-Item data/ships.csv data/ships_identity_only.csv -Force
```

## 9. 使用训练集积累档案和经验

创建配置：

```powershell
Copy-Item config.yaml config.train_memory.yaml -Force
```

确保：

```yaml
experiment:
  memory_mode: "full"
  policy_mode: "full"
  quality:
    enabled: true
  temporal_ledger:
    enabled: true
  archive:
    enabled: true
  experience:
    enabled: true
  risk:
    enabled: true
```

运行训练集：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.train_memory.yaml --split train --run-name train_memory_build --memory-mode full --policy-mode full --network-profile lan
```

检查档案：

```powershell
python -c "from config import load_config; from database import ShipDatabase; db=ShipDatabase(load_config()); rows=db.list_profile_memory(); print('profile_count=',len(rows)); print(rows[:3])"
```

检查经验：

```powershell
python -c "from config import load_config; from database import ShipDatabase; db=ShipDatabase(load_config()); rows=db.retrieve_recognition_experiences(top_k=10,min_similarity=0.0); print('experience_sample=',len(rows)); print(rows[:3])"
```

冻结最终实验数据库：

```powershell
Copy-Item data/ships.csv data/ships_experiment_frozen.csv -Force
```

```powershell
Copy-Item data/ship_memory.db data/ship_memory_experiment_frozen.db -Force
```

如果已经生成向量库，同时冻结：

```powershell
Copy-Item vector_store vector_store_frozen -Recurse -Force
```

以后每组实验前恢复：

```powershell
Copy-Item data/ships_experiment_frozen.csv data/ships.csv -Force; Copy-Item data/ship_memory_experiment_frozen.db data/ship_memory.db -Force
```

向量库需要恢复时使用：

```powershell
Remove-Item vector_store -Recurse -Force -ErrorAction SilentlyContinue; Copy-Item vector_store_frozen vector_store -Recurse -Force
```

## 10. 测试集自动轨迹和轨迹标注

恢复数据库：

```powershell
Copy-Item data/ships_experiment_frozen.csv data/ships.csv -Force; Copy-Item data/ship_memory_experiment_frozen.db data/ship_memory.db -Force
```

运行测试集：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name annotation_bootstrap --memory-mode full --policy-mode full --network-profile lan
```

生成预标注、CSV 和代表图：

```powershell
python -m experiments.bootstrap_annotations --run-dir experiment_outputs/annotation_bootstrap --output data/annotations/tracks_to_annotate.jsonl --csv data/annotations/tracks_to_annotate.csv --manifest data/annotations/manifest.jsonl --crop-dir data/annotations/test_representative_crops
```

### 10.1 自动字段

- `video_id`：视频编号。
- `track_id`：ByteTrack 自动 ID。
- `frame_start` 和 `frame_end`：轨迹起止帧。
- `observation_count`：观测次数。
- `best_frame_id`：最佳质量帧。
- `best_bbox`：最佳帧目标框。
- `best_observation_quality`：质量评分。
- `representative_crop`：自动导出的目标图。
- `predicted_hull_number`：系统预测。
- `predicted_identity_state`：系统身份状态。
- `predicted_risk_label`：系统风险预测。

### 10.2 人工字段

- `known_or_unknown` 只能为 `known` 或 `unknown`。
- `hull_number` 仅填写经过核验的真实舷号或匿名 ID。
- `vessel_id` 用于标记跨轨迹的同一真实船舶。
- `scene_tags` 记录困难条件。
- `risk_label` 只能为 `low`、`medium`、`high`。
- `risk_reasons` 记录可追溯视频证据。
- `annotation_status` 完成后改为 `checked`。
- `annotator` 记录标注人。

不能把 `predicted_hull_number` 直接复制为真值，除非已经人工确认。

### 10.3 场景标签建议

`long_range`、`small_target`、`glare`、`motion_blur`、`camera_shake`、`partial_occlusion`、`multi_vessel`、`clear_view`、`backlight`、`hull_number_invisible`。

### 10.4 跟踪错误处理

ID switch：同一个 track 中途换成另一艘船，应拆成两条标注并调整起止帧。

Track fragmentation：同一船被分成多个 track，不修改自动日志，只给这些 track 填相同的 `vessel_id`。

最终保存为 `data/annotations/tracks.jsonl`，然后验证：

```powershell
python -m experiments.validate_annotations data/annotations/tracks.jsonl
```

## 11. 所有消融实验的公平性要求

每组实验必须固定：测试视频、YOLO 权重、检测阈值、ByteTrack 参数、VLM、温度、Prompt、Embedding、随机种子和数据库快照。

每组实验前必须执行：

```powershell
Copy-Item data/ships_experiment_frozen.csv data/ships.csv -Force; Copy-Item data/ship_memory_experiment_frozen.db data/ship_memory.db -Force
```

不能连续运行多个方法而不恢复数据库，因为运行会新增档案和经验。

每组使用唯一 `run_name`。重跑前删除旧目录，例如：

```powershell
Remove-Item experiment_outputs\identity_tel -Recurse -Force -ErrorAction SilentlyContinue
```

## 12. 实验一：观测质量消融

### 12.1 Quality Off

创建配置：

```powershell
Copy-Item config.yaml config.quality_off.yaml -Force
```

修改：

```yaml
experiment:
  quality:
    enabled: false
```

其余保持 `memory_mode: full`、`policy_mode: full`、TEL 开启、档案开启、经验开启和风险开启。

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.quality_off.yaml --split test --run-name anchor_quality_off --memory-mode full --policy-mode full --network-profile lan
```

### 12.2 Quality On

创建配置：

```powershell
Copy-Item config.yaml config.quality_on.yaml -Force
```

修改：

```yaml
experiment:
  quality:
    enabled: true
    min_query_quality: 0.25
```

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.quality_on.yaml --split test --run-name anchor_quality_on --memory-mode full --policy-mode full --network-profile lan
```

比较平均查询质量、低质量过滤率、VLM 调用数、每目标调用数、确认率、未解决率、识别成功率和确认时间。

没有人工 MOT 标注时，不报告 HOTA、IDF1 或标准 fragmentation 改进，只将该实验描述为 Observation Selection and System Efficiency Analysis。

## 13. 实验二：身份记忆消融

所有方法统一开启观测质量、TEL 总开关、档案总开关、经验总开关和风险模块，只通过 `memory_mode` 与 `policy_mode` 控制方法。每组前恢复数据库。

### 13.1 Single Frame

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name identity_single_frame --memory-mode none --policy-mode recognize_once --network-profile lan
```

该组不使用跨帧融合，每个目标只查询一次。

### 13.2 Fixed Majority

确保 `experiment.policy.fixed_interval_frames: 150`。

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name identity_fixed_majority --memory-mode majority --policy-mode fixed_interval --network-profile lan
```

### 13.3 TEL

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name identity_tel --memory-mode tel --policy-mode uncertainty --network-profile lan
```

该组使用质量、置信度和时间衰减，但不使用档案确认。

### 13.4 TEL plus Archive

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name identity_tel_archive --memory-mode tel_archive --policy-mode uncertainty_archive --network-profile lan
```

### 13.5 Full Memory

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name identity_full_memory --memory-mode full --policy-mode full --network-profile lan
```

### 13.6 评测命令

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_single_frame --annotations data/annotations/tracks.jsonl
```

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_fixed_majority --annotations data/annotations/tracks.jsonl
```

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_tel --annotations data/annotations/tracks.jsonl
```

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_tel_archive --annotations data/annotations/tracks.jsonl
```

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_full_memory --annotations data/annotations/tracks.jsonl
```

报告 Exact Hull Number Accuracy、Normalized Edit Similarity、Identity Verification Accuracy、Unknown Precision、Unknown Recall、Unknown F1、False Identity Rate、Conflict Resolution Rate、Time to Confirmation 和 VLM Calls per Target。

## 14. 实验三：档案记忆单独消融

### 14.1 Archive Off

```powershell
Copy-Item config.yaml config.archive_off.yaml -Force
```

设置：

```yaml
experiment:
  archive:
    enabled: false
  experience:
    enabled: false
```

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.archive_off.yaml --split test --run-name archive_off --memory-mode tel --policy-mode uncertainty --network-profile lan
```

### 14.2 Archive On

```powershell
Copy-Item config.yaml config.archive_on.yaml -Force
```

设置：

```yaml
experiment:
  archive:
    enabled: true
  experience:
    enabled: false
```

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.archive_on.yaml --split test --run-name archive_on --memory-mode tel_archive --policy-mode uncertainty_archive --network-profile lan
```

关注已知身份准确率、Unknown F1、False Identity Rate、档案 Top 1、Top 3、冲突消解率和确认时间。

## 15. 实验四：经验记忆消融

### 15.1 Experience Off

```powershell
Copy-Item config.yaml config.experience_off.yaml -Force
```

设置：

```yaml
experiment:
  experience:
    enabled: false
```

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.experience_off.yaml --split test --run-name experience_off --memory-mode full --policy-mode uncertainty_archive --network-profile lan
```

### 15.2 Experience On

```powershell
Copy-Item config.yaml config.experience_on.yaml -Force
```

设置：

```yaml
experiment:
  experience:
    enabled: true
    top_k: 5
    min_similarity: 0.60
```

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.experience_on.yaml --split test --run-name experience_on --memory-mode full --policy-mode experience --network-profile lan
```

比较重复失败后的成功率、调用数、确认时间、错误持续时间、未解决率和动作成功率。

## 16. 实验五：主动策略消融

所有策略固定 `memory_mode=full`，质量、TEL、档案、经验和风险全部开启，只改变 `policy_mode`。

Recognize Once：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name policy_recognize_once --memory-mode full --policy-mode recognize_once --network-profile lan
```

Fixed Interval：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name policy_fixed_interval --memory-mode full --policy-mode fixed_interval --network-profile lan
```

Uncertainty：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name policy_uncertainty --memory-mode full --policy-mode uncertainty --network-profile lan
```

Uncertainty plus Archive：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name policy_archive --memory-mode full --policy-mode uncertainty_archive --network-profile lan
```

Experience：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name policy_experience --memory-mode full --policy-mode experience --network-profile lan
```

Full：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name policy_full --memory-mode full --policy-mode full --network-profile lan
```

固定间隔参数为 `experiment.policy.fixed_interval_frames`，不确定性参数为 `uncertainty_threshold` 和 `min_gap_frames`。报告确认率、未解决率、每目标调用数、确认时间、错误身份持续时间、等待更好视角比例、查询成功率、误上报率和上传量。

## 17. 实验六：端云网络实验

LAN：约 10 ms 延迟、3 ms 抖动、100 Mbps、无失败。

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name network_lan --memory-mode full --policy-mode full --network-profile lan
```

Stable Mobile：约 60 ms 延迟、20 ms 抖动、20 Mbps、无失败。

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name network_stable_mobile --memory-mode full --policy-mode full --network-profile stable_mobile
```

Limited：约 180 ms 延迟、60 ms 抖动、3 Mbps、失败率 0.01。

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name network_limited --memory-mode full --policy-mode full --network-profile limited
```

Interruption：约 120 ms 延迟、80 ms 抖动、8 Mbps、失败率 0.15。

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split test --run-name network_interruption --memory-mode full --policy-mode full --network-profile interruption
```

报告 Cloud Round Trip、P50、P95、每目标上传量、请求失败率、确认率、未解决率和退化模式完成率。模拟网络与真实 Orin 网络必须分别报告。

## 18. 实验七：风险模块消融

Risk Off：

```powershell
Copy-Item config.yaml config.risk_off.yaml -Force
```

设置 `experiment.risk.enabled: false`，运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.risk_off.yaml --split test --run-name risk_off --memory-mode full --policy-mode full --network-profile lan
```

Risk On：

```powershell
Copy-Item config.yaml config.risk_on.yaml -Force
```

设置：

```yaml
experiment:
  risk:
    enabled: true
    scale_growth_threshold: 0.012
    image_motion_threshold: 0.10
    persistence_frames: 15
    identity_uncertainty_threshold: 0.70
    multi_target_threshold: 3
    medium_threshold: 0.30
    high_threshold: 0.65
```

运行：

```powershell
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.risk_on.yaml --split test --run-name risk_on --memory-mode full --policy-mode full --network-profile lan
```

报告 Macro F1、High Risk Recall、False Alerts per Hour、Risk Trend Accuracy、Time to First Alert、日志完整率和证据覆盖率。当前代码直接支持风险开关；若论文需要 Single Frame、Trajectory Rule、Temporal Evidence、Full Evidence 四个严格基线，还需进一步增加 `risk_mode` 配置后再运行。

## 19. 风险解释人工评价

从测试集分层抽取 50 至 100 个事件，覆盖低、中、高风险，已知和未知身份，多船、远距离、反光、模糊和遮挡。至少两名标注者独立评价。

每条解释按 1 至 5 分评价 Correctness、Evidence Consistency、Usefulness、Clarity 和 Overclaiming。Overclaiming 反向计分。报告均值、标准差和 Cohen kappa、加权 kappa 或 Krippendorff alpha。

事件日志完整率检查每个关键事件是否包含目标、时间、证据、判断、动作和结果六项。证据覆盖率检查解释中的原因是否能回溯到轨迹、身份、风险或动作日志。

## 20. Orin 真实部署实验

边缘端运行视频采集、YOLO、ByteTrack、Crop、Observation Quality 和本地策略。服务器运行 VLM、Embedding、档案检索、经验检索和解释生成。

建议 Orin 配置：

```yaml
pipeline:
  concurrent_mode: true
  target_fps: 15
  max_concurrent: 2
  max_queued_frames: 30
  process_every_n_frames: 15
  detect_every_n_frames: 2
  device: "0"
  save_output_video: false
  save_screenshots: false
experiment:
  enabled: true
  quality:
    enabled: true
  network:
    simulate: false
```

启动性能记录：

```bash
tegrastats --interval 1000 > tegrastats_full_system.log
```

每种条件至少运行 10 分钟：单目标、多目标、远距离、反光、稳定网络、受限网络和临时断网。

记录 Edge FPS、YOLO latency、tracking latency、quality latency、CPU、GPU、RAM、显存、功耗、温度和丢帧率。真实部署结果不能与模拟网络结果混为同一表述。

## 21. 参数敏感性实验

所有参数只在验证集搜索，测试集只运行最终值。

### 21.1 TEL decay

测试 `0.90、0.94、0.97、0.99、1.00`，修改：

```yaml
experiment:
  temporal_ledger:
    decay: 0.97
```

观察准确率、冲突消解率和确认时间。

### 21.2 TEL min support

测试 `0.10、0.20、0.30、0.40、0.50`，修改 `experiment.temporal_ledger.min_support`。

### 21.3 Unknown threshold

测试 `0.30、0.40、0.45、0.50、0.60`，修改：

```yaml
experiment:
  archive:
    unknown_threshold: 0.45
```

观察 Unknown Precision、Recall、F1 和 False Identity Rate。

### 21.4 Uncertainty threshold

测试 `0.45、0.55、0.65、0.75、0.85`，修改：

```yaml
experiment:
  policy:
    uncertainty_threshold: 0.65
```

观察调用数、确认率、确认时间和上传量。

### 21.5 Min query quality

测试 `0.00、0.15、0.25、0.35、0.45`，修改：

```yaml
experiment:
  quality:
    min_query_quality: 0.25
```

观察查询时质量、每次调用成功率、未解决率和确认时间。

每个参数值建立独立配置和唯一 `run_name`，每次恢复数据库。不要在测试集根据结果选择最优参数。

## 22. 输出文件解释

每个运行目录位于 `experiment_outputs/<run_name>`。

- `resolved_config.yaml`：实际使用的完整配置，必须保留。
- `observations.jsonl`：视频 ID、track ID、帧号、框、置信度、质量分数和质量分量。
- `actions.jsonl`：策略动作、是否查询、得分、原因、memory mode 和 policy mode。
- `recognition.jsonl`：原始舷号、融合舷号、身份状态、匹配类型、语义候选、不确定性和风险。
- `edge_cloud.jsonl`：上传量、网络时延、云端时间、总时延、成功状态和结果。
- `summary.json`：一次流水线摘要。
- `metrics.json`：评测指标。

无人工标注评测：

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_full_memory
```

有轨迹标注评测：

```powershell
python -m experiments.evaluate --run-dir experiment_outputs/identity_full_memory --annotations data/annotations/tracks.jsonl
```

汇总：

```powershell
python -m experiments.summarize --root experiment_outputs --output experiment_outputs/summary.csv
```

## 23. 重复实验和统计检验

网络抖动、网络失败、并发调度、Orin 性能和人工评价至少重复三次，报告 `mean ± standard deviation`。

关键成对比较建议使用配对 bootstrap 或 Wilcoxon signed-rank test，包括 Single Frame 与 TEL、TEL 与 TEL plus Archive、Uncertainty 与 Full Policy、Quality Off 与 Quality On。除 p 值外建议报告效应量。

## 24. 论文结果表和图

### 24.1 数据集表

视频数量、总时长、自动轨迹数、已知身份数、未知轨迹数、平均轨迹长度、场景标签和风险标签分布。

### 24.2 身份主表

方法包括 Single Frame、Fixed Majority、TEL、TEL plus Archive 和 Full Memory。指标包括 Exact Accuracy、Edit Similarity、Verification Accuracy、Unknown F1、False Identity Rate、Confirmation Time 和 Calls per Target。

### 24.3 策略表

方法包括 Recognize Once、Fixed Interval、Uncertainty、Archive、Experience 和 Full。指标包括 Confirmed Rate、Unresolved Rate、Calls、Confirmation Time、Upload Bytes 和 False Escalation。

### 24.4 端云表

条件包括 LAN、Stable Mobile、Limited、Interruption 和 Real Orin。指标包括 Edge FPS、Cloud Latency、P50、P95、Upload Bytes 和 Completion Rate。

### 24.5 风险表

方法包括 Single Frame、Trajectory Rule、Temporal Evidence 和 Full Evidence。指标包括 Macro F1、High Risk Recall、False Alerts per Hour、Log Completeness、Evidence Coverage 和 Human Rating。

### 24.6 定性成功案例

展示首次远距离出现、第一次误读、更清晰观测、TEL 累积、档案核验、身份确认、风险解释和策略日志。

### 24.7 失败案例

展示目标过小、严重反光、长时间遮挡、ID switch、舷号不可见、未知船误匹配和网络中断。

## 25. 最终检查表

数据阶段：

- [ ] 所有视频可解码。
- [ ] manifest 路径全部存在。
- [ ] 按完整视频或航次划分。
- [ ] 测试集未用于调参。

身份库阶段：

- [ ] 使用训练集代表图。
- [ ] VLM 输出经过人工确认。
- [ ] 测试未知船未进入目录。
- [ ] 已保存 `ships_identity_only.csv`。
- [ ] 已保存 `ships_experiment_frozen.csv`
- [ ] 已保存 `ship_memory_experiment_frozen.db`。

标注阶段：

- [ ] 自动轨迹已生成。
- [ ] 代表图已导出。
- [ ] 已知或未知已校正。
- [ ] 真实身份已校正。
- [ ] 风险标签已校正。
- [ ] ID switch 已处理。
- [ ] 标注验证通过。

消融阶段：

- [ ] 每组前恢复数据库。
- [ ] 每组使用唯一运行名。
- [ ] 每次只改变目标模块。
- [ ] 保存 resolved config。
- [ ] 质量消融完成。
- [ ] 身份记忆消融完成。
- [ ] 档案消融完成。
- [ ] 经验消融完成。
- [ ] 策略消融完成。
- [ ] 网络实验完成。
- [ ] 风险实验完成。

部署和论文阶段：

- [ ] Orin FPS、功耗和温度已记录。
- [ ] 真实网络与模拟网络分开报告。
- [ ] 所有运行生成 metrics。
- [ ] 已生成 summary CSV。
- [ ] 关键实验完成统计检验。
- [ ] 已制作成功和失败案例。
- [ ] 没有无依据的 CPA、TCPA 或真实距离声称。

## 26. 最小可执行投稿方案

如果数据和时间有限，建议至少完成：

1. 30 段视频和完整 manifest。
2. 训练集代表图自动导出。
3. 通过 Web 页面建立至少 20 个可靠身份。
4. 运行训练集生成档案和经验并冻结数据库。
5. 测试集生成至少 100 条自动轨迹。
6. 人工校正 50 至 100 条轨迹。
7. 完成 Single Frame、TEL、TEL plus Archive、Full Memory。
8. 完成 Recognize Once、Fixed Interval、Uncertainty、Full Policy。
9. 完成 LAN、Limited、Interruption。
10. Orin 至少运行 30 分钟并记录资源。
11. 标注至少 50 个风险事件。
12. 生成身份、策略、端云和风险四张主表。

完成以上内容后，论文具备基本完整的实验支撑链条。


