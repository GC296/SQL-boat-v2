# SQL-boat-v2 论文实验最终教程

本文档是实体级代码版本的正式实验流程。所有实验必须回答论文问题，不为了增加表格而执行无关消融。

论文的核心论点是：海事视频中的跟踪 ID 会因遮挡、转向、出画和重新检测而碎片化；海事瞭望 Agent 应维护持续的物理船舶实体，在实体级证据记忆上执行档案身份核验和主动观测，而不是把每个临时 `track_id` 当成一艘新船。

本文不将网络模拟包装为端云协同贡献。原有网络与服务调用代码继续保留用于工程运行，但论文只报告真实系统响应时间和边缘设备部署效率。

## 1. 论文论点与实验对应

| 论文论点 | 必做实验 | 主要指标 |
|---|---|---|
| Track ID碎片化破坏持续身份理解 | Entity Reconciliation On/Off | Pair Precision、Pair Recall、Pair F1 |
| 实体级记忆能够跨轨迹继承证据 | No Entity、No TEL、Full | Entity Identity Accuracy、Unresolved Rate |
| 主动策略减少固定重复调用 | Single、Fixed、Active TEL、Full | VLM Calls per Entity、身份指标 |
| 档案匹配支持开放集核验 | 已知与未知实体测试 | Known Accuracy、Unknown FAR、Unknown Rejection |
| 系统可以部署到Orin | 真实边缘部署 | FPS、RTF、P95、RAM、GPU、Power、Energy |

风险技能只有在数据中存在真实会遇或危险事件时才做定量实验。当前数据如果全部为正常绕拍，则风险部分只展示定性案例。经验记忆只有在具有独立历史任务数据时才作为贡献，否则只保留模块，不强行解释增益。

## 2. 三种ID

| 字段 | 含义 |
|---|---|
| `track_id` | ByteTrack产生的临时轨迹片段，同一艘船可能有多个 |
| `entity_id` | 系统自动重关联得到的视频内物理船实体 |
| `archive_id` | 档案库中的长期身份，例如`012`和`003` |

人工标注中的 `vessel_id` 只用于构造实体真值和评价。运行时禁止读取 `vessel_id` 进行重关联，否则属于标签泄漏。

## 3. 最终数据协议：无验证集

本项目不训练YOLO、VLM或Embedding模型，因此`train`不表示梯度训练。最终协议只保留档案构建和测试两种用途：

- `train`或`archive`：只用于构建文本与视觉档案，不参与最终指标。
- `test`：冻结配置和档案后执行的一次性最终测试。

取消验证集并不允许在测试集调参。当前`config.yaml`中的实体关联、身份阈值、开放集阈值和主动策略参数视为方法开发阶段已经确定的参数。新增100多个公开视频用于最终扩展测试；产生测试结果后禁止根据结果修改这些参数。

### 3.1 Manifest字段

每条视频至少包含`video_id`、`video_path`、`split`、`voyage_id`、`location`和`date`。最终论文数据还应增加：

- `dataset_subset`：`controlled`或`public`。
- `source_url`：公开视频原始链接，自采视频填写采集记录编号。
- `license`：公开许可、研究使用说明或自采数据授权状态。

同一`voyage_id`的全部切片必须保持相同用途，不能一部分建档、一部分测试。否则相邻切片会共享背景、视角和压缩特征。

### 3.2 两种档案协议

论文主表使用独立来源档案协议：

- 已知测试船的档案图片来自另一视频、官网图片或独立采集。
- 未知测试船在档案中没有对应身份。
- 只在档案中作为干扰项的身份不要求具有测试视频。

单视频船舶可以补充一个`same-encounter enrollment`辅助协议：从视频固定登记时段截取参考图，并只在时间隔离的查询时段评价。登记帧及其相邻近重复帧不得进入查询，身份技能在登记时段不得运行。若仍使用包含登记帧的完整视频计算身份指标，该结果只能标为`enrollment-inclusive upper bound`，不能进入论文主表。

文本和视觉档案对每个身份使用相同数量的原型，例如3个视觉原型和3条文本原型。文本描述可以按侧视、船首和船尾分别生成，但最终评价按`archive_id`而不是按原型条目计分。固定原型数量可以避免某一身份因条目更多而获得最大相似度优势。

### 3.3 数据预检

在服务器执行：

`python -m experiments.prepare_final_dataset --manifest data/annotations/manifest.jsonl --output-dir data/annotations/final_protocol --archive-splits train,archive --test-split test --probe-videos`

输出包括`manifest_audit.json`、`video_inventory.jsonl`、`video_inventory.csv`、`manual_review.csv`和`protocol_lock.json`。必须修复全部`errors`，并在正式实验前补齐公开来源、许可、预计实体数和独立档案来源。

## 4. 身份档案

`data/ships.csv`中每艘物理船只保留一个编号：

```csv
hull_number,description
012,"白色船体，具有大型弧形上层建筑，上层建筑侧面分布大面积深色玻璃窗，船尾下方具有多个黑色圆形固定装置。"
003,"黄色小型机动船，黑色上层建筑，驾驶舱和天线位于船体中前部，船尾安装舷外机，船体侧面具有稳定的红色编号或标志。"
```

档案描述必须来自档案构建来源的多视角图片，并经过人工确认。不得写入天气、背景和拍摄距离。主协议中的已知测试船必须使用独立来源档案；同视频截帧建立的条目必须标记为`same-encounter enrollment`。

检查实际加载内容：

`python -c "from config import load_config; from database import ShipDatabase; db=ShipDatabase(load_config('config.final.yaml')); print(db._data); print('012=',repr(db.lookup('012'))); print('003=',repr(db.lookup('003'))); print('002=',repr(db.lookup('002')))"`

冻结档案：

`cp data/ships.csv data/ships_archive_frozen.csv`

测试期间禁止使用`--allow-memory-write`，禁止根据测试结果修改档案。

## 5. 正式配置

复制配置：

`cp config.yaml config.final.yaml`

### 5.1 实体重关联

```yaml
experiment:
  entity_reconciliation:
    enabled: true
    max_gap_frames: 180
    min_gap_frames: 1
    min_appearance_similarity: 0.78
    max_center_distance: 0.35
    max_log_area_ratio: 1.20
    min_association_score: 0.72
    appearance_weight: 0.70
    position_weight: 0.20
    scale_weight: 0.10
    appearance_ema: 0.80
    max_appearance_prototypes: 6
    appearance_novelty_threshold: 0.90
```

- `max_gap_frames`控制轨迹消失后允许重关联的最大帧间隔。
- `min_appearance_similarity`是HSV外观描述符最低余弦相似度。
- `max_center_distance`约束重新出现位置。
- `max_log_area_ratio`约束目标尺度变化。
- `min_association_score`是外观、位置和尺度的融合阈值。
- `appearance_ema`控制当前外观模板更新速度。
- `max_appearance_prototypes`控制每个实体保留的多视角外观原型数量。
- `appearance_novelty_threshold`控制何时把新的转向视角加入原型库。候选重关联使用原型库中的最大外观相似度，而不是只比较最后一个视角。

这些参数沿用当前代码版本中已经确定的值，并在最终测试前冻结。最终测试后不得根据Pair F1回调参数。当前实现会对同一帧可见的不同`track_id`施加互斥约束，避免因检测遍历顺序把同时存在的两艘船错误合并。

### 5.2 档案身份判定

```yaml
experiment:
  archive:
    enabled: true
    structure_in_archive_threshold: 0.735
    structure_uncertain_threshold: 0.70
    structure_min_margin: 0.00
    min_structure_observations: 1
    strong_single_threshold: 0.88
    consistent_in_archive_threshold: 0.73
    min_consistent_observations: 2
    min_rejection_observations: 2
    min_consistency_ratio: 0.67
    preserve_verified_identity: true
    conflict_observations_to_downgrade: 2
```

`structure_min_margin`及其他身份阈值沿用最终测试前已经确定的配置，不再执行测试集网格搜索。身份状态机采用三段证据规则：单帧只有达到`strong_single_threshold`才确认；中等分数必须达到`consistent_in_archive_threshold`并具有至少`min_consistent_observations`次一致证据；低于不确定阈值的结果必须累计至少`min_rejection_observations`次才判定不在库。新重关联轨迹视为新视角，即使图像质量未明显提高，也允许在查询预算内主动复核。同一轨迹的中等分数证据如果长时间没有获得第二次确认，则在`confirmation_retry_gap_frames`后允许一次受查询预算约束的确认性复核。

### 5.3 主动策略

```yaml
experiment:
  policy:
    uncertainty_threshold: 0.65
    min_gap_frames: 45
    fixed_interval_frames: 150
    gray_zone_reobserve: true
    max_queries_per_track: 3
    require_quality_improvement: true
    min_quality_improvement: 0.03
    min_confirmation_observations: 2
    confirmation_retry_gap_frames: 120
```

实体重关联启用后，`recognition_attempts`和证据历史会继承到新轨迹片段，因此查询预算由同一实体共享。

### 5.4 批量实验关闭显示

```yaml
pipeline:
  demo: false
  save_output_video: false
```

批量实验不要添加`--display`和`--save-output-video`。

## 6. 生成最终测试轨迹标注

基础推理必须关闭实体重关联，使每个ByteTrack轨迹都被记录：

`python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final.yaml --split test --run-name bootstrap_test --memory-mode none --policy-mode recognize_once --network-profile real --entity-reconciliation off`

生成测试集草稿：

`python -m experiments.bootstrap_annotations --run-dir experiment_outputs/bootstrap_test --manifest data/annotations/manifest.jsonl --output data/annotations/test_tracks_draft.jsonl --csv data/annotations/test_tracks_draft.csv --crop-dir data/annotations/test_representative_crops`

人工确认后保存为`data/annotations/test_tracks.jsonl`。

每条有效轨迹必须确认：`vessel_id`、`known_or_unknown`、已知船`hull_number`、`hull_visible`、`risk_label`和`annotation_status`。

短轨迹不要因为短而直接删除。如果它确实属于完整物理船，应保留用于碎片重关联实验；如果只是设备、局部船体、误检或发生身份切换，则删除。标准必须对已知和未知目标一致。

`python -m experiments.validate_annotations data/annotations/test_tracks.jsonl`

## 7. 构造实体真值

生成测试实体：

`python -m experiments.build_entity_annotations --tracks data/annotations/test_tracks.jsonl --output data/annotations/test_entities.jsonl`

脚本按`(video_id,vessel_id)`聚合。必须人工检查：同一物理船所有track是否在同一实体、同一视频中的不同船是否分开、中白小白小蓝保持`unknown`且编号为空、大白统一为`012`、小黄统一为`003`。

检查测试实体：

`python -c "import json,collections; p='data/annotations/test_entities.jsonl'; rows=[json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]; print('entities=',len(rows)); print(collections.Counter((r['known_or_unknown'],r['hull_number']) for r in rows)); [print(r['video_id'],r['vessel_id'],r['member_track_ids']) for r in rows]"`

## 8. 无验证集的配置与档案冻结

在任何正式测试推理之前冻结配置：

`cp config.yaml experiment_outputs/frozen_config.yaml`

完成`ships.csv`和视觉原型后执行最终预检与输入冻结：

`python -m experiments.prepare_final_dataset --manifest data/annotations/manifest.jsonl --output-dir data/annotations/final_protocol --archive-splits train,archive --test-split test --probe-videos --config experiment_outputs/frozen_config.yaml --archive-csv data/ships.csv --visual-archive-root data/archive/visual_prototypes`

保存`data/annotations/final_protocol/protocol_lock.json`。正式实验结果必须对应其中的manifest、配置、文本档案和视觉档案哈希。测试结果产生后，任何参数或档案变化都必须作为新的方法版本重新声明，不能覆盖原结果。

## 9. 正式主实验

主实验四种方法均启用实体重关联，使比较聚焦观测和记忆策略。

| 方法 | 论文解释 |
|---|---|
| `main_single` | 每实体一次观察，无时间融合 |
| `main_fixed` | 每实体固定间隔重复观察 |
| `main_active_tel` | 实体TEL加不确定性主动观察 |
| `main_full` | 实体TEL、主动策略及其他技能信号 |

运行：

`python -m experiments.run_paper_suite --manifest data/annotations/manifest.jsonl --config experiment_outputs/frozen_config.yaml --split test --group main --execute`

汇总轨迹和实体指标：

`python -m experiments.evaluate_paper_suite --root experiment_outputs --annotations data/annotations/test_tracks.jsonl --entity-annotations data/annotations/test_entities.jsonl --group main --output experiment_outputs/paper_suite_summary.csv`

论文主表优先报告`entity_archive_matching_precision`、`entity_archive_matching_success_rate`、`entity_unknown_false_acceptance_rate`、`entity_unknown_rejection_precision`、`entity_unknown_rejection_recall`、`entity_unknown_rejection_f1`、`entity_unresolved_unknown_rate`、`avg_vlm_calls_per_entity`和`avg_upload_bytes_per_entity`。 实体评价会聚合同一真值船覆盖的全部预测碎片：调用和上传量求和，未知船任一碎片被接收入库即计为误接受，避免重关联失败时只评价最大碎片而低估错误和成本。

轨迹级结果作为端到端诊断，不得把87条轨迹描述为87艘独立船。

## 10. 实体重关联核心实验

开启实体重关联：

`python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config experiment_outputs/frozen_config.yaml --split test --run-name entity_on --memory-mode full --policy-mode full --network-profile real --entity-reconciliation on --experience off --risk off`

关闭实体重关联：

`python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config experiment_outputs/frozen_config.yaml --split test --run-name entity_off --memory-mode full --policy-mode full --network-profile real --entity-reconciliation off --experience off --risk off`

评价：

`python -m experiments.evaluate_entities --run-dir experiment_outputs/entity_on --annotations data/annotations/test_entities.jsonl`

`python -m experiments.evaluate_entities --run-dir experiment_outputs/entity_off --annotations data/annotations/test_entities.jsonl`

必须同时报告Pair Precision、Pair Recall和Pair F1。只报告身份准确率无法判断增益来自正确重关联还是错误合并。

## 11. 消融实验

运行：

`python -m experiments.run_paper_suite --manifest data/annotations/manifest.jsonl --config experiment_outputs/frozen_config.yaml --split test --group ablation --execute`

评价：

`python -m experiments.evaluate_paper_suite --root experiment_outputs --annotations data/annotations/test_tracks.jsonl --entity-annotations data/annotations/test_entities.jsonl --group ablation --output experiment_outputs/ablation_summary.csv`

重点解释：`ablation_no_entity`验证跨轨迹实体记忆，`ablation_no_tel`验证跨帧证据融合，`ablation_fixed_policy`验证主动策略相对固定间隔的效率，`ablation_no_archive`验证档案核验技能，`ablation_no_quality`验证观测质量门控。

`ablation_no_experience`只在经验数据库真实存在时解释，`ablation_no_risk`只解释风险动作。没有增益的模块应如实写入限制，不得为了得到正结果反复调整测试集。

## 12. 主动观测实验

主动策略的论文问题是：在获得相近身份性能时，是否比固定间隔策略使用更少的VLM调用。

联合报告Known Entity Accuracy、Unknown FAR、Unknown Rejection Recall、Unresolved Entity Rate、VLM Calls per Entity、Upload Bytes per Entity和Confirmation Frames。

如果主动方法只提高解决率但增加未知误接受，只能表述为提高决策覆盖率，不能宣称身份准确率提高。

## 13. Jetson Orin端侧部署实验

端侧实验用于证明可部署性，不包装为端云协同算法。

### 13.1 必报指标

- 平均FPS。
- Real-Time Factor，处理FPS除以源视频FPS。
- YOLO平均和P95延迟。
- Entity Reconciliation平均和P95延迟。
- VLM平均和P95响应时间，作为外部服务系统指标。
- Process CPU Percent。
- Peak RSS。
- GPU平均和P95利用率。
- RAM平均和峰值。
- 平均和峰值功耗。
- Energy per Video或Energy per Frame。
- Orin上的实体身份指标。

### 13.2 预热

`python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config experiment_outputs/frozen_config.yaml --split train --run-name orin_warmup --memory-mode full --policy-mode full --network-profile real --entity-reconciliation on --max-frames 300`

### 13.3 正式采集

`mkdir -p experiment_outputs/orin_measurement`

`sudo tegrastats --interval 1000 --logfile experiment_outputs/orin_measurement/tegrastats.log &`

`/usr/bin/time -v -o experiment_outputs/orin_measurement/time.txt python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config experiment_outputs/frozen_config.yaml --split test --run-name orin_full --memory-mode full --policy-mode full --network-profile real --entity-reconciliation on --experience off --risk off`

`sudo tegrastats --stop`

`python -m experiments.summarize_tegrastats --log experiment_outputs/orin_measurement/tegrastats.log --interval-ms 1000 --output experiment_outputs/orin_measurement/tegrastats_summary.json`

`python -c "import json; d=json.load(open('experiment_outputs/orin_full/deployment_summary.json',encoding='utf-8')); print(json.dumps(d,ensure_ascii=False,indent=2))"`

`python -c "import json; d=json.load(open('experiment_outputs/orin_measurement/tegrastats_summary.json',encoding='utf-8')); frames=json.load(open('experiment_outputs/orin_full/deployment_summary.json',encoding='utf-8'))['total_frames']; d['energy_per_frame_j']=round(d['energy_wh']*3600/max(1,frames),6); print(json.dumps(d,ensure_ascii=False,indent=2))"`

`python -m experiments.evaluate --run-dir experiment_outputs/orin_full --annotations data/annotations/test_tracks.jsonl`

`python -m experiments.evaluate_entities --run-dir experiment_outputs/orin_full --annotations data/annotations/test_entities.jsonl`

每个`video_XXXX_summary.json`保存该视频全程阶段延迟。`experiment_outputs/orin_full/deployment_summary.json`汇总整个测试集的`throughput_fps`、`weighted_source_fps`、`realtime_factor`、`process_cpu_percent`、`peak_rss_mb`，以及各阶段的`weighted_avg_ms`、`mean_video_p95_ms`和`max_video_p95_ms`。阶段包含`yolo`、`entity_reconciliation`、`vlm`和`demo`。不要再使用会被最后一个视频覆盖的`summary.json`作为整套测试集部署结果。

至少重复3次端侧实验，报告均值和标准差。身份结果如果VLM具有随机性，也需要重复3次。

端侧实验回答：新增实体记忆后是否仍可实时运行、实体重关联增加多少延迟、主动策略是否降低任务调用成本、边缘部署是否保持身份功能。

## 14. 论文表格和图

### 14.1 数据统计表

报告视频数、物理身份数、已知身份数、未知身份数、tracklet数、entity数、平均每实体tracklet数和最大碎片数。

### 14.2 实体身份主表

列：方法、Known Accuracy、Archive Precision、Unknown FAR、Unknown Rejection Recall、Unresolved Rate、VLM Calls per Entity。

### 14.3 重关联表

列：方法、Pair Precision、Pair Recall、Pair F1、Tracklet Coverage、Predicted Entity Count。

### 14.4 主动策略表

列：方法、身份指标、每实体调用数、上传字节、P50/P95响应时间和确认帧数。

### 14.5 碎片鲁棒性图

按每实体tracklet数量分为1条、2至3条、4至10条和超过10条，绘制身份准确率、未解决率和VLM调用数。

### 14.6 Orin部署表

列：FPS、RTF、YOLO P95、Entity P95、VLM P95、Peak RSS、GPU利用率、平均功耗和能耗。

## 15. 最终检查

- `ships.csv`编号正确且已冻结。
- manifest中没有`val`记录。
- 同一`voyage_id`没有跨档案构建和最终测试。
- 公开视频具有来源链接、许可状态和`dataset_subset`。
- 主协议中的已知测试船具有独立来源档案，未知测试船不在档案中。
- 同视频登记实验与主协议分表报告，登记帧不参与查询评价。
- `vessel_id`只用于评价。
- `test_entities.jsonl`已经人工检查。
- 关联参数、身份阈值和主动策略在正式测试前已经冻结。
- 正式测试后未根据结果修改配置或档案。
- 测试配置已冻结。
- 正式实验未启用长期记忆写入。
- 批量实验未保存视频。
- 主表以实体级指标为主。
- 轨迹级结果未被解释为独立船舶数量。
- 网络模拟未被包装成端云协同贡献。
- Orin报告真实吞吐、延迟、内存和功耗。
- 无风险数据时不强行报告风险定量增益。
- 无经验数据时不强行报告长期经验增益。

## 16. 最小投稿实验集合

1. Entity Reconciliation On与Off。
2. Entity Single、Entity Fixed、Entity Active TEL和Full。
3. `ablation_no_tel`。
4. `ablation_no_archive`。
5. 轨迹级与实体级身份评价。
6. 小蓝多轨迹片段的实体重关联案例。
7. Jetson Orin完整方法的FPS、P95、内存和功耗。

这组实验直接支撑论文叙事：轨迹碎片化会破坏持续身份证据，实体级记忆使Agent跨轨迹维护同一船舶状态，并通过主动观测减少固定重复识别，同时保持边缘设备可部署性。
## 17. 单页案例研究报告

定性实验固定展示三类案例：跨轨迹已知船、开放集未知船，以及持续冲突或持续不确定后待复核的船。完整实验结束后执行：

```bash
python -m experiments.generate_case_studies \
  --run-dir experiment_outputs/main_full \
  --manifest data/annotations/manifest.jsonl \
  --annotations data/annotations/test_entities.jsonl \
  --archive-root data/archive/visual_prototypes
```

如果运行目录中还没有`episodes.jsonl`和`review_records.jsonl`，命令会先根据已有识别、动作、观测、实体、视觉和端云日志自动构建。生成的`case_studies/`目录包含三份自包含HTML；系统存在Chromium、Chrome或Edge时，同时生成三份A4横向单页PDF。实时图、档案参考图和历史帧直接嵌入报告，不显示为文件路径。

论文案例确定后，使用显式选择器冻结案例：

```bash
python -m experiments.generate_case_studies \
  --run-dir experiment_outputs/main_full \
  --manifest data/annotations/manifest.jsonl \
  --annotations data/annotations/test_entities.jsonl \
  --archive-root data/archive/visual_prototypes \
  --known-case V006:E000001 \
  --unknown-case V024:E000007 \
  --conflict-case V010:E000001
```

如果程序无法从`PATH`找到PDF渲染器，增加`--browser /usr/bin/chromium`。只生成可打印的嵌入图片HTML时使用`--no-pdf`。

案例报告只把明确的不兼容身份信息记为真实冲突，包括：不同时序的合格观测支持不同候选、可靠舷号候选相互矛盾、两个合格身份判断支持不同候选，或状态机明确进入`conflicting`。低质量检测框、没有识别记录、普通`unknown/uncertain`状态、查询预算耗尽和单独的`review_requested`均不构成真实冲突。

在冻结第三个论文案例前，先审计所有已有实验目录：

```bash
python -m experiments.audit_conflict_cases --root experiment_outputs --annotations data/annotations/test_entities.jsonl
```

输出只列出带有明确冲突证据的实体，并显示实验目录、真值类别、状态、是否复核、档案分数、视觉分数、帧、轨迹和冲突信号。优先选择`truth=unknown`、`review=true`、具有非零视觉证据且实时框覆盖完整船体的案例。若输出为`true_conflict_candidates=0`，论文不得把普通漏检包装为冲突；第三个案例应保留为`Persistent Uncertainty and Review`，或改为一个具有真实日志支撑的主动延迟查询案例。

当消融运行中没有`out_of_archive`实体时，可只生成冲突报告，避免生成器同时要求已知和开放集未知案例：

```bash
python -m experiments.generate_case_studies --run-dir experiment_outputs/paper_no_open_set --output-dir experiment_outputs/case_preview_conflict --manifest data/annotations/manifest.jsonl --annotations data/annotations/test_entities.jsonl --archive-root data/archive/visual_prototypes --only-case conflict --conflict-case V024:E000007
```
