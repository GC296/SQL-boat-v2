# 中央 VLM 船舶智能体：代码修复与运行说明

日期：2026-09-09

## 1. 本次交付是什么

已直接修改现有 SQL-boat-v2 代码中的中央 VLM 分支，保留 legacy 分支。本次重点修复证据进入模型、模型使用工具后的反馈、预算结算、状态提交及并发一致性；没有把中央决策改写成固定的 SearchArchive → ReadArchive → VerifyEvidence 顺序，也没有新增根据 DINO 分数自动判定身份的规则。

这些修改解决的是实现层面的障碍。自动化测试使用模拟模型回复，证明程序按协议执行、传递和保存信息；不能据此宣称真实 VLM 已经具备稳定的证据判断能力，也不能把测试结果写成论文的识别效果。

## 2. 现在的决策闭环

1. 检测与跟踪产生实体及其真实来源帧的裁剪，实体关联仍使用既有实现。
2. 每个实体维护独立 episode，包含观测、证据、claims、候选评估、待解决问题、动作历史、预算及最终记录。
3. 控制器构造一致的状态快照，提供当前图、被请求的历史图、档案参考图和逐张对应的 image manifest。
4. VLM 输出工具调用、等待或最终结论，也可提交带来源的 belief_patch 和 assessment。
5. 工具结果写入实体证据记忆。工具返回后立即进入下一轮控制器决策，无须等待下一采样时刻。
6. 等待表示结束本次推理链；新观测进入后再次调度。忙碌实体的新帧保留最新待处理状态，不为每帧堆积一个后台任务。
7. VLM 输出 known、out_of_archive 或 review。程序校验协议、引用归属和档案 ID 是否存在，并提交到实体及关联轨迹。预算/执行异常造成的 review 单独标记为 system_fallback。

终止校验不评估证据是否足以证明身份：例如相似度低并不会覆盖 VLM 的 known。但是 known/OOA 仍须引用本 episode 中未撤回的成功证据，known 须提供真实档案 ID。这是防止空引用和虚构对象，不是额外的身份分类器。

## 3. 核心修复

### 3.1 模型真正接收什么

- 用 JSON 状态替代 Python 字典字符串，显式给出 evidence_count、成功检索次数、候选、工具历史、剩余预算及最近校验反馈。
- ReadHistory 按真实 view_ids 过滤，被请求的旧图获得输入优先级；不再只返回文字记录却继续发送最新图。
- ReadArchive 返回档案 ID、档案真实舷号、描述、结构属性及参考图元数据；参考图像实际加入下一次多模态输入。
- 图片编码失败时只移除该图片，manifest、图片索引与实际输入保持一致，并记录失败原因。
- 推理期间固定正在使用的观测，避免新帧淘汰导致模型返回的 view_id 无法执行。
- 模型请求使用本次 pipeline 的 llm 配置，避免实验指定模型却实际调用全局默认模型。

### 3.2 证据与 belief 的一致性

- 新增 claim 可以在同一次 belief_patch 中被候选评估引用。
- patch 按事务执行：其中一项非法，则回滚该组修改，避免部分成功导致难以解释的状态。
- 模型不能通过 source_role 冒充工具来源，也不能直接覆盖原始工具 claim。
- 撤回证据后重新计算视觉候选，避免留下已失效的 candidate/score；恢复来源时恢复其派生 claim 的状态。
- ReadHistory 的持久记忆只保存引用和摘要，避免把历史中的历史反复嵌套。
- 档案编号 archive_id、档案舷号 archived_hull_number、现场读数 observed_hull_number 分开表达。
- 请求读取舷号时，底层独立读取任务不传入上层问题里的预期号码。看不清应返回不确定读数，而不是自动认定与档案冲突。

### 3.3 工具与预算

| 工具 | 执行内容 | 本实现计费 |
|---|---|---|
| SearchArchive | 对指定实体视图执行已配置的视觉档案检索 | 1 次查询 |
| ReadArchive | 读取候选文本档案、结构信息及参考图 | 0 |
| ReadHistory | 读取本实体已观察视图和证据 | 0 |
| VerifyEvidence | 对指定视图执行语义观察或独立舷号读取 | 1 次查询 |

上述计费是资源代理量，不等于 GPU 时间、token 或货币成本。中央 VLM 调用另外受控制调用次数约束；真实返回 usage 和 HTTP 尝试次数进入诊断日志。

预算用完后，拒绝的是新增付费工具执行，不是已有证据。拒绝原因反馈给 VLM，仍允许读取免费工具或直接 finish。单次链路和整个 episode 都有调用上限，避免用免费工具或新帧绕过预算。最终结算阶段仅接受 finish；若仍无法形成可提交输出，记录系统兜底 review。

工具能力清单标注 unavailable。未配置视觉服务时 SearchArchive 不再被表现成可正常检索的服务。非法 top_k、未知参数等作为协议错误返回，而不是崩溃整个任务。

### 3.4 身份提交、并发与记忆

- 中央终态写入 recognized、pending、episode 状态，不再把真实身份结论留在界面“未识别”状态。
- 不把模型确认伪装成置信度 1 或不确定性 0。
- 完成轨迹仍能接收本实体的结果；跨实体结果被拒绝。新轨迹可以继承已完成轨迹状态。
- 中央 review 可以撤销投影中的已确认身份；legacy 的原有送审保护仍保留。
- EOF 不再调用 legacy visual-only gate 覆盖中央结论。
- 拒绝提交不会被当作成功关闭；兜底也提交失败时记录 failed。
- 后台任务使用所有权 token，避免旧任务清理新任务；每实体只保留最新待处理请求，限制实际 worker 数。
- 每实体保留有限裁剪；终止后释放裁剪像素，保留证据和观测元数据；工具缓存及请求锁有上限。

## 4. 配置及运行

现有 config.yaml 仍保留原来的视觉服务开关和后端选择，避免静默改变已有实验。但它目前 visual_archive.enabled=false，因此直接使用默认启动命令不能获得 DINO 检索结果。

在项目根目录，用已有 Python/模型运行环境启动：

```powershell
python -m pipeline "你的视频.mp4" --controller-mode central_vlm --visual-backend dinov2 --dinov2-repo "dinov2-main" --dinov2-weights "dinov2_vits14_pretrain.pth" --max-frames 300
```

也可以在配置中设置 experiment.visual_archive.enabled=true、backend=dinov2，并填写 model_repo、weights、archive_root。新增 `--config` 支持指定完整配置文件；它按照现有 load_config 行为与代码默认值合并，不是叠加当前 config.yaml 的补丁。

先用短视频检查模型服务和档案可读性，再扩大视频长度。此次没有更改模型服务地址，也没有替用户安装 torch、YOLO 或部署 VLM。

experiment.central_controller 新增/明确的参数：

| 参数 | 当前 config.yaml 值 | 含义 |
|---|---:|---|
| query_limit | 3 | 每实体付费工具预算 |
| max_control_steps | 12 | 每次推理链的常规控制调用上限 |
| max_episode_steps | 48 | 跨观测累计的常规控制调用上限 |
| settlement_steps | 3 | 上限后的有限终止重试 |
| max_memory_views | 64 | 实体保留裁剪数；在途引用可能造成短暂超出 |
| max_context_views | 4 | 一次实际发送的图片数上限 |
| max_context_evidence | 20 | 常规上下文证据条数上限 |
| max_context_chars | 32000 | 快照裁剪目标，整条省略而非破坏 JSON |
| max_tokens | 1536 | 控制器输出长度上限 |
| diagnostics_enabled | true | 独立诊断，不依赖 experiment.enabled |

诊断在 output/agent_diagnostics/events.jsonl；包含实际输入上下文、图片对应关系、模型回复、返回 usage、校验拒绝和 episode_closed。图片保存在 images 子目录。日志约 10 MB 轮转，保留一份 previous，图片上限 256；这用于运行诊断，不是永久科研证据仓库。诊断目录写入失败只警告，不中断智能体。

## 5. 验证结果与边界

完整 tests 测试集：207 项通过（最终执行记录另附）。新增 20 个回归用例/参数实例，涵盖预算反馈后自主 known、EOF 兜底来源、非法参数、历史图片输入、参考图输入、编码失败、并发最新观测、独立舷号读取、patch 原子性、完成轨迹提交、跨帧总预算、显式模型配置和诊断故障。

测试运行采用隔离目录中的 OpenCV；没有替换项目原有 Python 环境。没有进行真实 YOLO + DINOv2 + VLM 的完整视频推理。因此尚未验证真实模型是否会停止重复搜索、是否能正确识别舷号、是否合理判定证据充分，也没有新识别指标。

## 6. 仍需区分的未完成事项

1. **真实模型行为验收**：拿曾出现循环的视频检查日志。关键是第一次成功检索之后，下一次实际 request_context 是否已有候选和证据。如果已有而模型仍宣称没有，才能进一步定位模型能力、上下文理解或服务模板问题。
2. **论文评测迁移**：summary 已包含 central_episodes 和 terminal_record，但旧版基于 actions/recognition/visual 的指标脚本并未整体重写。不能把旧 MLP 指标直接当作中央 VLM 新系统指标；尤其要分开 VLM review、系统兜底和提交失败。
3. **长期在线生命周期**：本次限制单实体裁剪和请求缓存，终止释放图片，但没有引入全局实体归档数据库及无限时长运行的实体回收机制。实体记录仍随会话增长。
4. **证据保留与上下文裁剪**：超限会省略整条记录并披露省略数；max_context_chars 是当前快照裁剪目标，不含固定提示词，也不是服务 tokenizer 的严格 token 上限。超长单条字段仍可能超过目标。旧视图被淘汰后不能凭空恢复。
5. **重新开案与轨迹关联**：没有重构 re-ID，也没有实现已关闭实体遇到后续强冲突自动新建复核 episode 的完整策略。
6. **同图变更 top_k**：相同视图与相同参数的成功查询会去重；改变 top_k 仍可能执行。是否把“扩大候选范围”计为有效新查询属于后续策略/实验设计，目前仍由 VLM 决定且受预算约束。

本次可以据此进入真实短视频联调；不要把“207 项软件测试通过”等同于“论文方法效果已验证”。

## 7. 主要变更文件

- pipeline/central_controller.py：上下文、预算反馈、调度、终止记录。
- pipeline/agent_state.py：有界视图、事务 belief、证据修正、终态内存释放。
- pipeline/identity_tools.py：档案参考图、历史视图、参数、舷号独立读取。
- pipeline/controller_schema.py：互斥输出、参数与 assessment 校验。
- pipeline/agent_diagnostics.py：独立有界诊断。
- pipeline/pipeline.py、pipeline/tracker.py：配置传递、帧来源、实体归属和终态投影。
- tools/__init__.py、pipeline/prompts/：真实模型输入、usage、提示词。
- config.py、config.yaml、pipeline/cli.py：资源配置、诊断及显式视觉启动参数。
- tests/test_agent_repair.py：新增回归测试；test_central_controller.py 和 audit_final_design.py 改用结构化 snapshot 取引用。

修改前备份位于本次 Codex 工作目录 work/agent_repair_backup。项目原先已有大量未提交改动，本次没有执行 git reset、清理或提交。
