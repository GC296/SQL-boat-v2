# 任务驱动在线实时海事具身智能体：思路梳理、案例设计与控制接入路线

## 1. 文档目的

本文档基于当前 `SQL-boat-v2` 代码，重新梳理项目从“船舶舷号识别系统”升级为“任务驱动、在线、实时的海事具身智能体”的合理主线，并回答以下问题：

1. 当前工作的核心研究对象究竟是什么；
2. `Skills` 除舷号识别外应该包含什么；
3. 哪些任务适合成为论文和横向项目的重点；
4. 如何通过一到两个 Case Study 证明系统真正落地；
5. 后续如何把云台控制和无人船控制接入 `Action`；
6. “捡垃圾”任务需要增加哪些感知、定位、控制和执行能力；
7. 甲方需要提供哪些硬件、接口、数据、规则和测试条件；
8. 后续研发与论文实验应按什么顺序推进。

本文档只做方案设计，不修改当前代码。

---

## 2. 对当前工作最准确的重新定义

当前项目不应再被描述成一个单纯的舷号 OCR 或视频识别系统。更合理的定义是：

> 一个面向无人船连续视频流的任务驱动海事具身智能体。智能体持续维护周围目标的实体状态，根据具体任务调用可组合视觉与知识技能，在证据不足时主动选择新的观测或核验动作，并在接入船控和云台后通过真实物理动作改变后续观测与任务状态。

这里的关键词必须分别落实：

### 2.1 任务驱动

系统不是对所有目标机械地执行相同流程，而是先接收任务，再决定：

- 关注什么目标；
- 需要哪些证据；
- 调用哪些 Skills；
- 何时停止识别；
- 何时继续跟踪；
- 何时控制云台获取更清晰视图；
- 何时调整船体位置；
- 何时上报、接近、跟随或执行作业。

### 2.2 在线

系统处理持续到达的视频和传感器信息，不能预先访问完整未来视频。智能体必须在当前时刻维护状态并做决定。

### 2.3 实时

实时不等于所有大模型推理都达到视频帧率。合理的实时系统包含不同时间尺度：

- 10–30 Hz：视频采集、检测、跟踪、控制反馈；
- 1–10 Hz：目标状态、风险线索、云台跟踪；
- 0.1–1 Hz：VLM 属性分析、身份核验、任务推理；
- 事件触发：人工上报、任务切换、船体机动。

### 2.4 具身

具身不是“代码部署在无人船上”本身，而是：

> 智能体的动作能够改变传感器视角、船体位置或作业状态，新观测再反馈到智能体状态中。

当前系统已具备在线感知和软件级 Action 决策，但物理具身闭环还未完成。后续接入云台和船控后，才会形成更完整的：

```text
Task → Observe → Understand → Remember → Decide → Act → Observe Again
```

---

## 3. 当前代码能力盘点

### 3.1 已经具备的基础

| 能力 | 当前状态 | 主要代码位置 | 对具身 Agent 的价值 |
|---|---|---|---|
| 离线视频处理 | 已具备 | `pipeline/pipeline.py` | 可复现实验、构造案例、调试策略 |
| 实时摄像头输入 | 已具备 | `pipeline/virtual_camera.py`、`web/routes/pipeline_api.py` | 在线 Agent 的输入基础 |
| 船舶检测 | 已具备 | `pipeline/detector.py` | 形成候选目标 |
| 多目标跟踪 | 已具备 | `pipeline/tracker.py` | 维护短期轨迹 |
| 实体级轨迹重关联 | 已具备 | `pipeline/entity.py` | 将碎片 Track 恢复为视频内实体 |
| 目标裁剪和质量估计 | 已具备 | `pipeline/quality.py` | 筛选值得提交给 VLM 的观测 |
| 舷号与结构描述 | 已具备 | `pipeline/pipeline.py`、`agent/__init__.py` | 身份证据 |
| 船舶档案检索 | 已具备 | `database/`、`pipeline/archive.py` | 已知船身份核验 |
| 跨帧证据记忆 | 已具备 | `pipeline/tracker.py` | 目标状态与证据融合 |
| 主动查询策略 | 已具备，规则式 | `pipeline/policy.py` | 决定是否再次调用 VLM |
| 图像空间风险线索 | 已具备，初级 | `pipeline/risk.py` | 任务优先级和上报依据 |
| 实验日志和指标 | 已具备 | `pipeline/experiment.py`、`experiments/` | 论文评价和行为追踪 |
| Web 实时展示 | 已具备 | `web/` | 甲方演示与操作员交互 |

### 3.2 当前还缺少的关键能力

| 缺口 | 影响 |
|---|---|
| 明确的 Mission Schema | 系统仍主要围绕识别流程运行，任务目标没有成为一等对象 |
| 正式 Skill Registry | Skills 主要散落在 Pipeline 内，难以按任务编排和扩展 |
| 统一 Action Schema | 当前 Action 是字符串决策，不能直接驱动硬件 |
| Action Executor | 没有把高层动作转换成云台、航控或作业机构命令 |
| Safety Supervisor | 没有对模型动作做权限、边界、碰撞和失效保护 |
| 船体状态输入 | 缺少 GPS、IMU、航向、速度、控制模式和电量等统一状态 |
| 云台状态输入 | 缺少 pan、tilt、zoom、限位和控制反馈 |
| 坐标变换 | 无法将图像目标转换为方位角、地理目标或航控参考 |
| 任务执行反馈 | 无法确认物理 Action 是否已执行、是否达到目标 |
| 垃圾检测与定位 | 当前检测类别主要是船舶，无法直接支持垃圾任务 |
| 回收执行机构接口 | 没有打捞装置、机械臂或传送机构控制 |

因此，当前系统已经是一个较完整的在线视觉认知 Agent 原型，但还不是完整的物理行动 Agent。

---

## 4. 推荐的总体 Agent 架构

建议后续把系统逻辑明确为七层，而不是继续在 `pipeline.py` 中增加条件分支。

### 4.1 Mission Layer

负责描述“这次出航要完成什么”。

建议任务结构：

```yaml
mission_id: patrol_001
mission_type: vessel_watchkeeping
area_of_interest: harbor_zone_A
priority: high
target_constraints:
  known_or_unknown: any
  watchlist_ids: ["012", "003"]
success_conditions:
  - all_targets_logged
  - watchlist_targets_verified
termination_conditions:
  - operator_stop
  - low_battery
  - mission_timeout
```

### 4.2 World and Entity State

维护：

- 无人船自身状态；
- 云台状态；
- 视频内船舶实体；
- 垃圾或漂浮物实体；
- 航行区域和禁入区；
- 当前任务进度；
- 风险与不确定性；
- 已执行动作及结果。

### 4.3 Skill Registry

每个 Skill 都应具有：

- `name`；
- `required_inputs`；
- `preconditions`；
- `output_schema`；
- `latency_class`；
- `cost_class`；
- `confidence/evidence fields`；
- `failure states`。

### 4.4 Memory and Evidence Layer

继续使用当前实体级证据账本，但扩展为：

- identity evidence；
- task evidence；
- spatial evidence；
- action evidence；
- safety evidence；
- execution feedback。

### 4.5 Policy and Planner

根据任务状态选择下一项 Skill 或 Action，不直接输出底层电机值。

### 4.6 Action Executor

将高层动作转换为：

- 云台命令；
- 航控命令；
- 作业机构命令；
- 人工确认请求。

### 4.7 Safety Supervisor

位于 Policy 和真实硬件之间，拥有否决权：

```text
Agent proposal → Safety validation → Hardware command
```

大模型或高层 Agent 永远不应直接输出 PWM、舵角或推进器转速。

---

## 5. 任务、Skill 与 Action 的层级关系

必须区分以下概念：

| 层级 | 示例 | 时间尺度 |
|---|---|---|
| Mission | 港区巡逻并核验重点船舶 | 分钟到小时 |
| Task | 持续跟踪目标 E01 并确认身份 | 秒到分钟 |
| Skill | 读取舷号、提取属性、估计运动趋势 | 百毫秒到秒 |
| Action | 云台转向、继续观察、跟随目标、上报 | 毫秒到秒 |
| Control Command | 目标航向、目标速度、pan/tilt 角度 | 10–100 ms |

论文和系统中应主要强调 Mission、Task、Skill、Action。底层控制命令属于工程实现，不由 VLM 直接管理。

---

## 6. 建议的任务体系

### 6.1 主任务 A：动态船舶持续瞭望与身份核验

这是最适合当前代码、也最适合作为论文主 Case Study 的任务。

#### 任务目标

- 发现进入视野的船舶；
- 保持视频内实体连续性；
- 判断是否属于关注目标；
- 在证据不足时主动获取更清晰观测；
- 输出身份状态、运动趋势、风险线索和任务日志；
- 对高风险或长期无法确认的目标上报人工。

#### 为什么适合当前项目

当前代码已经具备约 70%–80% 的软件基础：

- 实时视频；
- 船舶检测与跟踪；
- Tracklet 重关联；
- 舷号和结构属性；
- 档案库；
- 实体证据记忆；
- 主动查询；
- 风险线索与日志。

后续只需逐步加入云台 Action 和船体跟随 Action，就能形成完整具身闭环。

### 6.2 主任务 B：重点目标主动取证

任务输入不是“识别所有船”，而是：

```text
寻找并核验档案 ID 012；若舷号不可见，则保持跟踪并主动调整观测。
```

任务成功条件：

- 找到目标；
- 建立持续实体；
- 获得足够身份支持；
- 保存可审计证据；
- 在预算内确认或上报无法确认原因。

该任务最适合体现 Mission-Oriented 与 Active Perception。

### 6.3 扩展任务 C：漂浮垃圾巡检

初期不要直接定义为“自动捡垃圾”，而应分成：

1. 漂浮物发现；
2. 垃圾类型与可回收性判断；
3. 目标定位和持续跟踪；
4. 接近可行性与安全判断；
5. 人工确认；
6. 自动接近；
7. 回收执行；
8. 成功验证。

建议先完成“发现、定位、跟踪和接近”，再接入打捞机构。

### 6.4 扩展任务 D：区域巡逻与异常发现

- 按航点巡逻；
- 发现区域内新目标；
- 判断是否偏离常规活动；
- 对未知船、漂浮物、人员或障碍建立事件；
- 根据任务优先级决定继续巡逻还是停留核验。

### 6.5 可选任务 E：落水人员搜寻与上报

该任务社会价值高，但安全责任也最高。初期只建议做：

- 人员或救生衣检测；
- 持续跟踪；
- 位置记录；
- 云台保持目标；
- 远程告警。

在没有经过严格验证前，不建议宣传自动救援控制。

---

## 7. Skills 除舷号识别外应该放什么

不建议把所有能力都包装成 VLM Skill。Skills 应包括视觉模型、几何算法、数据库工具和任务规则。

### 7.1 感知类 Skills

#### Vessel Detection Skill

输出：

- bounding box；
- detection confidence；
- coarse category；
- observation timestamp。

当前基础：已具备。

#### Multi-Object Tracking Skill

输出：

- track ID；
- trajectory；
- visibility state；
- track quality。

当前基础：已具备。

#### Entity Reconciliation Skill

输出：

- entity ID；
- member tracklets；
- association score；
- reassociation reason。

当前基础：已具备视频内轻量重关联。

#### Observation Quality Skill

输出：

- scale；
- blur；
- brightness；
- contrast；
- clipping；
- total quality；
-是否值得进行高成本推理。

当前基础：已具备。

#### Floating Object Detection/Segmentation Skill

面向垃圾任务新增：

- object mask；
- debris class；
- estimated size；
- confidence；
- collection suitability。

当前基础：未具备。

### 7.2 目标语义 Skills

#### Vessel Attribute Extraction

建议输出结构化字段：

- vessel type；
- hull color；
- superstructure color；
- approximate size class；
- visible side；
- deck equipment；
- distinctive structures；
- activity state。

当前基础：已有描述，但需要进一步结构化。

#### Hull-Number Recognition

输出：

- text hypothesis；
- normalized hull number；
- visibility；
- character-level uncertainty；
- source crop reference。

当前基础：已具备主要流程。

#### Identity Verification

融合：

- hull number；
- visual structure；
-档案候选；
-多帧一致性；
-可选 AIS/外部数据。

输出：

- confirmed known；
- provisional known；
- unknown；
- conflicting；
- unresolved。

当前基础：已具备。

#### Target Relevance Skill

根据 Mission 判断目标是否值得继续处理，例如：

- 是否为关注档案；
- 是否进入任务区域；
- 是否需要身份确认；
- 是否可以忽略；
- 是否需要提高优先级。

当前基础：缺少正式任务输入。

### 7.3 时空与态势 Skills

#### Motion Trend Skill

基于轨迹和船体状态输出：

- approaching/receding；
- left/right crossing；
- image-plane speed；
- relative bearing trend；
- track persistence。

当前 `risk.py` 已有初级图像空间趋势。

#### Encounter Assessment Skill

在只有视频时应输出：

- image-space encounter cues；
- concern level；
- evidence and limitations。

接入 GPS、IMU、距离或 AIS 后再输出：

- relative bearing；
- relative speed；
- CPA/TCPA；
- encounter type；
- COLREG-related context。

#### Target Geolocation Skill

将图像目标转成：

- camera bearing；
- relative direction；
- estimated range；
- world coordinate or latitude/longitude。

这是船控和捡垃圾任务的关键新增 Skill。

### 7.4 记忆与证据 Skills

#### Evidence Update Skill

将当前观测写入实体状态，包含：

- evidence source；
- timestamp；
- quality；
- hypothesis；
- support/conflict；
- raw crop reference。

#### Archive Retrieval Skill

查询船舶档案和历史图片。

#### Experience Retrieval Skill

查询相似失败场景、有效动作和历史结果。

#### Evidence Sufficiency Skill

判断：

- 当前任务是否已有足够证据；
- 哪一项证据缺失；
- 新观测预期能改善什么；
- 是否应该停止继续调用模型。

### 7.5 任务与决策 Skills

#### Mission Decomposition Skill

把“巡逻并核验重点船”拆成：

```text
patrol → detect → select target → maintain entity → verify identity
→ assess relevance → continue/track/report → mission completion
```

#### Action Selection Skill

选择高层 Action，但不输出底层控制量。

#### Event Logging Skill

输出：

- what；
- when；
- where；
- target；
- evidence；
- decision；
- action；
- outcome。

#### Human Clarification Skill

生成对操作员的明确请求，例如：

- 是否允许跟随目标；
- 是否确认垃圾可回收；
- 是否将未知船加入关注列表；
- 是否允许进入某区域。

### 7.6 Action Skills

未来真实接入后包括：

#### 云台动作

- `point_camera_at_entity`；
- `track_entity_with_gimbal`；
- `set_zoom_level`；
- `scan_sector`；
- `return_to_home_pose`；
- `stop_gimbal`。

#### 船体动作

- `hold_position`；
- `follow_waypoints`；
- `follow_entity_at_offset`；
- `approach_target`；
- `orbit_target`；
- `slow_down`；
- `stop_motion`；
- `return_home`。

#### 作业动作

- `prepare_collector`；
- `deploy_collector`；
- `collect_target`；
- `retract_collector`；
- `verify_collection`。

---

## 8. 最推荐的两个 Case Study

论文和项目演示不应同时平均展示五个任务。建议选择一个主 Case 和一个扩展 Case。

## 8.1 Case Study 1：动态船舶持续瞭望与主动身份核验

这是当前最应该做深的 Case。

### 场景设定

无人船在港区或水域巡航，视野中出现多艘移动船舶。目标可能：

- 距离较远；
- 舷号暂时不可见；
- 被遮挡；
- 离开画面后重新出现；
- 因跟踪中断产生多个 Tracklet；
- 属于档案船或未知船。

### Agent 闭环

```text
1. Detect vessels
2. Build tracklets
3. Reconcile persistent entities
4. Evaluate mission relevance
5. Extract current evidence
6. Check evidence sufficiency
7. If insufficient:
   a. continue observation
   b. select a better buffered frame
   c. command gimbal to center/zoom target
   d. optionally adjust USV observation geometry
8. Verify identity against archive
9. Maintain or revise entity state
10. Report confirmed, unknown, conflicting, or high-priority target
```

### 最能体现亮点的事件

#### 事件 A：舷号由不可见变为可见

- 初始远距离图像质量不足；
- Agent 不强行输出身份；
- 持续跟踪实体；
- 云台将目标居中并适度变焦；
- 新观测进入 Evidence Ledger；
- 舷号与结构证据共同确认身份。

#### 事件 B：目标 Track ID 发生切换

- 同一艘船被跟踪器切成多个 Tracklet；
- Entity Reconciliation 恢复同一实体；
- 新 Track 继承历史身份和动作预算；
- 不重复从零调用 VLM。

#### 事件 C：未知船拒识

- 目标结构与现有档案相似；
- Agent 发现候选支持不足或证据冲突；
- 输出 unknown/unresolved，而不是强行匹配；
- 对任务相关未知船生成上报。

### Case Study 输出

建议 Demo 同时显示：

- 实时视频和实体框；
- `track_id` 与 `entity_id`；
- 当前 Mission；
- 当前身份状态；
- Evidence Timeline；
- 当前不确定性原因；
- 选择的 Skill；
- 选择的 Action；
- 云台或船控反馈；
- 最终任务结果。

### Case Study 指标

- Entity association F1；
- Exact hull-number accuracy；
- Entity archive matching accuracy；
- Unknown false acceptance rate；
- Time to correct confirmation；
- VLM calls per correctly confirmed entity；
- Gimbal centering error；
- Target retention rate；
- Action success rate；
- Real-time FPS；
- End-to-end response latency。

## 8.2 Case Study 2：漂浮垃圾巡检与辅助回收

这是非常适合体现具身性的扩展 Case，但当前距离完整实现较远。

### 推荐分成三级

#### Level 1：垃圾发现与报告

- 检测漂浮物；
- 分类垃圾类型；
- 持续跟踪；
- 记录位置；
- 上报操作员。

#### Level 2：主动接近与检查

- 选择可疑垃圾；
- 控制云台保持目标；
- 无人船低速接近；
- 获取更清晰图像；
- 判断是否值得回收；
- 保持安全距离等待确认。

#### Level 3：自动回收

- 估计目标相对位置；
- 规划末端接近路线；
- 控制船体进入回收窗口；
- 启动作业机构；
- 判断垃圾是否进入收集装置；
- 记录成功或失败。

### 为什么不建议马上把“捡垃圾”作为论文主任务

它需要新增：

- 专用垃圾数据；
- 检测或分割模型；
- 距离估计；
- 目标地理定位；
- 水面漂移预测；
- 船体控制；
- 回收机构；
- 成功检测；
- 安全试验。

如果当前论文重点是海事瞭望 Agent，垃圾回收更适合作为框架泛化和未来完整具身能力的 Case，而不是与船舶身份核验平分贡献。

---

## 9. 论文与项目演示的最佳组合

### 论文主线

> 动态船舶持续瞭望与主动身份核验。

突出：

- 在线视频；
- 实体状态；
- 可组合 Skills；
- 跨帧和跨 Tracklet 记忆；
- 不确定性驱动再观测；
- 云台主动取证；
- 必要时人工上报。

### 横向项目演示

建议两个 Demo：

1. **当前可交付 Demo**：动态船舶跟踪、身份核验、未知船拒识、任务日志；
2. **增强 Demo**：云台自动跟随目标，并在目标过小或偏离中心时执行主动取证。

垃圾任务可以先做：

> 漂浮垃圾发现、持续跟踪、位置记录和接近建议。

等甲方提供控制和回收机构后，再升级为真实回收。

---

## 10. 接入真实 Action 前必须增加的系统结构

当前 `LookoutPolicy` 输出：

- `should_query`；
- `action` 字符串；
- `score`；
- `reasons`。

它适合决定是否调用 VLM，但不足以直接控制硬件。建议增加以下抽象。

### 10.1 Action Request

```json
{
  "action_id": "A000123",
  "mission_id": "patrol_001",
  "entity_id": "E000004",
  "action_type": "track_entity_with_gimbal",
  "parameters": {
    "desired_center": [0.5, 0.5],
    "zoom_policy": "quality_driven",
    "max_duration_s": 20
  },
  "priority": 70,
  "reason": ["target_off_center", "hull_number_unreadable"],
  "deadline": 1720000000.0
}
```

### 10.2 Action Result

```json
{
  "action_id": "A000123",
  "status": "succeeded",
  "started_at": 1720000000.1,
  "finished_at": 1720000003.6,
  "feedback": {
    "final_center_error": 0.03,
    "new_observation_quality": 0.86
  },
  "failure_reason": ""
}
```

### 10.3 Action 状态机

```text
proposed → validated → accepted → executing
→ succeeded / failed / cancelled / timed_out / rejected
```

### 10.4 Safety Decision

每个物理动作在执行前必须经过：

- 权限检查；
- 当前控制模式检查；
- 禁航区检查；
- 最大速度和转向限制；
- 最近障碍物检查；
- 通信健康检查；
- 人工接管状态检查；
- 电量与返航余量检查。

---

## 11. 云台控制接入路线

云台是最适合作为第一个真实具身 Action 的设备，因为风险比船体控制低，且能直接改善视觉证据。

### 11.1 甲方可能提供的接口类型

需要先确认云台属于哪一种：

1. ONVIF PTZ 网络摄像机；
2. MAVLink Gimbal v2；
3. 厂商 HTTP/SDK；
4. Pelco-D/P 串口协议；
5. RS-485/RS-232 自定义协议；
6. 船载工控机已有中间件接口；
7. ROS/ROS 2 Topic、Service 或 Action。

不要在确认硬件前直接选择协议。

### 11.2 建议的统一接口

```python
class GimbalController:
    def get_state(self) -> GimbalState: ...
    def point_to(self, pan_deg, tilt_deg, speed=None): ...
    def set_rate(self, pan_rate, tilt_rate): ...
    def set_zoom(self, zoom_ratio): ...
    def stop(self): ...
    def home(self): ...
```

上层 Agent 只调用统一接口，底层再实现：

- `OnvifGimbalAdapter`；
- `MavlinkGimbalAdapter`；
- `VendorGimbalAdapter`；
- `MockGimbalAdapter`。

### 11.3 云台闭环控制所需信息

#### 相机内参

- 分辨率；
- 焦距；
- 主点；
- 畸变参数；
- 不同 zoom 下的视场角。

#### 相机与云台关系

- 相机光轴；
- pan/tilt 零位；
- 安装偏角；
- 云台限位；
- 控制方向正负号。

#### 云台反馈

- 当前 pan；
- 当前 tilt；
- 当前 zoom；
- 是否运动中；
- 命令是否成功；
- 错误码。

### 11.4 从图像目标到云台命令

目标中心为：


a) 计算归一化偏差：

```text
error_x = target_center_x - image_center_x
error_y = target_center_y - image_center_y
```

b) 根据当前视场角换算角度误差：

```text
pan_error ≈ error_x / image_width × horizontal_FOV

tilt_error ≈ error_y / image_height × vertical_FOV
```

c) 通过 PID 或限速比例控制输出云台速度。

不建议让 VLM 直接输出 pan/tilt 角度。VLM 只提出：

```text
需要保持目标居中并获取更清晰视图
```

几何控制器负责真正的连续控制。

### 11.5 Zoom 策略

Zoom 不能只依据“目标小”盲目放大。建议考虑：

- bbox 占画面比例；
- 当前目标是否接近边缘；
- 跟踪稳定性；
- 舷号可见性；
- 模糊程度；
- 云台运动速度；
- 多目标任务优先级。

简单策略：

```text
目标稳定居中 + bbox 过小 → 增加 zoom
目标接近边缘或运动快 → 减小 zoom
图像模糊或跟踪不稳定 → 暂停 zoom，先稳定云台
```

### 11.6 云台接入测试顺序

1. Mock 云台回放；
2. 单步 pan/tilt/zoom 命令；
3. 限位与停止测试；
4. 静态目标自动居中；
5. 人工移动目标跟踪；
6. 船岸静态环境测试；
7. 船体静止、水面目标运动；
8. 船体运动、目标运动；
9. 与身份核验策略联动；
10. 故障、断连和人工接管测试。

### 11.7 云台实验指标

- target centering error；
- target-in-view ratio；
- target loss count；
- recovery time；
- observation quality gain；
- hull-number readable-frame gain；
- identity verification gain；
- action latency；
- command failure rate；
- unnecessary gimbal action rate。

---

## 12. 船体控制接入路线

船体控制比云台风险高，应在云台闭环稳定后接入。

### 12.1 需要确认的控制架构

甲方船可能采用：

- ArduPilot Rover/Boat + MAVLink；
- PX4 或其他自动驾驶仪；
- ROS/ROS 2 航控节点；
- 工控机自研 UDP/TCP 协议；
- CAN 总线；
- 串口控制器；
- PLC/Modbus；
- 遥控器与自动控制切换模块。

必须先获得现有航控接口，不能绕过已有自动驾驶仪直接控制推进器。

### 12.2 推荐分层

```text
Mission Agent
  ↓ high-level action
Safety Supervisor
  ↓ validated action
Navigation Action Server
  ↓ waypoint / heading / speed reference
Existing Autopilot
  ↓ low-level control
Rudder / Thruster
```

### 12.3 Agent 可调用的高层船控动作

#### Hold Position

用于等待更清晰观测或人工确认。

#### Follow Waypoints

用于区域巡逻。

#### Follow Entity at Offset

保持相对目标一定距离和方位，不直接追到目标后方。

参数：

- desired distance；
- desired bearing offset；
- max speed；
- timeout；
- abort distance。

#### Approach Target

用于垃圾检查或近距离取证。

必须包含：

- maximum approach speed；
- minimum stand-off distance；
- collision abort；
- operator authorization。

#### Orbit/Loiter Around Target

用于从不同侧面获取船体结构或舷号证据。

该动作非常适合具身论文，但需要可靠目标定位和航控。

#### Stop and Return Home

安全动作必须始终可用，且优先级高于 Agent 任务。

### 12.4 目标定位是船控接入的核心难点

仅有 bbox 无法控制船接近目标。至少需要一种距离来源：

- 双目相机；
- 激光雷达；
- 毫米波雷达；
- 单目深度估计加尺度约束；
- AIS/GNSS；
- 已知目标尺寸估距；
- 多帧运动视差；
- 人工或外部定位。

再结合：

- 无人船 GPS；
- IMU；
- 航向；
- 云台角度；
- 相机外参；

才能把目标转换到船体或世界坐标系。

### 12.5 推荐坐标系

至少定义：

- image frame；
- camera optical frame；
- gimbal frame；
- USV body frame；
- local ENU/NED frame；
- world/GPS frame。

所有变换必须带时间戳，避免视频、GPS、IMU 和云台状态不同步。

### 12.6 安全机制

必须由确定性模块实现：

- 地理围栏；
- 最大速度；
- 最大转向率；
- 最小目标距离；
- 静态与动态障碍避碰；
- 通信超时；
- GPS/IMU 异常；
- 低电量返航；
- 遥控优先；
- 紧急停止；
- Action 超时自动取消。

### 12.7 船控测试阶段

1. 软件仿真；
2. Hardware-in-the-loop；
3. 岸上架空推进器测试；
4. 封闭水池低速测试；
5. 空旷水域单动作测试；
6. 人工授权的目标跟随；
7. 云台与船体协调；
8. Mission 级闭环；
9. 故障注入；
10. 甲方验收场景。

### 12.8 船控实验指标

- waypoint success rate；
- path tracking error；
- target standoff error；
- target retention rate；
- time to acquire useful view；
- identity verification improvement after maneuver；
- action completion rate；
- abort rate；
- safety intervention count；
- operator takeover count；
- energy and mission time cost。

---

## 13. 漂浮垃圾回收任务的完整技术链

### 13.1 感知

需要新增数据集和模型，覆盖：

- 塑料瓶；
- 塑料袋；
- 泡沫；
- 木块；
- 水草；
- 渔网；
- 大型漂浮物；
- 非垃圾干扰物。

建议优先做实例分割，因为垃圾形状不规则，bbox 对回收定位不够准确。

### 13.2 分类与任务价值判断

Agent 判断：

- 是否可能为垃圾；
- 是否可以由当前机构回收；
- 是否过大或危险；
- 是否处于禁入区；
- 是否需要人工确认；
- 多个垃圾目标的处理顺序。

### 13.3 定位和漂移预测

需要估计：

- 相对方位；
- 距离；
- 相对速度；
- 水流和风造成的漂移；
- 预计接触位置。

### 13.4 接近规划

考虑：

- 船体惯性；
- 最小转弯半径；
- 回收机构所在船侧；
- 风和水流方向；
- 目标漂移；
- 避障；
- 接近速度。

### 13.5 回收机构

甲方必须明确提供：

- 回收机构类型；
- 工作区域；
- 最大目标尺寸；
- 允许接触速度；
- 开合或启停协议；
- 状态反馈；
- 堵塞检测；
- 紧急停止；
- 回收容量。

### 13.6 成功验证

不能只用“目标消失”判断回收成功。建议至少使用：

- 回收仓传感器；
- 机构电流变化；
- 近距离相机；
- 前后图像对比；
- 操作员确认。

### 13.7 垃圾 Case 指标

- debris detection AP/recall；
- tracking success rate；
- geolocation error；
- approach success rate；
- collection success rate；
- false collection attempt rate；
- time per collected object；
- energy per object；
- human intervention rate；
- safety abort rate。

---

## 14. 甲方目前必须提供的材料

这一部分是后续工作能否推进的关键。建议形成正式《接口与试验条件需求清单》，由甲方逐项确认。

## 14.1 无人船总体资料

甲方需要提供：

- 船体型号、尺寸、重量；
- 推进器和舵机布局；
- 最大速度、低速稳定性、最小转弯半径；
- 动力和续航；
- 工控机、Jetson、路由器和航控拓扑；
- 系统供电规格；
- 船载网络拓扑和 IP 分配；
- 可安装传感器和计算设备的位置；
- 防水、防盐雾和散热限制。

## 14.2 航控与船控接口

必须提供：

- 自动驾驶仪型号和固件版本；
- 控制协议文档；
- SDK 或示例代码；
- 通信方式：串口、UDP、TCP、CAN、ROS 等；
- 当前支持的控制模式；
- waypoint、航向、速度、停止、返航接口；
- 命令 ACK 和状态反馈格式；
- GPS、IMU、航向、速度、控制模式和电量数据接口；
- 遥控/自动切换机制；
- 紧急停止机制；
- 地理围栏和避障模块情况；
- 仿真器或 Hardware-in-the-loop 条件。

最重要的问题不是“能不能发控制命令”，而是：

> 是否存在经过甲方验证的高层控制接口，允许本系统安全地下发 waypoint、目标航向或目标速度。

## 14.3 云台和摄像机资料

需要提供：

- 云台和摄像机型号；
- 控制协议或 SDK；
- 是否支持绝对角、速度控制、预置位和自动跟踪；
- pan/tilt 范围、速度、精度和限位；
- zoom 范围和不同 zoom 下视场角；
- 控制和视频延迟；
- 云台状态反馈；
- 视频流协议和编码；
- 相机内参、畸变参数；
- 相机、云台和船体安装外参；
- 是否有时间同步；
- 夜视、红外、自动曝光和防抖能力。

如果没有标定参数，甲方需要允许项目组进行现场标定。

## 14.4 传感器资料

需要明确是否具备：

- GPS/GNSS；
- IMU；
- 磁罗盘；
- AIS；
- 雷达；
- 激光雷达；
- 双目或深度相机；
- 测距仪；
- 超声波；
- 风速和水流信息。

每个传感器需要：

- 型号；
- 更新频率；
- 精度；
- 坐标系；
- 时间戳；
- 数据接口；
- 安装位置；
- 可否记录原始数据。

## 14.5 任务需求

甲方必须明确业务任务，而不仅是说“做一个智能 Agent”。需要书面确认：

- 主要应用水域；
- 首要任务排名；
- 需要识别的目标类型；
- 是否有重点船舶名单；
- 未知船需要如何处理；
- 允许跟踪多长时间；
- 是否允许船体主动接近；
- 是否允许云台自主转动；
- 哪些动作必须人工确认；
- 风险等级定义；
- 上报对象和格式；
- 任务完成判据；
- 可接受误报和漏报水平；
- 实时性要求；
- 网络不可用时的要求。

## 14.6 数据与标注

甲方需要提供或协助采集：

- 不同天气和时段的视频；
- 不同距离和侧面的船舶视频；
- 同一艘船多次出现的视频；
- 舷号真值；
- 船舶档案；
- 未知船视频；
- 多船交互视频；
- 遮挡和轨迹中断视频；
- 摄像机与航行状态同步日志；
- 云台角度日志；
- 船控命令与执行反馈；
- 垃圾任务数据和类别定义。

甲方还应指定能够确认船舶身份和任务事件的业务专家。

## 14.7 安全和权限

必须确认：

- 自动控制测试由谁审批；
- 哪些水域允许测试；
- 是否有保险和安全员；
- 紧急停止责任人；
- 遥控接管人员；
- 最大允许速度；
- 最小安全距离；
- 禁航区域；
- 人员和其他船舶靠近时的规则；
- 数据保密和视频使用权限；
- 是否允许论文发表图片和实验结果。

## 14.8 回收机构资料

如果要做捡垃圾，必须额外提供：

- 回收机构设计图；
- 控制接口；
- 执行器反馈；
- 允许回收的垃圾范围；
- 机构与相机相对位置；
- 最佳接近方向；
- 允许速度；
- 堵塞和失败处理；
- 清仓方式；
- 现场测试垃圾样本。

## 14.9 甲方配合人员

至少需要：

- 一名船控/航控工程师；
- 一名云台/视频工程师；
- 一名业务专家；
- 一名现场驾驶或安全员；
- 一名网络/服务器联系人；
- 一名项目验收负责人。

---

## 15. 建议甲方先回答的十个决定性问题

1. 船的自动驾驶仪是什么，是否支持高层航点、航向和速度控制？
2. 能否提供控制协议、SDK、仿真环境和示例程序？
3. 云台型号是什么，能否程序控制 pan、tilt 和 zoom？
4. 摄像机和船体是否完成标定，能否获取云台角度和船体姿态？
5. 是否有 GPS、IMU、AIS、雷达或测距传感器？
6. 哪些物理动作允许自动执行，哪些必须人工授权？
7. 甲方最希望展示的一个核心任务是什么？
8. 甲方对实时性、识别率、控制精度和安全距离的验收指标是什么？
9. 是否已有垃圾回收机构，如果有，控制和反馈接口是什么？
10. 能否提供封闭水域、测试船、目标船、操作员和连续试验时间？

只要这十个问题中有一半没有答案，就不应立即进入真实船控开发。

---

## 16. 推荐研发阶段

### Phase 0：任务冻结与接口确认

交付：

- Mission 清单；
- Skill/Action 清单；
- 甲方接口表；
- 安全规则；
- 验收指标。

### Phase 1：软件 Agent 闭环整理

目标：

- 增加 Mission 对象；
- 统一 Skill Schema；
- 统一 Action Request/Result；
- 建立 Action 状态机；
- 将当前 Policy 与任务状态连接；
- 不接真实硬件。

### Phase 2：云台具身闭环

目标：

- 云台 Adapter；
- 目标居中；
- 自动 zoom；
- 云台反馈；
- 身份证据改善实验。

这是最优先的真实 Action。

### Phase 3：船控只读接入

先读取：

- GPS；
- IMU；
- 航向；
- 速度；
- 电量；
- 控制模式。

暂不发送运动命令。

### Phase 4：受限船控 Action

只开放：

- hold；
- stop；
- waypoint；
- return home。

必须经过 Safety Supervisor 和人工授权。

### Phase 5：目标跟随与主动取证

增加：

- follow entity；
- maintain offset；
- orbit/loiter；
- view-quality-driven maneuver。

### Phase 6：垃圾巡检

增加垃圾检测、定位和上报。

### Phase 7：垃圾回收

接入回收机构，完成接近、执行、反馈和成功验证。

---

## 17. 论文包装建议

### 17.1 当前最合理的核心故事

> Existing maritime perception systems passively analyze available frames. The proposed agent maintains task-conditioned entity states in an online video stream and actively selects perception, reasoning, and reporting actions. Physical camera and vessel actions can further close the loop by changing future observations.

中文：

> 传统海事视觉系统被动分析已有帧，而本工作在在线视频流中维护任务条件化的船舶实体状态，并主动选择感知、推理和上报动作；云台和船体 Action 的接入进一步通过改变未来观测形成物理闭环。

### 17.2 当前论文不要过度承诺

在云台和船控尚未接入前，使用：

- online agent；
- real-time watchkeeping；
- active evidence acquisition；
- platform-grounded agent；
- embodied deployment context。

谨慎使用：

- full embodied autonomy；
- autonomous navigation decision；
- physical active perception；
- closed-loop vessel control。

接入云台后，可以更有力地使用：

- embodied active perception；
- action-conditioned observation；
- closed-loop visual evidence acquisition。

接入船体后，可以使用：

- task-driven embodied maritime agent；
- perception-action closed loop；
- active viewpoint acquisition through USV maneuvering。

### 17.3 最适合论文的 Case

主 Case：

> Dynamic vessel watchkeeping with active identity verification.

辅助 Case：

> Floating-debris inspection and approach as a task-transfer demonstration.

垃圾自动回收应在硬件闭环完成后再作为完整 Case。

---

## 18. 后续实验建议

### 18.1 在线性能

- input FPS；
- detection/tracking FPS；
- target-state update latency；
- VLM request latency；
- action-decision latency；
- dropped-frame rate；
- queue depth；
- resource usage。

### 18.2 Agent 行为

- mission completion rate；
- correct action rate；
- unnecessary action rate；
- action success rate；
- evidence gain per action；
- time to mission completion；
- human intervention rate。

### 18.3 船舶 Case

- entity retention；
- identity accuracy；
- unknown rejection；
- target-in-view ratio；
- time to clear observation；
- VLM calls/entity；
- gimbal/boat action benefit。

### 18.4 垃圾 Case

- detection and segmentation；
- tracking；
- localization；
- approach；
- collection；
- safety abort；
- operator intervention。

---

## 19. 推荐使用的控制标准与接口参考

以下只是接口选型参考，最终必须以甲方硬件为准。

- MAVLink Command Protocol：用于命令确认和长时动作语义；
  - https://mavlink.io/en/services/command.html
- MAVLink Gimbal Protocol v2：用于云台管理与设备控制；
  - https://mavlink.io/en/services/gimbal_v2.html
- ArduPilot Rover Guided Mode：若甲方使用 ArduPilot，可通过 Guided 模式接收位置、速度和航向目标；
  - https://ardupilot.org/rover/docs/guided-mode.html
- ROS 2 Actions：适合表示需要持续反馈、可取消、具有最终结果的长时动作；
  - https://docs.ros.org/en/rolling/Concepts/Basic/About-Actions.html
- ONVIF PTZ：若云台为网络 PTZ 摄像机，应向甲方确认是否支持 ONVIF PTZ Profile/Service；
  - https://www.onvif.org/profiles/

推荐原则：

- 短时状态使用 Topic/Telemetry；
- 查询使用 Service；
- 跟随、接近、扫描等长时任务使用 Action；
- 所有真实控制命令必须有 ACK、超时、取消和状态反馈。

---

## 20. 最终建议

### 20.1 研究重点

当前最值得投入的是：

> 动态船舶实体的在线持续瞭望、任务相关 Skill 编排和主动取证。

### 20.2 第一个物理 Action

优先接云台，而不是先接船控：

- 风险较低；
- 与视觉任务直接相关；
- 容易量化证据增益；
- 最能自然强化论文的 Active Perception；
- 甲方演示效果直观。

### 20.3 第二个物理 Action

完成云台后接入受限船控：

- hold；
- stop；
- waypoint；
- target follow at safe offset。

### 20.4 第二个 Case

垃圾任务先做“巡检、跟踪、定位、接近建议”，不要一步跳到全自动回收。

### 20.5 近期最关键的甲方输入

近期应立即向甲方索要：

1. 航控型号与接口；
2. 云台型号与接口；
3. GPS/IMU/AIS/测距能力；
4. 相机与船体标定信息；
5. 自动动作权限与安全规则；
6. 最优先业务任务和验收指标；
7. 可用测试水域与现场配合人员；
8. 垃圾回收机构是否真实存在。

在这些信息明确前，代码层最合理的下一步不是直接写具体船控协议，而是先设计 Mission、Skill、Action、Safety 和 Feedback 的统一接口。
