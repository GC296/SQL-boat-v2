# SQL-boat-v2 最终确定版论文实验方案

本文件是当前最终实验协议。普通 Web 和视频运行保留在线自动建档功能；`experiments.run_experiment` 默认关闭长期记忆写入，因此测试过程中不会修改 `ships.csv` 或 `ship_memory.db`，也不需要每组恢复数据库。

## 一、数据协议

使用三个集合：Archive Construction Set、Development Set 和 Query Test Set。Archive Construction Set 用于建立 `ships.csv` 和训练期长期记忆；Development Set 只用于阈值选择；Query Test Set 同时包含档案内已知船和档案外未知船，用于最终评价。

每条测试轨迹至少标注 `video_id`、`track_id`、`hull_number`、`known_or_unknown`、`hull_visible`、`frame_start` 和 `frame_end`。未知船的 `hull_number` 为空。

## 二、正常运行与实验运行

普通系统运行使用 `config.yaml` 中 `memory_write.profile: true` 和 `memory_write.experience: true`，保留原来的在线档案和经验更新功能。

论文实验统一通过 `experiments.run_experiment` 或 `experiments.run_paper_suite` 启动。实验运行器默认将长期写入强制关闭。只有构建档案时显式增加 `--allow-memory-write` 才允许写入。

档案构建命令：

```bash
python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.yaml --split train --run-name archive_construction --memory-mode full --policy-mode full --network-profile real --quality on --ledger on --archive on --experience on --risk on --allow-memory-write
```

## 三、主实验

主实验不是消融，而是完整方法与三种合理基线的整体比较。

1. `main_single`：单次识别，无时序融合。
2. `main_fixed`：固定间隔重识别，多数投票。
3. `main_active_tel`：不确定性感知主动重识别和 TEL。
4. `main_full`：质量筛选、TEL、档案、经验、风险和完整主动策略。

预览命令：

```bash
python -m experiments.run_paper_suite --manifest data/annotations/manifest.jsonl --config config.yaml --split test --group main
```

正式执行：

```bash
python -m experiments.run_paper_suite --manifest data/annotations/manifest.jsonl --config config.yaml --split test --group main --execute
```

主表报告：舷号识别成功率、档案匹配准确率、档案匹配成功率、已知目标未识别率、错误匹配率、未知船错误接收率、平均 VLM 调用次数、平均单次识别耗时、平均确认帧数、P50 和 P95 时延。

## 四、消融实验

消融全部以 `ablation_full` 为参照，每次只关闭一个组件。

1. `ablation_no_quality`：关闭观测质量。
2. `ablation_no_tel`：关闭跨帧证据融合。
3. `ablation_no_archive`：关闭档案核验。
4. `ablation_no_experience`：关闭经验检索。
5. `ablation_fixed_policy`：把主动策略替换为固定间隔。
6. `ablation_no_risk`：关闭风险状态对完整策略的影响。

正式执行：

```bash
python -m experiments.run_paper_suite --manifest data/annotations/manifest.jsonl --config config.yaml --split test --group ablation --execute
```

消融表重点报告档案匹配成功率、错误匹配率、未知拒识 F1、调用次数、确认帧数和上传量。

## 五、主动重识别增益

主动重识别的整体增益为 `main_active_tel` 减 `main_single`。主动策略相对固定策略的增益为 `main_active_tel` 减 `main_fixed`。完整档案与经验的额外增益为 `main_full` 减 `main_active_tel`。

需要同时报告成功率变化和调用次数变化，不能只报告成功率。

## 六、端云网络实验

固定完整方法，比较 LAN、稳定移动网络、受限网络和临时中断。

```bash
python -m experiments.run_paper_suite --manifest data/annotations/manifest.jsonl --config config.yaml --split test --group network --execute
```

报告平均调用耗时、P50、P95、上传量、调用失败率、匹配成功率和未解决率。另行在 Orin 上报告真实 FPS、功耗、温度和真实网络延迟。

## 七、风险和日志实验

如果论文保留风险解释贡献，必须额外标注低、中、高风险，并报告 Macro F1、High Risk Recall、False Alerts per Hour、首次告警时间、日志完整率、证据覆盖率和人工评分。当前主身份实验不依赖风险标签，但 `main_full` 保留风险状态参与完整策略。

## 八、一键评测

运行完主实验、消融和网络实验后执行：

```bash
python -m experiments.evaluate_paper_suite --root experiment_outputs --annotations data/annotations/tracks.jsonl --group all --output experiment_outputs/paper_suite_summary.csv
```

输出 `paper_suite_summary.csv`、各运行目录的 `metrics.json` 和 `paper_suite_gains.json`。

## 九、最终论文结果结构

主表一是身份与档案匹配主结果。表二是组件消融。表三是主动策略效率。表四是网络和 Orin 部署。风险贡献保留时增加表五风险解释与日志。另需一张困难场景分组图和一张成功、失败案例图。
