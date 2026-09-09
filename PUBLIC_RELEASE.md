# Public source release / 公开源码说明

This repository contains the complete application source, central VLM controller,
legacy pipeline, web interface, training/experiment scripts and tests from the
modified SQL-boat-v2 workspace (2026-09-09).

公开版本包含修改后的完整项目源码。为避免公开运行环境信息，发布副本中的配置凭据
使用占位值，局域网服务地址改为 localhost；请部署后填写自己的服务配置。
源工作目录及原远程仓库未被修改。

## External assets

原始视频、现场图片、船舶档案和标注、模型权重、缓存、运行日志未发布。
data/ships.csv 仅保留原始表头，请填入自己的档案。
DINOv2 第三方源码/权重需从官方项目获取（https://github.com/facebookresearch/dinov2），
YOLO 权重和 VLM 服务按 README 配置。它们不是本项目自编源码的一部分。

## Central VLM

详见 [修复及启动说明](docs/central_agent_repair_20260909.md)。
本地验证：207 项软件测试通过；真实视频识别效果尚未在此次修复后验证。

## Provenance

Source workspace originally referenced https://github.com/hyshhh/SQL-boat-v2.
This is a snapshot of the locally modified code, including previously uncommitted
work. Existing authorship notices are retained. No new license grant is added;
public visibility alone does not change applicable third-party rights.
