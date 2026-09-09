> 公开源码发布说明及外部依赖：[PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)
> 中央 VLM 修复与运行：[docs/central_agent_repair_20260909.md](docs/central_agent_repair_20260909.md)

# 🚢 SQL-boat-v2 — 智能船只舷号识别系统

[![Python](https://img.shields.io/badge/Python-≥3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![YOLO](https://img.shields.io/badge/YOLO-v8-FF6F00?logo=ultralytics)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

基于 **LangChain + YOLO + Qwen VLM** 的智能船只舷号识别与管理系统。支持从图片/视频/摄像头实时识别船体舷号，自动入库，并提供完整的 Web 管理界面。

---

## 📑 目录

- [核心功能](#-核心功能)
- [系统架构](#-系统架构)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
  - [环境要求](#1-环境要求)
  - [安装依赖](#2-安装依赖)
  - [配置](#3-配置)
  - [启动服务](#4-启动服务)
- [Web 界面操作指南](#-web-界面操作指南)
  - [界面总览](#界面总览)
  - [Tab 1：数据库管理](#tab-1数据库管理)
    - [查看统计数据](#查看统计数据)
    - [搜索船只](#搜索船只)
    - [新增船只](#新增船只)
    - [编辑船只](#编辑船只)
    - [删除船只](#删除船只)
    - [批量导入](#批量导入)
    - [图片识别入库](#图片识别入库)
  - [Tab 2：视频 Demo](#tab-2视频-demo)
    - [上传视频](#上传视频)
    - [视频列表管理](#视频列表管理)
    - [启动 Pipeline 处理](#启动-pipeline-处理)
    - [查看处理结果](#查看处理结果)
    - [任务历史](#任务历史)
  - [Tab 3：摄像头 Demo](#tab-3摄像头-demo)
    - [配置输入源](#配置输入源)
    - [启动摄像头识别](#启动摄像头识别)
    - [停止与监控](#停止与监控)
- [CLI 命令行使用](#-cli-命令行使用)
- [API 参考](#-api-参考)
- [配置说明](#-配置说明)
- [常见问题](#-常见问题)
- [技术栈](#-技术栈)
- [License](#-license)

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 🖼️ **图片识别** | 上传船只照片，VLM 自动识别舷号和描述，一键入库 |
| 🎬 **视频处理** | 上传视频 → YOLO 检测船只 → VLM 逐帧识别 → 输出标注结果视频 |
| 📷 **摄像头实时识别** | 接入本地摄像头或 RTSP 流，实时检测与识别 |
| 🗄️ **数据库管理** | 支持 CSV / SQLite 双后端，CRUD + 批量导入 + 关键词搜索 |
| 🧠 **语义检索** | 基于 Embedding 向量的语义搜索，描述模糊匹配 |
| 🌐 **Web 界面** | 全功能 Web 管理面板，三个 Tab 覆盖全部操作 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     Web UI (FastAPI + Jinja2)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ 数据库管理 │  │ 视频 Demo │  │ 摄像头 Demo          │   │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘   │
│       │              │                    │               │
├───────┼──────────────┼────────────────────┼───────────────┤
│       ▼              ▼                    ▼               │
│  ┌─────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ShipService│  │Pipeline API │  │Camera API   │          │
│  └────┬─────┘  └──────┬──────┘  └──────┬──────┘          │
│       │               │                │                  │
│  ┌────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐          │
│  │ShipDatabase│ │ YOLO + VLM  │  │ YOLO + VLM  │          │
│  │(CSV/SQLite)│ │ (异步流水线)  │  │ (实时流处理)  │          │
│  └────┬─────┘  └──────┬──────┘  └──────┬──────┘          │
│       │               │                │                  │
│  ┌────▼───────────────▼────────────────▼──────┐           │
│  │         Qwen VLM (视觉语言模型)              │           │
│  │    Qwen3-VL-4B-AWQ via OpenAI-compatible    │           │
│  └────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
SQL-boat-v2/
├── config.py                # 配置加载（唯一配置源）
├── config.yaml              # 全局配置文件
├── pyproject.toml           # 项目元数据与依赖
├── data/
│   └── ships.csv            # 示例船只数据
├── database/
│   ├── __init__.py          # ShipDatabase 核心类（双通道检索）
│   ├── base.py              # 数据源抽象基类
│   ├── csv_source.py        # CSV 数据源实现
│   └── sql_source.py        # SQLite 数据源实现
├── agent/                   # Agent 模块（待扩展）
├── cli/                     # CLI 模块（待扩展）
├── pipeline/                # 视频处理流水线（待扩展）
├── tools/                   # 工具模块（待扩展）
└── web/
    ├── app.py               # FastAPI 应用入口
    ├── models/
    │   └── schemas.py       # Pydantic 请求/响应模型
    ├── routes/
    │   ├── api.py           # REST API（船只 CRUD + VLM 识别）
    │   ├── pages.py         # 页面路由
    │   └── pipeline_api.py  # Pipeline API（视频/摄像头控制）
    ├── services/
    │   └── ship_service.py  # 业务逻辑服务层
    ├── static/
    │   ├── css/style.css    # 样式文件
    │   └── js/
    │       ├── app.js       # 数据库管理前端逻辑
    │       └── pipeline.js  # Pipeline 前端逻辑
    └── templates/
        └── index.html       # 主页模板
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python ≥ 3.10**
- **Qwen VLM 服务**（OpenAI-compatible API）
- **Embedding 服务**（可选，用于语义检索）

### 2. 安装依赖

```bash
pip install -e .
# 或手动安装
pip install fastapi uvicorn jinja2 python-multipart pyyaml \
    langchain-core langchain-openai langgraph httpx \
    opencv-python numpy ultralytics
```

### 3. 配置

编辑 `config.yaml`，至少配置 VLM 服务地址：

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "YOUR_API_KEY"
  base_url: "http://your-vlm-server:7890/v1"

# 数据库后端（csv 或 sqlite）
database:
  backend: "sqlite"
  sqlite_path: "./data/ships.db"

# Web 服务
web:
  host: "0.0.0.0"
  port: 9000
```

> 完整配置项说明见 [config.yaml](config.yaml) 注释。

### 4. 启动服务

```bash
# 方式一：python -m 启动
python -m web

# 方式二：uvicorn 启动（支持热重载，推荐开发环境）
uvicorn web.app:app --ws-ping-interval 0 --host 0.0.0.0 --port 9000 --reload
```

> **说明**：`--ws-ping-interval 0` 禁用 websockets 内置心跳，防止推流时 WebSocket 连接崩溃。

### 访问地址

| 地址 | 说明 |
|------|------|
| `http://localhost:9000` | Web 管理界面 |
| `http://localhost:9000/docs` | Swagger API 文档（自动生成） |
| `http://localhost:9000/redoc` | ReDoc API 文档 |

---

## 🖥️ Web 界面操作指南

### 界面总览

Web 界面分为三个标签页（Tab），通过顶部导航栏切换：

```
┌─────────────────────────────────────────────────────┐
│  🚢 船只舷号管理系统                                  │
├─────────────────────────────────────────────────────┤
│  [🗄️ 数据库管理]  [🎬 视频 Demo]  [📷 摄像头 Demo]   │
├─────────────────────────────────────────────────────┤
│                                                     │
│              （当前 Tab 的内容区域）                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

| Tab | 功能 | 适用场景 |
|-----|------|----------|
| 🗄️ 数据库管理 | 船只数据的增删改查、搜索、批量导入、图片识别 | 日常数据管理 |
| 🎬 视频 Demo | 上传视频 → Pipeline 自动处理 → 结果对比播放 | 批量视频分析 |
| 📷 摄像头 Demo | 接入摄像头/RTSP 流，实时识别 | 现场部署 |

---

### Tab 1：数据库管理

#### 查看统计数据

页面顶部显示两个统计卡片：

- **船只总数**：当前数据库中的船只记录数
- **数据后端**：当前使用的存储后端（CSV 或 SQLITE）

数据在页面加载时自动获取，点击 **刷新** 按钮可手动更新。

#### 搜索船只

在工具栏的搜索框中输入关键词，表格会**实时过滤**显示匹配的船只：

- 支持按**舷号**搜索（如输入 `0014`）
- 支持按**描述**搜索（如输入 `白色`、`客轮`）
- 搜索不区分大小写
- 清空搜索框恢复显示全部

#### 新增船只

1. 点击工具栏的 **+ 新增船只** 按钮
2. 在弹出的对话框中填写：
   - **舷号**：船只的唯一编号（如 `0014`、`海巡123`、`A01`）
   - **描述**：船只的外观描述（如 `白色大型客轮，上层建筑为蓝色涂装`）
3. 点击 **确认** 提交
4. 如果舷号已存在，会提示错误

> 💡 **提示**：描述越详细，后续语义检索效果越好。建议包含船型、颜色、特殊标志等信息。

#### 编辑船只

1. 在表格中找到目标船只
2. 点击该行右侧的 **编辑** 按钮
3. 在弹出的对话框中修改描述（舷号不可修改）
4. 点击 **确认** 保存

#### 删除船只

1. 在表格中找到目标船只
2. 点击该行右侧的 **删除** 按钮
3. 在确认对话框中点击 **确定**
4. 删除后该船只的所有数据（包括 Embedding 向量）将被清除

> ⚠️ **注意**：删除操作不可撤销。

#### 批量导入

适用于一次性导入多条船只数据：

1. 点击工具栏的 **批量导入** 按钮
2. 在文本框中输入 **JSON 格式** 数据：

```json
{
  "A001": "白色巡逻艇，船身有蓝色条纹",
  "A002": "灰色货轮，船尾有起重机",
  "海巡001": "白色海巡船，上层建筑为灰色"
}
```

3. 点击 **导入** 提交
4. 系统会显示导入结果：成功添加数 + 跳过数（已存在的舷号会跳过）

> 💡 **格式要求**：键为舷号（字符串），值为描述（字符串），用英文双引号。

#### 图片识别入库

这是系统的核心功能之一，利用 VLM（视觉语言模型）自动识别船只图片：

**操作步骤**

1. 点击工具栏的 **📷 上传图片识别** 按钮
2. 在弹出的对话框中，**选择图片** 或 **拖拽图片** 到上传区域
   - 支持格式：JPG、PNG、BMP、WebP、GIF
   - 大小限制：20MB
3. 图片上传后会显示预览
4. 点击 **🔍 识别** 按钮，等待 VLM 分析
5. 识别结果会显示：
   - **识别到的弦号**（可手动修改）
   - **船只描述**（可手动修改）
   - 如果该弦号已存在，会显示警告提示
6. 确认无误后，点击 **✅ 确认添加** 入库

**识别质量提示**

- 图片中船体侧面的文字编号越清晰，识别效果越好
- 建议拍摄角度为船体侧面正对
- 夜间或远距离拍摄的图片识别率较低，可手动修正结果

---

### Tab 2：视频 Demo

此标签页用于上传视频并通过 Pipeline 自动处理（YOLO 检测 + VLM 识别）。

#### 上传视频

1. 切换到 **🎬 视频 Demo** 标签页
2. 在 **📤 添加视频** 区域：
   - 点击上传区域选择文件，或
   - 直接拖拽视频文件到上传区域
3. 支持的格式：MP4、AVI、MKV、MOV、FLV、WMV、WebM
4. 大小限制：500MB
5. 上传过程中会显示**进度条**
6. 上传完成后自动刷新视频列表

> 💡 如果上传的文件名与已有文件重复，系统会自动添加数字后缀（如 `video_1.mp4`）。

#### 视频列表管理

上传的视频显示在 **🎥 Demo 视频列表** 中：

- 每个视频卡片显示：文件名、文件大小
- **▶ 预览**：点击可播放原始视频
- **🗑️**：点击可删除该视频（需确认）

点击某个视频卡片会**选中**该视频，并显示 Pipeline 控制面板。

#### 启动 Pipeline 处理

选中视频后，**⚡ Pipeline 控制** 面板会显示：

**配置选项**

| 选项 | 说明 |
|------|------|
| **Agent 模式** | 启用 LangChain Agent 增强识别（更智能但更慢） |
| **并发模式** | 多帧并行处理，显著提升处理速度 |

**启动流程**

1. 从列表中选择要处理的视频
2. 根据需要勾选选项
3. 点击 **▶ 开始处理**
4. 系统会：
   - 生成唯一的任务 ID
   - 在后台启动 Pipeline 进程
   - 自动开始轮询任务状态

**处理过程中的状态指示**

- 🟠 **运行中**（橙色脉冲点）：Pipeline 正在处理
- 🟢 **完成**（绿色点）：处理成功
- 🔴 **失败**（红色点）：处理出错

处理过程中可以随时点击 **⏹ 停止** 终止任务。

#### 查看处理结果

Pipeline 完成后：

1. **视频对比播放**区域会自动加载结果视频
2. 左侧为**原始视频**，右侧为**处理结果**
3. 两个视频可以独立播放控制
4. 结果视频中标注了检测到的船只和识别到的舷号

> 💡 结果视频也保存在 `demo_output` 目录中，可直接访问文件。

#### 任务历史

**📋 任务历史** 区域显示所有 Pipeline 任务的执行记录：

- 每条记录包含：状态图标、视频名、任务 ID、进度/错误信息
- **▶ 播放**：点击可直接播放该任务的输出视频
- **⏹ 停止**：运行中的任务可以停止
- 点击 **刷新** 按钮更新列表

---

### Tab 3：摄像头 Demo

此标签页用于接入摄像头或 RTSP 流进行实时船只识别。

#### 配置输入源

在 **📷 摄像头 / RTSP 配置** 面板中：

| 输入源选项 | 说明 | 示例 |
|-----------|------|------|
| **本地摄像头 (0)** | 使用设备默认摄像头 | 直接选择即可 |
| **RTSP 流** | 网络摄像头的 RTSP 地址 | `rtsp://127.0.0.1/stream` |
| **自定义** | 任意 OpenCV 支持的输入 | 视频文件路径、HTTP 流等 |

**处理选项**

| 选项 | 说明 |
|------|------|
| **Agent 模式** | 启用 Agent 增强 |
| **并发模式** | 多帧并行处理 |
| **显示实时画面** | 处理过程中显示实时画面窗口 |

#### 启动摄像头识别

1. 选择输入源并配置选项
2. 点击 **▶ 启动摄像头识别**
3. 系统会：
   - 显示 **📊 运行状态** 面板
   - 状态指示灯变为橙色脉冲（运行中）
   - 开始每 3 秒轮询一次状态

#### 停止与监控

- 点击 **⏹ 停止** 按钮终止摄像头识别
- 状态面板会实时显示当前处理状态
- 处理结果保存到 `demo_output` 目录
- 后端也可以通过 CLI 独立运行 pipeline

> 💡 **提示**：摄像头 Demo 的后端处理逻辑与视频 Demo 相同，区别仅在于输入源是实时流而非文件。

---

## 💻 CLI 命令行使用

```bash
# 单次查询
ship-hull "帮我查一下弦号0014是什么船"

# 交互模式
ship-hull --interactive

# 详细调用链
ship-hull --verbose "我看到一艘灰色军舰"
```

---

## 🔌 API 参考

启动服务后，访问以下地址查看自动生成的 API 文档：

| 地址 | 说明 |
|------|------|
| `http://localhost:9000/docs` | Swagger UI（交互式文档，可直接测试） |
| `http://localhost:9000/redoc` | ReDoc（阅读式文档） |

### 船只管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/ships` | 获取所有船只列表 |
| `GET` | `/api/ships/{hull_number}` | 查询单条船只 |
| `POST` | `/api/ships` | 新增船只 |
| `PUT` | `/api/ships/{hull_number}` | 更新船只描述 |
| `DELETE` | `/api/ships/{hull_number}` | 删除船只 |
| `POST` | `/api/ships/bulk` | 批量添加船只 |
| `GET` | `/api/ships/search?q=关键词` | 按描述搜索 |
| `GET` | `/api/ships/stats` | 数据库统计 |
| `POST` | `/api/ships/recognize` | 上传图片识别（不入库） |
| `POST` | `/api/ships/recognize-and-add` | 上传图片识别并自动入库 |

### Pipeline API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/pipeline/videos` | 获取 Demo 视频列表 |
| `POST` | `/api/pipeline/videos/upload` | 上传视频 |
| `DELETE` | `/api/pipeline/videos/{filename}` | 删除视频 |
| `POST` | `/api/pipeline/start` | 启动 Pipeline 处理 |
| `GET` | `/api/pipeline/status` | 获取所有任务状态 |
| `GET` | `/api/pipeline/status/{task_id}` | 获取单个任务状态 |
| `POST` | `/api/pipeline/stop/{task_id}` | 停止任务 |
| `GET` | `/api/pipeline/outputs` | 获取输出视频列表 |
| `GET` | `/api/pipeline/outputs/{filename}` | 下载输出视频 |
| `DELETE` | `/api/pipeline/tasks/clear` | 清除历史任务 |

### API 调用示例

```bash
# 获取所有船只
curl http://localhost:9000/api/ships

# 新增船只
curl -X POST http://localhost:9000/api/ships \
  -H "Content-Type: application/json" \
  -d '{"hull_number": "TEST01", "description": "测试船只"}'

# 上传图片识别
curl -X POST http://localhost:9000/api/ships/recognize \
  -F "file=@ship_photo.jpg"

# 搜索
curl "http://localhost:9000/api/ships/search?q=白色"
```

---

## ⚙️ 配置说明

所有配置集中在 `config.yaml`，支持以下模块：

| 配置块 | 说明 |
|--------|------|
| `llm` | VLM 对话模型（用于图片识别） |
| `embed` | Embedding 模型（用于语义检索） |
| `retrieval` | RAG 检索参数（top_k、阈值） |
| `vector_store` | 向量存储路径 |
| `database` | 数据库后端（csv/sqlite）及路径 |
| `web` | Web 服务 host/port |
| `demo_video` | 视频 Demo 目录与限制 |
| `pipeline` | 视频处理流水线参数（YOLO、追踪器、并发等） |

配置优先级：`config.yaml` > 内置默认值。支持深层合并，只需覆盖需要修改的字段。

### 配置速查表

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `web.host` | `0.0.0.0` | 监听地址 |
| `web.port` | `9000` | 监听端口 |
| `database.backend` | `sqlite` | 数据后端（csv/sqlite） |
| `database.sqlite_path` | `./data/ships.db` | SQLite 文件路径 |
| `llm.model` | `Qwen/Qwen3-VL-4B-AWQ` | VLM 模型名 |
| `llm.base_url` | `http://localhost:7890/v1` | VLM 服务地址 |
| `demo_video.dir` | `./demovid` | 视频存储目录 |
| `demo_video.output_dir` | `./demo_output` | 输出目录 |
| `demo_video.max_file_size_mb` | `500` | 最大上传大小 (MB) |
| `pipeline.concurrent_mode` | `true` | 是否并发处理 |
| `pipeline.max_concurrent` | `4` | 最大并发数 |
| `pipeline.process_every_n_frames` | `15` | 每隔 N 帧处理一次 |
| `pipeline.save_output_video` | `true` | 是否保存推理结果视频到 output_dir |
| `retrieval.top_k` | `3` | 语义检索返回的候选数量 |

---

## ❓ 常见问题

### Q: 启动后页面空白或报错？

**A:** 检查以下几点：
1. 确认 `config.yaml` 中的 VLM 服务地址正确且可达
2. 确认 VLM 服务正在运行
3. 检查终端日志中的错误信息

### Q: 图片识别一直失败？

**A:** 可能原因：
1. VLM 服务未启动或地址配置错误
2. 图片格式不支持（请使用 JPG/PNG/BMP/WebP）
3. 图片过大（限制 20MB）
4. VLM 服务响应超时

### Q: 视频 Pipeline 处理很慢？

**A:** 优化建议：
1. 勾选**并发模式**启用多帧并行
2. 调大 `config.yaml` 中的 `pipeline.process_every_n_frames`（减少处理帧数）
3. 调大 `pipeline.detect_every_n_frames`（减少检测频率）
4. 使用 GPU 加速的 VLM 服务

### Q: 批量导入的 JSON 格式是什么？

**A:** 标准 JSON 对象，键为舷号，值为描述：
```json
{
  "0014": "白色大型客轮",
  "0123": "白色邮轮"
}
```

### Q: 数据库后端如何切换？

**A:** 修改 `config.yaml`：
```yaml
database:
  backend: "csv"      # 切换为 CSV
  # backend: "sqlite"  # 切换为 SQLite
```
重启服务后生效。注意：两种后端的数据不互通。

### Q: 如何修改 Web 服务端口？

**A:** 修改 `config.yaml`：
```yaml
web:
  host: "0.0.0.0"
  port: 9000  # 改为你想要的端口
```
或启动时直接指定：
```bash
uvicorn web.app:app --host 0.0.0.0 --port 9000
```

### Q: 语义搜索不生效？

**A:** 语义搜索需要 Embedding 服务：
1. 确认 `config.yaml` 中的 `embed` 配置正确
2. 确认 Embedding 服务正在运行
3. 首次使用需要构建 Embedding 索引（系统会自动处理）

### Q: 如何备份数据？

**A:** 根据后端类型备份对应文件：
- **CSV 后端**：备份 `data/ships.csv`
- **SQLite 后端**：备份 `data/ships.db`

---

## 🛠️ 技术栈

- **后端框架**：FastAPI + Uvicorn
- **模板引擎**：Jinja2
- **视觉模型**：Qwen3-VL-4B-AWQ（OpenAI-compatible API）
- **目标检测**：YOLOv8（Ultralytics）
- **追踪算法**：ByteTrack
- **Embedding**：Qwen3-Embedding-0.6B
- **向量检索**：余弦相似度（SQLite 存储）
- **LLM 编排**：LangChain + LangGraph
- **前端**：原生 HTML/CSS/JS（无框架依赖）

---

## 📄 License

MIT License

## Agent-oriented outputs

- `GET /api/pipeline/memory/{task_id}`: returns `track_memory`, `event_log`, and `agent_trace` for mission-oriented analysis.
- Pipeline summary now includes `paper_summary`, uncertainty statistics, and risk counts for experiment reporting.

---

## 学习型 Policy 实验流程

本节记录论文中三种学习型认知策略的完整复现实验命令。三种策略共用同一套日志数据和测试协议：

1. **Learned Classification Policy**：从 oracle action 训练 softmax 分类器。
2. **Linear Utility Policy**：从 counterfactual utility 训练线性效用回归器。
3. **MLP Utility Policy**：从 counterfactual utility 训练小型非线性 MLP 效用网络。

正式对比时建议固定：

- 配置：`config.final_public.yaml`
- 测试轨迹标注：`data/annotations/test_tracks.final_public.jsonl`
- 测试实体标注：`data/annotations/test_entities.final_public.jsonl`
- 输出根目录：由 `config.final_public.yaml -> experiment.output_dir` 决定，当前通常为 `experiment_outputs/final_public_v1`

### 1. 收集 Policy 训练日志

先用固定间隔策略在 `policy_train` split 上收集较宽覆盖的状态-动作日志。该步骤会生成 `actions.jsonl`、`recognition.jsonl`、`visual.jsonl`、`entities.jsonl` 等文件。

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final_public.yaml --split policy_train --run-name policy_collect_fixed --memory-mode full --policy-mode fixed_interval --network-profile real --quality on --ledger on --archive on --experience off --risk off --visual-archive on --visual-decision on --visual-open-set on --entity-reconciliation on --max-frames 0
```

如果 `config.final_public.yaml` 中的 `experiment.output_dir` 是 `experiment_outputs/final_public_v1`，则日志目录为：

```bash
experiment_outputs/final_public_v1/policy_collect_fixed
```

### 2. 生成基础 Policy 数据集

将收集到的日志与 `policy_train` 的实体级真值标注对齐，生成每个决策点的特征、teacher action 和 oracle action。

```bash
python -m experiments.build_policy_dataset --run-dir experiment_outputs/final_public_v1/policy_collect_fixed --annotations data/annotations/policy_train_entities.jsonl --output data/policy/policy_train_dataset.jsonl
```

检查样本数、跳过数量和动作类别分布：

```bash
cat data/policy/policy_train_dataset.summary.json
head -n 3 data/policy/policy_train_dataset.jsonl
```

重点看：

- `samples`：可训练样本数。
- `skipped_no_truth`：日志实体没有匹配到训练标注的数量。
- `oracle_action_counts`：oracle 标签分布。
- `teacher_action_counts`：原始规则策略动作分布。

### 3. 方法一：Learned Classification Policy

该方法直接学习 `oracle_action`，输出五类动作概率：`defer`、`query`、`stop_known`、`stop_out_of_archive`、`escalate_review`。

训练命令：

```bash
python -m experiments.train_learned_policy --dataset data/policy/policy_train_dataset.jsonl --output models/policy/learned_policy_v1.json --target oracle_action --validation-fraction 0.2 --epochs 1200 --learning-rate 0.05 --balance balanced
```

查看离线训练结果：

```bash
cat models/policy/learned_policy_v1.summary.json
```

正式实验命令：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final_public.yaml --split test --run-name main_learned_v1 --memory-mode full --policy-mode learned --policy-model models/policy/learned_policy_v1.json --network-profile real --quality on --ledger on --archive on --experience off --risk off --visual-archive on --visual-decision on --visual-open-set on --entity-reconciliation on --max-frames 0
```

### 4. 生成 Utility 数据集

Linear Utility 和 MLP Utility 都使用 counterfactual utility 标签。该步骤在基础 policy 数据集上，为每条样本生成 `utility_by_action` 和 `utility_action`。

```bash
python -m experiments.build_policy_utility_dataset --dataset data/policy/policy_train_dataset.jsonl --output data/policy/policy_train_utility_dataset.jsonl
```

查看效用标签分布：

```bash
cat data/policy/policy_train_utility_dataset.summary.json
head -n 3 data/policy/policy_train_utility_dataset.jsonl
```

默认 utility 包含查询成本、时延成本、延迟决策成本、复核成本、错误接受惩罚、错误拒识惩罚和不确定性收益。需要调参时可使用：

```bash
python -m experiments.build_policy_utility_dataset --dataset data/policy/policy_train_dataset.jsonl --output data/policy/policy_train_utility_dataset.jsonl --query-cost 0.18 --latency-cost 0.04 --defer-cost 0.08 --escalation-cost 0.35 --false-accept-penalty 1.00 --false-reject-penalty 0.90 --wrong-known-penalty 0.95 --uncertainty-bonus 0.20
```

### 5. 方法二：Linear Utility Policy

该方法训练每个动作的 utility 分数，在线执行时选择效用最高的动作，并仍保留 action eligibility mask 以避免明显不合法动作。

训练命令：

```bash
python -m experiments.train_utility_policy --dataset data/policy/policy_train_utility_dataset.jsonl --output models/policy/learned_utility_policy_v1.json --validation-fraction 0.2 --l2 0.01 --balance sqrt
```

查看离线训练结果：

```bash
cat models/policy/learned_utility_policy_v1.summary.json
```

正式实验命令：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final_public.yaml --split test --run-name main_learned_utility_v1 --memory-mode full --policy-mode learned_utility --policy-model models/policy/learned_utility_policy_v1.json --network-profile real --quality on --ledger on --archive on --experience off --risk off --visual-archive on --visual-decision on --visual-open-set on --entity-reconciliation on --max-frames 0
```

### 6. 方法三：MLP Utility Policy

该方法使用小型 MLP 学习非线性的 evidence-utility scorer。训练时需要 PyTorch；导出的 `.json` 模型在线推理只依赖 NumPy，因此正式实验路径不需要 PyTorch。

训练命令：

```bash
python -m experiments.train_mlp_utility_policy --dataset data/policy/policy_train_utility_dataset.jsonl --output models/policy/mlp_utility_policy_v1.json --validation-fraction 0.2 --hidden-sizes 32,16 --activation relu --dropout 0.1 --epochs 500 --batch-size 128 --learning-rate 0.001 --weight-decay 0.0001 --balance sqrt --ranking-weight 0.25 --ranking-margin 0.15
```

查看离线训练结果：

```bash
cat models/policy/mlp_utility_policy_v1.summary.json
```

正式实验命令：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final_public.yaml --split test --run-name main_mlp_utility_v1 --memory-mode full --policy-mode learned_utility --policy-model models/policy/mlp_utility_policy_v1.json --network-profile real --quality on --ledger on --archive on --experience off --risk off --visual-archive on --visual-decision on --visual-open-set on --entity-reconciliation on --max-frames 0
```

### 7. 三种学习方法评估

轨迹级评估：

```bash
python -m experiments.evaluate --run-dir experiment_outputs/final_public_v1/main_learned_v1 --annotations data/annotations/test_tracks.final_public.jsonl
python -m experiments.evaluate --run-dir experiment_outputs/final_public_v1/main_learned_utility_v1 --annotations data/annotations/test_tracks.final_public.jsonl
python -m experiments.evaluate --run-dir experiment_outputs/final_public_v1/main_mlp_utility_v1 --annotations data/annotations/test_tracks.final_public.jsonl
```

实体级评估：

```bash
python -m experiments.evaluate_entities --run-dir experiment_outputs/final_public_v1/main_learned_v1 --annotations data/annotations/test_entities.final_public.jsonl
python -m experiments.evaluate_entities --run-dir experiment_outputs/final_public_v1/main_learned_utility_v1 --annotations data/annotations/test_entities.final_public.jsonl
python -m experiments.evaluate_entities --run-dir experiment_outputs/final_public_v1/main_mlp_utility_v1 --annotations data/annotations/test_entities.final_public.jsonl
```

推荐同时和规则主方法做同口径对比：

```bash
for run in main_full main_learned_v1 main_learned_utility_v1 main_mlp_utility_v1; do
  echo "=== $run track ==="
  python -m experiments.evaluate --run-dir experiment_outputs/final_public_v1/$run --annotations data/annotations/test_tracks.final_public.jsonl
  echo "=== $run entity ==="
  python -m experiments.evaluate_entities --run-dir experiment_outputs/final_public_v1/$run --annotations data/annotations/test_entities.final_public.jsonl
done
```

关键指标：

- `known_identity_accuracy`：在库实体身份核验准确率。
- `entity_unknown_rejection_f1`：档案外实体拒识 F1。
- `entity_unknown_false_acceptance_rate`：档案外船误接收入库率。
- `avg_vlm_calls_per_entity`：每个实体平均 VLM 调用数。
- `success_per_vlm_call`：单位 VLM 调用带来的任务成功数。
- `unnecessary_query_rate`：无效查询比例。

确认 learned policy 是否真的启用：

```bash
grep -m 5 "learned_policy_action" experiment_outputs/final_public_v1/main_mlp_utility_v1/actions.jsonl
```

动作分布快速统计：

```bash
python - <<'PY'
import json, collections
from pathlib import Path
for name in ["main_full", "main_learned_v1", "main_learned_utility_v1", "main_mlp_utility_v1"]:
    p = Path("experiment_outputs/final_public_v1") / name / "actions.jsonl"
    rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
    actions = collections.Counter(r.get("action") for r in rows)
    learned = collections.Counter((r.get("skill_signals") or {}).get("learned_policy_action") for r in rows if (r.get("skill_signals") or {}).get("learned_policy_action"))
    queries = sum(bool(r.get("should_query")) for r in rows)
    print("\n", name)
    print("actions=", dict(actions))
    print("learned=", dict(learned))
    print("queries=", queries, "total=", len(rows), "query_rate=", round(queries / max(1, len(rows)), 4))
PY
```

### 8. 审计脚本

#### 8.1 已知船错误审计

查看所有已知船问题，包括被拒识、部分拒识、未解析、错配：

```bash
python -m experiments.audit_identity_errors --run-dir experiment_outputs/final_public_v1/main_mlp_utility_v1 --entity-annotations data/annotations/test_entities.final_public.jsonl --category problem
```

只看严格在库船被拒识：

```bash
python -m experiments.audit_identity_errors --run-dir experiment_outputs/final_public_v1/main_mlp_utility_v1 --entity-annotations data/annotations/test_entities.final_public.jsonl --category known_false_rejection
```

输出文件：

```bash
experiment_outputs/final_public_v1/main_mlp_utility_v1/identity_error_audit.jsonl
experiment_outputs/final_public_v1/main_mlp_utility_v1/identity_error_audit.csv
```

查看 CSV：

```bash
column -s, -t < experiment_outputs/final_public_v1/main_mlp_utility_v1/identity_error_audit.csv | less -S
```

错误类别含义：

- `known_false_rejection`：真实在库船，预测 fragment 全部为 `out_of_archive`。
- `known_partial_rejection`：真实在库船，部分 fragment 被判为 `out_of_archive`。
- `known_wrong_identity`：真实在库船被确认成其他档案 ID。
- `known_unresolved`：真实在库船没有确认，也没有严格全拒识。
- `known_correct`：正确确认；默认不输出，使用 `--category all` 或 `--include-correct` 才输出。

#### 8.2 在库判定倾向审计

查看哪些档案 ID 更容易被系统判为在库，以及哪些真实船更容易被吸收入库：

```bash
python -m experiments.audit_archive_tendency --run-dir experiment_outputs/final_public_v1/main_mlp_utility_v1 --annotations data/annotations/test_entities.final_public.jsonl --top 30
```

输出文件：

```bash
experiment_outputs/final_public_v1/main_mlp_utility_v1/archive_tendency_by_archive.csv
experiment_outputs/final_public_v1/main_mlp_utility_v1/archive_tendency_by_truth_vessel.csv
experiment_outputs/final_public_v1/main_mlp_utility_v1/archive_tendency_accepted_entities.csv
```

查看档案 ID 维度：

```bash
column -s, -t < experiment_outputs/final_public_v1/main_mlp_utility_v1/archive_tendency_by_archive.csv | less -S
```

重点字段：

- `accepted_entities`：该档案 ID 被接受为在库的次数。
- `unknown_false_accepts`：档案外船被错判成该 ID 的次数。
- `wrong_known_accepts`：已知船被错配成该 ID 的次数。
- `candidate_mentions`：该 ID 被作为 archive candidate 的次数。
- `visual_top1_mentions`：视觉检索中该 ID 为 top-1 的次数。

查看真实船维度：

```bash
column -s, -t < experiment_outputs/final_public_v1/main_mlp_utility_v1/archive_tendency_by_truth_vessel.csv | less -S
```

重点判断：

- `truth_status=unknown` 且 `acceptance_rate` 高：该档案外船容易被误吸收入库。
- `truth_status=known` 且 `acceptance_rate` 低：该在库船容易被拒识或未解析。
- `accepted_identities`：该真实船通常被判成哪些档案 ID。

## Qwen3-VL direct visual archive matching

The visual archive can use Qwen3-VL-Reranker instead of DINOv2. This mode compares each qualified target crop with every archive prototype through vLLM's `/score` API, groups prototype scores by vessel identity, and writes the existing `visual_candidate_id`, `visual_similarity_score`, `visual_margin`, and `visual_matches` fields. Existing evaluation scripts therefore remain compatible.

Start the reranker server from the project root:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-Reranker-2B --served-model-name Qwen/Qwen3-VL-Reranker-2B --runner pooling --trust-remote-code --dtype bfloat16 --api-key abc123 --gpu-memory-utilization 0.2 --max-model-len 8192 --hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --chat-template configs/qwen3_vl_reranker.jinja --port 7894
```

Select the direct matcher in the experiment configuration:

```yaml
visual_reranker:
  model: Qwen/Qwen3-VL-Reranker-2B
  api_key: abc123
  base_url: http://localhost:7894
  request_timeout_seconds: 120
  max_retries: 1
  request_batch_size: 8

experiment:
  visual_archive:
    enabled: true
    backend: qwen_vl_reranker
    archive_root: data/archive/visual_prototypes
    top_k: 3
```

`request_batch_size` controls how many archive images are scored per HTTP request; every archive image is still evaluated. The existing decision thresholds were tuned for DINOv2 and must be recalibrated before enabling `decision_enabled` or `open_set_enabled` for reranker scores.

Set `experiment.visual_archive.backend: dinov2` to switch back to the original embedding matcher.

## Central VLM controller

The existing recognition chain remains the default. Set `experiment.controller_mode` to
`central_vlm` to let the shared VLM select `SearchArchive`, `ReadArchive`,
`ReadHistory`, or `VerifyEvidence` for each persistent entity. Each entity has an
independent episode, evidence ledger, and query budget; tool results are appended
before the next controller decision. Set it to `mock` for deterministic integration
checks without a model service.

```yaml
experiment:
  controller_mode: central_vlm  # legacy | central_vlm | mock
  central_controller:
    terminal_policy: multimodal_evidence
    query_limit: 3
    max_control_steps: 12
    max_workers: 4
```

The central mode needs `experiment.visual_archive.enabled: true` for
`SearchArchive`. The controller uses the existing `llm` endpoint, while
`VerifyEvidence` uses the same endpoint with a separate evidence-extraction prompt.
`legacy` keeps the existing MLP/rule scheduler and three-step identity path for
backward-compatible experiments.

The central controller keeps raw tool returns separate from source-linked
claims. A `belief_patch` may add or revise claims, candidate assessments, gaps,
and evidence validity only when the referenced records belong to the current
episode. It cannot change raw evidence, budgets, visual scores, or entity
association.

## Qwen reranker replacement: complete formal experiment

This workflow completely replaces DINOv2 with Qwen3-VL-Reranker while preserving the downstream evidence, policy, entity reconciliation, open-set, conflict, and evaluation logic. Do not select thresholds from `test`; any earlier test-set score probe is diagnostic only.

### 1. Start and verify the reranker

Run from the project root. The server must use the pooling runner and the supplied score template.

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-Reranker-2B --served-model-name Qwen/Qwen3-VL-Reranker-2B --runner pooling --trust-remote-code --dtype bfloat16 --api-key abc123 --gpu-memory-utilization 0.2 --max-model-len 8192 --hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --chat-template /media/ddc/新加卷/gc/SQL-boat-v2/configs/qwen3_vl_reranker.jinja --port 7894
```

```bash
curl -s -H "Authorization: Bearer abc123" http://127.0.0.1:7894/v1/models
```

Confirm that the effective training configuration selects Qwen rather than DINOv2:

```bash
python -c "from config import load_config; c=load_config('config.final_public.yaml'); print('backend=',c['experiment']['visual_archive']['backend']); print('model=',c['visual_reranker']['model']); print('base_url=',c['visual_reranker']['base_url'])"
```

Expected backend:

```text
qwen_vl_reranker
```

If the command prints `dinov2`, set the following field in `config.final_public.yaml` before collecting data:

```yaml
experiment:
  visual_archive:
    backend: qwen_vl_reranker
```

### 2. Collect new policy-training evidence

Use `fixed_interval` to collect broad state coverage. Visual decisions are disabled during collection so unknown thresholds cannot alter the training trajectories.

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final_public.yaml --split policy_train --run-name policy_collect_qwen_reranker_v1 --memory-mode full --policy-mode fixed_interval --network-profile real --quality on --ledger on --archive on --experience off --risk off --visual-archive on --visual-decision off --visual-open-set off --entity-reconciliation on --max-frames 0
```

Check that Qwen produced usable visual records:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiment_outputs/final_public_v1/policy_collect_qwen_reranker_v1/visual.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
print("records=", len(rows))
print("backends=", sorted({row.get("visual_backend", "") for row in rows}))
print("nonzero_scores=", sum(float(row.get("visual_similarity_score", 0) or 0) > 0 for row in rows))
print("pairs_scored=", sum(int(row.get("visual_pairs_scored", 0) or 0) for row in rows))
print("physical_requests=", sum(int(row.get("visual_model_request_count", 0) or 0) for row in rows))
PY
```

### 3. Calibrate and freeze Qwen thresholds

Use only `policy_train_entities.jsonl`. The formal command maximizes observation-level KAcc subject to `UFAR <= 2%`; this avoids letting the larger unknown class dominate threshold selection. The report also contains ATS, URec, balanced accuracy, and precision.

```bash
python -m experiments.calibrate_visual_reranker --run-dir experiment_outputs/final_public_v1/policy_collect_qwen_reranker_v1 --annotations data/annotations/policy_train_entities.jsonl --output data/policy/qwen_reranker_thresholds_v1.json --base-config config.final_public.yaml --output-config config.final_public.qwen_reranker.yaml --max-ufar 0.02 --objective kacc --score-step 0.01 --margin-step 0.01 --top 20
```

Inspect the selected operating point:

```bash
python -c "import json; x=json.load(open('data/policy/qwen_reranker_thresholds_v1.json',encoding='utf-8')); print(json.dumps(x['selected'],indent=2))"
```

Export the frozen thresholds for the policy-dataset command:

```bash
export QWEN_VISUAL_SCORE=$(python -c "import json; print(json.load(open('data/policy/qwen_reranker_thresholds_v1.json'))['selected']['score_threshold'])")
export QWEN_VISUAL_MARGIN=$(python -c "import json; print(json.load(open('data/policy/qwen_reranker_thresholds_v1.json'))['selected']['margin_threshold'])")
echo "score=$QWEN_VISUAL_SCORE margin=$QWEN_VISUAL_MARGIN"
```

From this point onward, do not modify the frozen configuration using test results.

### 4. Build the new policy datasets

The previous MLP policy was trained on DINOv2 score distributions and must not be reused. Build a new supervised dataset from the Qwen collection run using the frozen visual thresholds.

```bash
python -m experiments.build_policy_dataset --run-dir experiment_outputs/final_public_v1/policy_collect_qwen_reranker_v1 --annotations data/annotations/policy_train_entities.jsonl --output data/policy/policy_train_dataset_qwen_reranker_v1.jsonl --unknown-visual-score "$QWEN_VISUAL_SCORE" --unknown-visual-margin "$QWEN_VISUAL_MARGIN"
```

```bash
python -m experiments.build_policy_utility_dataset --dataset data/policy/policy_train_dataset_qwen_reranker_v1.jsonl --output data/policy/policy_train_utility_qwen_reranker_v1.jsonl
```

Check the generated summaries before training:

```bash
cat data/policy/policy_train_dataset_qwen_reranker_v1.summary.json
cat data/policy/policy_train_utility_qwen_reranker_v1.summary.json
```

### 5. Train the replacement MLP utility policy

The architecture and utility-learning logic remain unchanged; only the training observations now contain Qwen visual evidence.

```bash
python -m experiments.train_mlp_utility_policy --dataset data/policy/policy_train_utility_qwen_reranker_v1.jsonl --output models/policy/mlp_utility_qwen_reranker_v1.json --validation-fraction 0.2 --hidden-sizes 32,16 --activation relu --dropout 0.1 --epochs 500 --batch-size 128 --learning-rate 0.001 --weight-decay 0.0001 --balance sqrt --ranking-weight 0.25 --ranking-margin 0.15
```

```bash
cat models/policy/mlp_utility_qwen_reranker_v1.summary.json
```

### 6. Verify the frozen formal configuration

```bash
python - <<'PY'
from config import load_config

config = load_config("config.final_public.qwen_reranker.yaml")
visual = config["experiment"]["visual_archive"]
policy = config["experiment"]["policy"]
print("backend=", visual["backend"])
print("support_score=", visual["min_support_score"])
print("support_margin=", visual["min_support_margin"])
print("out_of_archive_score=", visual["out_of_archive_score"])
print("out_of_archive_margin=", visual["out_of_archive_margin"])
print("default_policy_model=", policy.get("learned_model_path"))
PY
```

### 7. Run the formal test experiment

This command preserves the original full Agent logic. The runtime replacement is limited to `DINOv2 -> Qwen3-VL-Reranker` and the newly trained policy weights required by the changed score distribution.

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment --manifest data/annotations/manifest.jsonl --config config.final_public.qwen_reranker.yaml --split test --run-name main_mlp_qwen_reranker_v1 --memory-mode full --policy-mode learned_utility --policy-model models/policy/mlp_utility_qwen_reranker_v1.json --network-profile real --quality on --ledger on --archive on --experience off --risk off --visual-archive on --visual-decision on --visual-open-set on --entity-reconciliation on --max-frames 0
```

Do not use `--keep-existing-output` for the final run. A formal run must start from an empty run directory, which `run_experiment` handles automatically.

### 8. Evaluate the formal run

Track-level metrics:

```bash
python -m experiments.evaluate --run-dir experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1 --annotations data/annotations/test_tracks.final_public.jsonl
```

Entity-level paper metrics:

```bash
python -m experiments.evaluate_entities --run-dir experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1 --annotations data/annotations/test_entities.final_public.jsonl
```

Confirm that the final run used only Qwen visual matching:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1/visual.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
print("records=", len(rows))
print("backends=", sorted({row.get("visual_backend", "") for row in rows}))
print("pairs_scored=", sum(int(row.get("visual_pairs_scored", 0) or 0) for row in rows))
print("physical_requests=", sum(int(row.get("visual_model_request_count", 0) or 0) for row in rows))
PY
```

### 9. Audit identity failures

```bash
python -m experiments.audit_identity_errors --run-dir experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1 --entity-annotations data/annotations/test_entities.final_public.jsonl --category problem
```

```bash
python -m experiments.audit_archive_tendency --run-dir experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1 --annotations data/annotations/test_entities.final_public.jsonl --top 30
```

Formal paper results must use the entity-level evaluator. The threshold-calibration report is an observation-level training diagnostic and must not be copied into the main result table.

## Reconcile final test annotations with the manifest

The manifest is the source of truth for dataset roles. Whenever its split assignments change, rebuild the track and entity annotation files so they contain exactly the current `test` videos. Do not edit entity rows independently; confirm track rows first and regenerate entities from them.

`V072`, `V126`, `V143`, and `V180` produced no valid annotation tracks and are explicitly retained in the manifest with `split: excluded` and an `exclusion_reason`. The final test split therefore contains 93 videos. Reconcile the existing confirmed tracks after any manifest update with:

```bash
python -m experiments.reconcile_test_annotations --manifest data/annotations/manifest.jsonl --base-tracks data/annotations/test_tracks.final_public.before_reconcile.jsonl --output-tracks data/annotations/test_tracks.final_public.jsonl --output-entities data/annotations/test_entities.final_public.jsonl --report data/annotations/test_annotations.final_public.audit.json
```

Validate and inspect the canonical result:

```bash
python -m experiments.validate_annotations data/annotations/test_tracks.final_public.jsonl
cat data/annotations/test_annotations.final_public.audit.json
wc -l data/annotations/test_tracks.final_public.jsonl data/annotations/test_entities.final_public.jsonl
```

The frozen audit must report `passed: true`, `manifest_test_videos: 93`, `output_track_rows: 339`, `output_entities: 232`, and no missing test videos. It also records removal of stale annotations for `V020`, `V115`, `V116`, and `V117`.

For annotation-only reconciliation, existing run directories that processed the earlier 97-video manifest do not need to be recomputed. When annotations are supplied, both evaluators filter predictions, observations, model calls, latency, and entity outputs to the reconciled annotation set. Re-evaluate an existing formal run with:

```bash
python -m experiments.evaluate --run-dir experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1 --annotations data/annotations/test_tracks.final_public.jsonl
python -m experiments.evaluate_entities --run-dir experiment_outputs/final_public_v1/main_mlp_qwen_reranker_v1 --annotations data/annotations/test_entities.final_public.jsonl
```

## Persistent identity and review eligibility

The online identity state machine applies the following constraints:

- A learned action cannot replace `confirmed` or `structure_verified` with `review_requested`.
- `escalate_review` is eligible only after the entity query budget is exhausted or after at least two failed recognition attempts.
- An `uncertain` archive candidate is promoted to `structure_verified` when structure and visual evidence independently agree and pass their calibrated support gates.
- Reliable disagreement between different structure and visual identities remains `conflicting`; it is not force-accepted.
- During entity aggregation, a redundant review fragment cannot erase an accepted identity. A review caused by a distinct conflicting identity is still preserved.

The visual agreement gate is configured under `experiment.visual_archive`:

```yaml
decision_enabled: true
agreement_confirmation_enabled: true
agreement_min_text_score: 0.70
min_support_score: 0.56
min_support_margin: 0.09
```

`min_support_score` and `min_support_margin` must be the calibrated values for the selected visual backend. Do not copy Qwen reranker thresholds to DINOv2 or vice versa.

### Audit review transitions

The identity audit now reports `learned_policy_actions`, actual `review_actions`, and `blocked_review_actions`. An actual review row includes the state before and after escalation, learned action, confidence, and query budget:

```bash
python -m experiments.audit_identity_errors \
  --run-dir experiment_outputs/final_public_v1/main_mlp_utility_v2 \
  --entity-annotations data/annotations/test_entities.final_public.jsonl \
  --category known_unresolved
```

### Smoke-test the previously unresolved known vessels

Create a temporary manifest containing only the seven affected videos:

```bash
python - <<'PY'
import json
from pathlib import Path

selected = {"V002", "V006", "V011", "V018", "V021", "V023", "V162"}
source = Path("data/annotations/manifest.jsonl")
output = Path("data/annotations/manifest.known_logic_smoke.jsonl")
rows = []
for line in source.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    if str(row.get("video_id", "")) in selected:
        row["split"] = "known_logic_smoke"
        rows.append(row)
output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
print(output, "videos=", len(rows), "ids=", [row["video_id"] for row in rows])
PY
```

Run the corrected logic with a new output name:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment \
  --manifest data/annotations/manifest.known_logic_smoke.jsonl \
  --config config.final_public.yaml \
  --split known_logic_smoke \
  --run-name known_logic_smoke_mlp_v2 \
  --memory-mode full \
  --policy-mode learned_utility \
  --policy-model models/policy/mlp_utility_policy_v1.json \
  --network-profile real \
  --quality on --ledger on --archive on --experience off --risk off \
  --visual-archive on --visual-decision on --visual-open-set on \
  --entity-reconciliation on --max-frames 0
```

Audit the smoke run before a full experiment:

```bash
python -m experiments.audit_identity_errors \
  --run-dir experiment_outputs/final_public_v1/known_logic_smoke_mlp_v2 \
  --entity-annotations data/annotations/test_entities.final_public.jsonl \
  --category problem
```

`V018` and `V162` may remain unresolved when the rerun still contains reliable identity disagreement; they are genuine conflict candidates. The other five cases should no longer become unresolved merely because a learned escalation overwrote a verified identity.

### Required formal rerun sequence

Old logs remain valid for diagnosing the bug, but they are not corrected autonomous results. After the smoke run passes:

1. Recollect the `policy_train` fixed-interval run with a new run name.
2. Rebuild the classification and utility datasets from that new run.
3. Retrain the linear classification, linear utility, and MLP utility policies using the commands in the learned-policy section above, writing new `v2` model files.
4. Run the complete 93-video formal test with a new run name and no `--keep-existing-output`.
5. Evaluate at entity level and run both identity audits.

This recollection is required because the corrected evidence fusion changes terminal states and therefore changes the policy-training state distribution.

## Final 50-identity archive expansion

The curated expansion promotes 33 previously out-of-archive test entities into registered identities. Identities without a readable physical hull number use their video ID, such as `V123`, as the archive identifier. The final files contain 50 distinct known identities:

```text
data/annotations/manifest.jsonl
data/annotations/test_tracks.final_public.jsonl
data/annotations/test_entities.final_public.jsonl
data/ships.csv
```

Only the 33 new visual identity directories are generated locally:

```text
data/archive/visual_prototypes/
```

Merge these directories into the existing server archive. Do not delete or overwrite the server's existing identities such as `003` and `012`:

```bash
rsync -av data/archive/visual_prototypes/ /media/ddc/新加卷/gc/SQL-boat-v2/data/archive/visual_prototypes/
```

The text archive uses descriptions from `public_pool_vlm_drafts.jsonl` plus curated structural summaries. Each new identity has at most four text prototypes. The reproducible selection and audit files are:

```text
data/annotations/archive50_selection.jsonl
data/annotations/archive50_expansion_report.json
```

Verify the synchronized server files:

```bash
python -m experiments.validate_annotations data/annotations/test_tracks.final_public.jsonl
python -c "import csv,json,pathlib; e=[json.loads(x) for x in pathlib.Path('data/annotations/test_entities.final_public.jsonl').open(encoding='utf-8') if x.strip()]; s=list(csv.DictReader(pathlib.Path('data/ships.csv').open(encoding='utf-8-sig'))); print('known_entities=',sum(x.get('known_or_unknown')=='known' for x in e)); print('known_identities=',len({x.get('hull_number') for x in e if x.get('known_or_unknown')=='known'})); print('ship_identities=',len({x.get('hull_number') for x in s}))"
```

Expected values are `known_entities=56`, `known_identities=50`, and `ship_identities=50`.

### Identity decision mode switch

`experiment.visual_archive.identity_decision_mode` controls whether text archive matching participates in identity decisions:

```yaml
experiment:
  visual_archive:
    identity_decision_mode: fused
```

Supported modes:

- `fused`: preserves the existing exact-hull, semantic-text, and visual evidence flow;
- `visual_only`: VLM hull text, descriptions, and structured attributes are retained in `recognition.jsonl` and reports, but `ships.csv` exact and semantic archive lookup is skipped. Identity candidates and terminal states come only from visual retrieval, temporal visual consistency, and the learned policy.

For the visual-only experiment, copy the final server configuration and change only this field:

```bash
cp config.final_public.yaml config.final_public.visual_only.yaml
```

```yaml
experiment:
  visual_archive:
    identity_decision_mode: visual_only
```

Verify the effective mode and backend before running:

```bash
python -c "from config import load_config; c=load_config('config.final_public.visual_only.yaml'); v=c['experiment']['visual_archive']; print('mode=',v['identity_decision_mode']); print('backend=',v['backend'])"
```

Expected values for the DINOv2 visual-only experiment are `mode=visual_only` and `backend=dinov2`. Keep the normal experiment switches `--archive on --visual-archive on --visual-decision on --visual-open-set on`; the mode itself prevents text archive identity matching.

Policy data and weights collected in `fused` mode must not be reused. Recollect `policy_train` and retrain with the visual-only config because text structure features are explicitly zeroed and `visual_only_mode=1` is added to the learned-policy input.

### DINOv2 guarded open-set rerun for the 50-identity archive

The 50-identity audit showed that a low Top-1/Top-2 margin was incorrectly treated as evidence that a vessel was outside the archive. The guarded flow now uses these semantics:

- one low visual score: `uncertain`, then reobserve;
- repeated low visual scores: `out_of_archive`;
- low visual margin: `uncertain`, never direct rejection;
- no text candidate plus a strong visual candidate: provisional `uncertain`;
- the same strong visual candidate in at least two observations: visual-consistency confirmation.
- repeated strong visual evidence may resolve a conflicting text candidate;
- a learned `stop_out_of_archive` action commits the tracker state instead of only logging `monitor_unknown`.
- in `visual_only`, a learned `stop_known` action becomes `confirm_visual_identity` when the same visual candidate is observed at least twice; it commits `structure_verified` and records `learned_visual_policy` evidence.
- in `visual_only`, a learned `stop_out_of_archive` action is executable only after `min_out_of_archive_observations` visual observations; this blocks one-frame false rejection while leaving the learned open-set decision active after repeated evidence. The deterministic visual open-set override still requires repeated low-score observations.

The previous visual-only run can therefore show many `learned_policy_action=stop_known` rows while still ending in `review_requested`: the old runtime had no executable confirmation action and redirected that prediction to another query. This is a runtime integration error, not evidence that the visual archive retrieval failed. Rebuild the visual-only policy data and model after this fix.

The DINOv2 visual-only gates are:

```yaml
experiment:
  visual_archive:
    visual_only_confirmation_enabled: true
    visual_only_min_observations: 2
    visual_only_min_score: 0.70
    visual_only_min_margin: 0.03
    visual_only_strong_score: 0.90
    visual_only_reject_score: 0.70
    visual_only_reject_margin: 0.08
    min_out_of_archive_observations: 2
```

The learned policy uses the same evidence gates as the runtime. A repeated
candidate cannot confirm an identity unless it passes the acceptance score and
margin. A learned out-of-archive action cannot reject a target unless the
visual score and margin are both weak after repeated observations; otherwise
the policy must query again or request review.

These values are initialization values. Any calibration used in the paper must use `policy_train`, never `test`.

After synchronizing the code and archive, recollect policy data because the corrected state transitions change the training distribution:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment \
  --manifest data/annotations/manifest.jsonl \
  --config config.final_public.yaml \
  --split policy_train \
  --run-name policy_collect_fixed_archive50_visual_guard_v3 \
  --memory-mode full --policy-mode fixed_interval \
  --network-profile real \
  --quality on --ledger on --archive on --experience off --risk off \
  --visual-archive on --visual-decision on --visual-open-set on \
  --entity-reconciliation on --max-frames 0
```

Build the corrected policy datasets:

```bash
python -m experiments.build_policy_dataset \
  --run-dir experiment_outputs/final_public_v1/policy_collect_fixed_archive50_visual_guard_v3 \
  --annotations data/annotations/policy_train_entities.jsonl \
  --output data/policy/policy_train_dataset_archive50_visual_guard_v3.jsonl \
  --known-visual-score 0.70 --known-visual-margin 0.03 \
  --known-visual-strong-score 0.90 \
  --unknown-visual-score 0.70 --unknown-visual-margin 0.08 \
  --min-visual-rejection-observations 2

python -m experiments.build_policy_utility_dataset \
  --dataset data/policy/policy_train_dataset_archive50_visual_guard_v3.jsonl \
  --output data/policy/policy_train_utility_archive50_visual_guard_v3.jsonl \
  --query-cost 0.12 --latency-cost 0.04 --defer-cost 0.08 --escalation-cost 0.35 \
  --false-accept-penalty 1.00 --false-reject-penalty 1.20 \
  --wrong-known-penalty 1.10 --uncertainty-bonus 0.30 \
  --known-visual-score 0.70 --known-visual-margin 0.03 \
  --known-visual-strong-score 0.90 \
  --unknown-visual-score 0.70 --unknown-visual-margin 0.08 \
  --min-visual-rejection-observations 2
```

Train a new MLP policy. Do not reuse weights trained before the archive expansion or before this state fix:

```bash
python -m experiments.train_mlp_utility_policy \
  --dataset data/policy/policy_train_utility_archive50_visual_guard_v3.jsonl \
  --output models/policy/mlp_utility_archive50_visual_guard_v3.json \
  --validation-fraction 0.2 --hidden-sizes 32,16 --activation relu \
  --dropout 0.1 --epochs 500 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --balance sqrt --ranking-weight 0.25 --ranking-margin 0.15
```

Run the formal test under a new output name:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_experiment \
  --manifest data/annotations/manifest.jsonl \
  --config config.final_public.yaml \
  --split test \
  --run-name main_mlp_utility_dino_archive50_visual_guard_v3 \
  --memory-mode full --policy-mode learned_utility \
  --policy-model models/policy/mlp_utility_archive50_visual_guard_v3.json \
  --network-profile real \
  --quality on --ledger on --archive on --experience off --risk off \
  --visual-archive on --visual-decision on --visual-open-set on \
  --entity-reconciliation on --max-frames 0
```

Evaluate and audit the new run:

```bash
python -m experiments.evaluate_entities \
  --run-dir experiment_outputs/final_public_v1/main_mlp_utility_dino_archive50_visual_guard_v3 \
  --annotations data/annotations/test_entities.final_public.jsonl

python -m experiments.audit_identity_errors \
  --run-dir experiment_outputs/final_public_v1/main_mlp_utility_dino_archive50_visual_guard_v3 \
  --entity-annotations data/annotations/test_entities.final_public.jsonl \
  --category known_false_rejection --print-limit 100

python -m experiments.audit_archive_tendency \
  --run-dir experiment_outputs/final_public_v1/main_mlp_utility_dino_archive50_visual_guard_v3 \
  --annotations data/annotations/test_entities.final_public.jsonl --top 50
```
