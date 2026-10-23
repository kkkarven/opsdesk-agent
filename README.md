# OpsDesk：设备事件数据清理与导入

[![Data ingestion checks](https://github.com/kkkarven/opsdesk-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kkkarven/opsdesk-agent/actions/workflows/ci.yml)

**当前版本 v0.1.0：仅完成设备事件数据的初步校验、清理、原文归档与 SQLite 导入。项目持续开发中，Agent、RAG、故障分析与工单功能尚未实现。**

OpsDesk 计划逐步发展为一个可追溯的运维 Agent。目前先完成输入数据基础：读取设备和事件 JSON，保留原始文件，校验记录，将合法数据与问题记录分开保存，使后续查询、检索与 Agent 有明确的数据来源。

样例中的设备、事件、时间与故障描述均为合成数据，见 [模拟数据说明](examples/SIMULATED.md)。当前结果验证软件处理行为，不代表真实工业诊断效果。

## 已实现的范围

| 能力 | 当前行为 |
|---|---|
| 输入校验 | 校验设备编号、必需字段、字符串长度、带时区的时间；检查事件引用的设备 |
| 时间规范化 | 合法时间转换为固定精度 UTC 文本，原始写法单独保留 |
| 原文归档 | 按文件内容生成 SHA256 批次标识，保留两份输入文件的原始字节 |
| 合法数据入库 | SQLite 保存设备、事件、导入批次及来源数组位置 |
| 异常记录隔离 | 保留原始记录、错误原因、批次与创建时间，导出到 quarantine 目录 |
| 重复与冲突处理 | 相同批次不重复导入；同编号相同内容跳过；同编号不同内容隔离并保留原记录 |
| 事务与命令入口 | 数据库批量写入使用事务；命令返回 JSON 结果和明确失败退出状态 |

这里的“清理”主要指校验、规范化、去重和隔离，不会自动修复坏时间戳、填补缺失值或删除原始信息。没有单独实现日志系统、数据查询 API、Web 页面或模型调用。

## 快速开始

需要 Python 3.12 或以上版本。当前本地验证环境为 WSL Ubuntu / Python 3.14.4；GitHub Actions 配置 Python 3.12 和 3.14 两组检查，实际状态以页面结果为准。

```bash
git clone https://github.com/kkkarven/opsdesk-agent.git
cd opsdesk-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.lock -e .
opsdesk --data-dir data import examples/normal
```

首次正常导入应接收3台设备、6条事件，拒绝0条记录。输出包含：

```json
{
  "accepted_devices": 3,
  "accepted_events": 6,
  "skipped_duplicates": 0,
  "rejected_records": 0,
  "already_imported": false
}
```

实际输出还包含 `batch_id`。再执行同一命令时，`already_imported` 为 `true`，不会重复写入；其他计数字段返回该批次第一次导入的报告，不是本次新增量。

## 查看异常处理

将坏时间戳样例导入独立目录：

```bash
opsdesk --data-dir data-invalid import examples/invalid_event
```

首次应接收3台设备、5条事件，拒绝1条记录。`EV-002` 的 `occurred_at` 为 `INVALID_TIMESTAMP`，原文与错误原因保存在 `data-invalid/quarantine/`。正常样例没有坏记录，因此 `data/quarantine/` 为空属于预期行为。

不同场景使用独立数据库，避免同编号但不同内容的输入触发冲突处理。

## 输入与输出

输入目录必须包含 `devices.json` 和 `events.json`，两个文件顶层都为 JSON 数组。字段示例见仓库自带样例和 `src/opsdesk/schemas.py`。

```text
data/
  opsdesk.sqlite3
  imports/<batch_id>/
    devices.json
    events.json
  quarantine/<issue_id>.json
```

- `imports` 保存输入原始字节；数据库中的 `raw_json` 保存解析后的完整原始对象。
- `batch_id` 关联输入批次，`source_index` 表示从0开始的源数组位置。
- 时间统一为 UTC，原始时间和额外字段仍保留在原始对象中。
- 单条坏记录被隔离，其他合法记录继续导入；整个文件格式错误则拒绝该批导入，已归档原文保留。
- 系统级数据库或磁盘错误明确失败。数据库提交后的异常文件导出失败，可通过重复执行相同导入补回。

## 检查

```bash
python -m unittest discover -s tests -v
python -m pip check
```

测试使用临时目录、临时数据库和公开模拟样例，不修改日常运行数据，不访问模型 API。覆盖正常导入、原文一致性、重复导入、坏时间戳、未知设备、编号冲突、时间规范化、额外字段保留、异常副本恢复以及命令入口。

本版验证记录见 [发布验证说明](docs/validation.md)。

## 代码结构

```text
src/opsdesk/
  config.py       数据位置
  schemas.py      输入校验规则
  storage.py      SQLite表结构、连接与事务
  ingestion.py    归档、校验、导入与异常导出
  cli.py          命令入口
tests/            自动功能检查
examples/         合成输入
```

处理链：`CLI → 配置与数据库初始化 → 原文归档 → 数据校验 → 合法数据/问题记录分流 → 导入报告`。

## 当前限制与后续计划

- 目前是文件批次导入工具，仅处理设备与事件 JSON，不具备任意数据清洗能力。
- 设备资料更新与事件更正尚无专用流程；同编号不同内容按冲突隔离。
- 样例规模很小，未进行大规模数据吞吐或高并发测试。
- 尚未接入 LLM、Embedding、向量检索、Agent 工具调用或工单流程，没有对应效果指标。
- 下一步计划实现设备/时间窗口查询，再加入带来源的资料检索，最后接入 Agent 与需确认的工单操作。

上述后续能力是规划，不作为当前成果。项目开发过程中使用 AI 辅助提供参考实现、解释、排错与发布检查；当前仓库保留开发者手写复现的模块与注释。

## License

[MIT](LICENSE)
