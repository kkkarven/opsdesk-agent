# OpsDesk Agent

设备事件数据服务，后续接入 Agent。

## 当前实现

- JSON 设备与事件导入、输入校验。
- 原始文件归档、坏记录隔离及原因保存。
- SQLite 持久化、重复导入去重、同编号冲突保护。
- 按设备、时间范围和等级查询。
- 事件统计、最近事件、数据质量提示。
- 原始来源访问、HTTP 接口与交互式文档。
- 请求编号、状态和耗时日志。
- 自动化功能测试。

设备、事件和故障信息均为模拟数据。
当前尚未实现模型调用、资料检索和工单功能。

## 安装

使用 WSL Ubuntu，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e ".[dev]"
```

Python 最低声明版本为 3.12；实际验收版本见下方记录。

## 导入与启动

```bash
opsdesk --data-dir data import examples/normal

OPSDESK_DATA_DIR="$PWD/data" python -m uvicorn opsdesk.api:create_app --factory --host 127.0.0.1 --port 8000
```

交互文档：http://127.0.0.1:8000/docs

查询示例：

- 设备：CV-07
- 开始：2026-10-19T07:00:00+08:00
- 结束：2026-10-19T08:00:00+08:00
- 预期：3条事件、2条warning，最近事件EV-006。

## 检查

```bash
mkdir -p docs
python -m pytest --junitxml=docs/stage1-tests.xml
python -m pip check
```

测试使用临时数据库，不使用日常运行数据目录。

## 模块关系

- cli.py：初始化和导入命令。
- ingestion.py：归档、校验、导入及隔离。
- storage.py：数据库连接、建表和读取。
- service.py：统计与业务结果组织。
- api.py：HTTP 参数和错误处理。
- request_logging.py：请求追踪。
- schemas.py：输入结构与检查规则。
- config.py：数据目录和数据库路径。

## 当前边界

- 仅验证模拟输入，不代表真实设备诊断有效。
- 接口用于本地开发，尚未实现用户认证和权限管理。
- 坏记录不计入正常统计，存在相关坏记录时提示可能不完整。
- limit只限制明细数量，不限制摘要统计范围。

## 第一阶段验收记录

自动化测试：12 项通过。
依赖检查：通过，无依赖冲突。
独立虚拟环境复现：安装成功，12 项测试通过。
已知事项：测试依赖存在 httpx 弃用警告，当前不影响运行。