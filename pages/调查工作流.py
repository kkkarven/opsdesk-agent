import json
import sqlite3
from pathlib import Path

import streamlit as st

from opsdesk.agent_tools import InvestigationInput
from opsdesk.config import Settings
from opsdesk.schemas import EventQuery
from opsdesk.workflow import resume_task, show_task, start_task


st.set_page_config(page_title="OpsDesk 调查工作流", layout="wide")
st.title("设备调查与恢复")
st.caption("自然语言问题 → 事件与手册 → 引用回答。暂停后可用任务编号继续。")

settings = Settings.from_env()
mode = st.sidebar.radio("数据来源", ["本地实时", "实跑回放"])

if mode == "实跑回放":
    path = Path(__file__).resolve().parents[1] / "docs/demo/workflow-replay.json"
    if not path.is_file():
        st.info("尚未保存工作流回放。可以先运行本地任务。")
        st.stop()

    replay = json.loads(path.read_text(encoding="utf-8"))
    st.caption("这里展示精简前保存的本地模型实跑记录。")
    scenarios = replay["scenarios"]
    labels = [item["label"] for item in scenarios]
    label = st.selectbox("查看运行", labels)
    result = scenarios[labels.index(label)]["result"]

else:
    with st.sidebar.form("new_task"):
        device = st.text_input("设备编号", "CV-07")
        event = st.text_input("事件编号，可留空", "EV-002")
        start = st.text_input("开始时间", "2026-10-19T07:00:00+08:00")
        end = st.text_input("结束时间", "2026-10-19T08:00:00+08:00")
        question = st.text_area(
            "调查问题", "结合事件和手册，E17 能证明硬件损坏吗？"
        )
        submitted = st.form_submit_button("开始调查")

    with st.sidebar.form("existing_task"):
        task_id = st.text_input("已有任务编号")
        opened = st.form_submit_button("读取任务")

    try:
        if submitted:
            request = InvestigationInput(
                device_id=device,
                event_id=event.strip() or None,
                query=EventQuery(start=start, end=end),
                question=question,
                max_model_calls=16,
            )
            with st.spinner("正在调查…"):
                st.session_state["workflow_result"] = start_task(settings, request)

        if opened:
            st.session_state["workflow_result"] = show_task(
                settings, task_id.strip()
            )
    except (ValueError, OSError, sqlite3.Error) as error:
        st.error(str(error))

    result = st.session_state.get("workflow_result")
    if not result:
        st.info("提交一个问题，或输入已有任务编号。")
        st.stop()

st.code(result["task_id"], language=None)
status_labels = {
    "completed": "调查完成",
    "waiting_for_input": "等待补充",
    "insufficient_evidence": "证据不足",
    "failed": "执行失败",
}
st.subheader(status_labels.get(result["status"], result["status"]))

if result.get("reason"):
    st.caption(result["reason"])

if result["status"] == "waiting_for_input" and mode == "本地实时":
    field = result.get("waiting_field") or "event_id"
    with st.form("resume"):
        value = st.text_input(
            "补充调查问题" if field == "question" else "补充或更正事件编号"
        )
        resumed = st.form_submit_button("继续这项调查")

    if resumed:
        try:
            with st.spinner("正在恢复调查…"):
                st.session_state["workflow_result"] = resume_task(
                    settings, result["task_id"], {field: value}
                )
            st.rerun()
        except (ValueError, OSError, sqlite3.Error) as error:
            st.error(str(error))

attempts = result["attempts"]
bundle = result.get("bundle") or (attempts[-1] if attempts else {})
answer = bundle.get("answer")

if answer:
    st.write(answer["explanation"])
    st.caption("引用来源 · " + " · ".join(answer["evidence_ids"]))
    if answer["missing_information"]:
        st.write("仍缺少的信息", answer["missing_information"])

if bundle.get("error"):
    st.error(bundle["error"])

with st.expander("任务计划"):
    st.json(result.get("plan") or result["current_request"])

with st.expander("本轮证据"):
    for source_id, source in bundle.get("source", {}).items():
        st.write(source_id)
        if source.get("kind") == "manual":
            st.text(source["text"])
            st.caption(
                f"{source['source_path']} · 行 "
                f"{source['line_start']}–{source['line_end']}"
            )
            if source.get("text_truncated"):
                st.warning("提供给模型的正文已截断。")
        else:
            st.json(source)

with st.expander("工具与模型调用"):
    st.json({key: bundle.get(key, []) for key in ("tool_trace", "model_trace")})

with st.expander("各轮历史"):
    st.json(attempts)