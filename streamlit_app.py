"""OpsDesk investigation console with recorded and local live modes."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import streamlit as st

from opsdesk.config import Settings
from opsdesk.embeddings import OllamaEmbedder
from opsdesk.llm import OllamaChat
from opsdesk.rag import explain_event
from opsdesk.retrieval import VectorRetriever
from opsdesk.schemas import EventQuery
from opsdesk.service import DeviceNotFound, query_events
from opsdesk.storage import Repository

REPLAY_PATH = Path(__file__).resolve().parent / "docs" / "demo" / "replay.json"
DISPLAY_TZ = timezone(timedelta(hours=8))
STATUS = {
    "answered": "已回答 · 有手册引用",
    "insufficient_evidence": "证据不足 · 需要补充信息",
    "no_applicable_manuals": "没有适用手册 · 已跳过生成",
}


def local_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")


def show_overview(report: dict) -> None:
    device = report["device"]
    summary = report["summary"]
    st.subheader(f"{device['device_id']} · {device['display_name']}")
    st.caption(f"型号 {device['model'] or '未知'}  /  站点 {device['site']}  /  界面时间 UTC+08:00")
    columns = st.columns(3, border=True)
    columns[0].metric("匹配事件", summary["event_count"])
    columns[1].metric("Warning 告警", summary["warning_count"])
    latest = summary["latest"]
    columns[2].metric("最近事件", latest["event_id"] if latest else "—")
    if latest:
        columns[2].caption(local_time(latest["occurred_at"]))

    quality = report["data_quality"]
    if quality["potentially_incomplete"]:
        st.warning(quality["note"])
        st.caption(
            f"设备相关坏记录 {quality['related_rejected_count']} 条 · "
            f"归属设备不明 {quality['unattributed_rejected_count']} 条"
        )
    if quality["unrecognized_levels"]:
        st.warning("发现未识别的事件等级：" + "、".join(quality["unrecognized_levels"]))
    if report["truncated"]:
        st.info(f"显示最近 {report['returned_count']} 条明细。上方统计覆盖完整查询范围。")


def show_events(report: dict) -> None:
    events = report["events"]
    if not events:
        st.info("该时间窗口内没有匹配的事件。")
        return

    rows = [
        {
            "事件": event["event_id"],
            "时间 (UTC+08)": local_time(event["occurred_at"]),
            "等级": event["level"],
            "代码": event["code"],
            "消息": event["message"],
        }
        for event in events
    ]
    start = datetime.fromisoformat(report["query"]["start"])
    duration = (datetime.fromisoformat(report["query"]["end"]) - start).total_seconds() / 60
    points = [
        {**row, "距起点分钟": (datetime.fromisoformat(event["occurred_at"]) - start).total_seconds() / 60}
        for event, row in zip(events, rows)
    ]
    colors = {"info": "#0284c7", "warning": "#d97706", "error": "#dc2626", "critical": "#9333ea"}
    levels = list(report["summary"]["level_counts"])
    chart = (
        alt.Chart(alt.Data(values=points))
        .mark_circle(size=155, opacity=0.9)
        .encode(
            x=alt.X(
                "距起点分钟:Q",
                title="距查询起点（分钟）",
                scale=alt.Scale(domain=[0, max(duration, 1)]),
                axis=alt.Axis(tickCount=7),
            ),
            y=alt.Y("等级:N", title=None),
            color=alt.Color(
                "等级:N",
                scale=alt.Scale(
                    domain=levels,
                    range=[colors.get(level, "#64748b") for level in levels],
                ),
                legend=None,
            ),
            tooltip=["事件:N", "时间 (UTC+08):N", "等级:N", "代码:N", "消息:N"],
        )
        .properties(height=145)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(rows, hide_index=True, width="stretch")
    with st.expander("查看原始事件记录"):
        event_id = st.selectbox("事件编号", [event["event_id"] for event in events], key="raw_event")
        event = next(event for event in events if event["event_id"] == event_id)
        st.caption(f"归档路径 {event['source_url']} · 归档索引 {event['source_index']}")
        st.json(event["raw_record"])


def show_explanation(result: dict, question: str) -> None:
    st.subheader(f"调查结果 · {result['event_id']}")
    st.text(question)
    answer = result["answer"]
    if answer is None:
        st.warning(STATUS[result["status"]])
        cited = []
    else:
        st.markdown(f"**{STATUS[result['status']]}**")
        st.text(answer["explanation"])
        cited = answer["evidence_ids"]
        st.caption("回答引用 · " + (" · ".join(cited) or "无"))
        if answer["missing_information"]:
            st.markdown("**还需补充**")
            for item in answer["missing_information"]:
                st.text("• " + item)

    st.markdown("#### 证据与来源")
    st.caption("手册按检索返回顺序排列。型号和有效时间由事件记录确定。")
    for source_id, source in result["sources"].items():
        citation = " · 已引用" if source_id in cited else ""
        if source["kind"] == "event":
            record = source["record"]
            with st.expander(f"{source_id} · SQLite 事件{citation}"):
                st.text(f"{record['code']} · {record['level']} · {local_time(record['occurred_at'])}")
                st.text(record["message"])
                st.caption(f"归档路径 {source['source_url']} · 归档索引 {source['source_index']}")
        else:
            chunk = source["chunk"]
            with st.expander(f"{source_id} · {chunk['title']} · {chunk['version']}{citation}"):
                valid_to = local_time(chunk["valid_to"]) if chunk["valid_to"] else "开放"
                st.caption(f"型号 {chunk['model']} · 有效期 {local_time(chunk['valid_from'])} 至 {valid_to}")
                st.text(chunk["text"])
                st.caption(
                    f"原文 {chunk['source_path']} · 第 {chunk['line_start']}–{chunk['line_end']} 行"
                    f" · 片段 {chunk['chunk_id']}"
                )


def live_report(capture: dict) -> None:
    settings = Settings.from_env()
    repository = Repository(settings)
    with st.sidebar.form("query"):
        device_id = st.text_input("设备 ID", "CV-07").strip()
        start = st.text_input("开始时间（ISO 8601，含时区）", capture["start"])
        end = st.text_input("结束时间（ISO 8601，含时区）", capture["end"])
        submitted = st.form_submit_button("查询设备事件", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("live_report", None)
        st.session_state.pop("live_explanation", None)
        try:
            query = EventQuery(start=start, end=end)
            if not settings.db_path.is_file():
                raise ValueError("未找到 SQLite 数据库。请先运行 opsdesk import，或检查 OPSDESK_DATA_DIR。")
            st.session_state["live_report"] = query_events(repository, device_id, query)
        except DeviceNotFound:
            st.error(f"找不到设备 {device_id}。")
        except (ValueError, OSError, sqlite3.Error) as error:
            st.error(str(error))

    report = st.session_state.get("live_report")
    if report is None:
        st.info("从左侧提交设备和时间范围。查询完成后可选择一条事件进行调查。")
        return
    show_overview(report)
    events_tab, analysis_tab = st.tabs(["事件时间线", "RAG 调查"])
    with events_tab:
        show_events(report)
    with analysis_tab:
        if not report["events"]:
            st.info("当前查询没有可调查的事件。")
            return
        with st.form("explain"):
            event_id = st.selectbox(
                "目标事件",
                [event["event_id"] for event in report["events"]],
                index=next((i for i, item in enumerate(report["events"]) if item["event_id"] == "EV-002"), 0),
            )
            question = st.text_area("调查问题", capture["scenarios"][0]["question"], max_chars=1500).strip()
            investigate = st.form_submit_button("运行本地 RAG", type="primary")
        if investigate:
            st.session_state.pop("live_explanation", None)
            try:
                if not question:
                    raise ValueError("请输入调查问题。")
                embedder = OllamaEmbedder()
                snapshot = settings.data_dir / "knowledge" / "retrieval-v1.npz"
                if not snapshot.is_file():
                    raise ValueError("未找到向量索引。请先运行 opsdesk knowledge-build。")
                with st.spinner("正在检索适用手册并调用本地模型…"):
                    retriever = VectorRetriever.load(snapshot, expected_model=embedder.model)
                    result = explain_event(
                        repository, EventQuery.model_validate(report["query"]),
                        report["device"]["device_id"], event_id, question,
                        retriever, embedder, OllamaChat(),
                    )
                st.session_state["live_explanation"] = (result, question)
            except (DeviceNotFound, OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                st.error(f"调查执行失败：{error}")
        if "live_explanation" in st.session_state:
            show_explanation(*st.session_state["live_explanation"])


def main() -> None:
    st.set_page_config(page_title="OpsDesk · 设备告警调查", page_icon="🔎", layout="wide")
    capture = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    st.sidebar.title("OpsDesk")
    mode = st.sidebar.radio("运行模式", ["案例回放", "本地实时"])
    st.sidebar.divider()
    st.sidebar.caption("SQLITE 事件 → 适用手册 → FAISS → QWEN → 引用校验")
    st.title("设备告警调查")
    if mode == "本地实时":
        st.caption("本地实时 · 使用已导入的 SQLite 数据、向量索引及 Ollama。")
        live_report(capture)
        return

    scenarios = {scenario["title"]: scenario for scenario in capture["scenarios"]}
    scenario = scenarios[st.sidebar.selectbox("调查案例", list(scenarios))]
    st.sidebar.markdown("**回放来源**")
    st.sidebar.caption(f"Embedding · {capture['embedding_model']}")
    st.sidebar.caption(f"LLM · {capture['generation_model']}")
    st.sidebar.caption(f"{local_time(capture['start'])} → {local_time(capture['end'])}")
    st.caption("阶段二 · 案例回放 · 模拟设备数据及本地模型运行记录")
    report = scenario["report"]
    show_overview(report)
    events_tab, analysis_tab = st.tabs(["事件时间线", "RAG 调查"])
    with events_tab:
        show_events(report)
    with analysis_tab:
        show_explanation(scenario["explanation"], scenario["question"])


if __name__ == "__main__":
    main()
