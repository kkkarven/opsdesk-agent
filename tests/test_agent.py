import json
from pathlib import Path

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_ollama")

from langchain.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama
from pydantic import Field

from opsdesk import agent, agent_tools
from opsdesk.agent_tools import InvestigationRequest, build_tools
from opsdesk.config import Settings
from opsdesk.ingestion import import_dataset
from opsdesk.knowledge import load_manuals
from opsdesk.retrieval import VectorRetriever
from opsdesk.schemas import EventQuery


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EVENT_ID = "event:EV-002"


class ScriptedChat(ChatOllama):
    """Use the real LangChain graph with predefined model responses."""

    replies: list[AIMessage]
    requests: list[dict] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = len(self.requests)
        assert index < len(self.replies), "Unexpected model request"
        self.requests.append({"messages": messages, **kwargs})
        return ChatResult(generations=[ChatGeneration(message=self.replies[index])])


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    normal = EXAMPLES / "normal"
    (source / "devices.json").write_bytes((normal / "devices.json").read_bytes())
    events = json.loads((normal / "events.json").read_text(encoding="utf-8"))
    events.append({
        "event_id": "EV-HISTORICAL",
        "device_id": "CV-07",
        "occurred_at": "2026-09-20T07:10:00+08:00",
        "code": "E17",
        "level": "warning",
        "message": "Historical test event",
    })
    (source / "events.json").write_text(json.dumps(events), encoding="utf-8")
    settings = Settings(tmp_path / "data")
    import_dataset(source, settings)

    chunks = load_manuals(EXAMPLES / "retrieval" / "manuals")
    ranking = {
        "cv-a-v2": [1.0, 0.0],
        "cv-a-maintenance-v1": [1.0, 1.0],
    }
    retriever = VectorRetriever(
        chunks=chunks,
        vectors=[ranking.get(chunk.doc_id, [0.0, 1.0]) for chunk in chunks],
        embedding_model=agent_tools.OllamaEmbedder().model,
    )
    retriever.save(settings.data_dir / "knowledge" / "retrieval-v1.npz")
    monkeypatch.setattr(
        agent_tools.OllamaEmbedder, "embed_texts",
        lambda self, texts: [[1.0, 0.0] for _ in texts],
    )
    return settings, {chunk.doc_id: f"manual:{chunk.doc_id}:{chunk.line_start}" for chunk in chunks}


def task(mode="investigate", device="CV-07", event_id="EV-002", budget=5, day="2026-10-19"):
    return InvestigationRequest(
        task=mode,
        device_id=device,
        event_id=event_id if mode != "events" else None,
        query=EventQuery(start=f"{day}T07:00:00+08:00", end=f"{day}T08:00:00+08:00"),
        question="检查当前范围内的事件或适用手册。",
        max_model_calls=budget,
    )


def tool_call(name, **args):
    return {"name": name, "args": args, "id": f"test-{name}", "type": "tool_call"}


def collection(*tools):
    args = {"search_knowledge": {"topic": "E17"}}
    return [
        *(AIMessage(content="", tool_calls=[tool_call(name, **args.get(name, {}))])
          for name in tools),
        AIMessage(content="", tool_calls=[tool_call("CollectionDone", phase="collected")]),
    ]


def report(*evidence_ids):
    return AIMessage(content=json.dumps({
        "status": "answered",
        "explanation": "Scripted report",
        "evidence_ids": list(evidence_ids),
        "missing_information": [],
    }))


def use_model(monkeypatch, replies):
    model = ScriptedChat(model="scripted", replies=replies)
    monkeypatch.setattr(agent, "ChatOllama", lambda **kwargs: model)
    return model


@pytest.mark.parametrize(
    ("mode", "tools", "expected_phases"),
    [
        ("events", ("query_events",), ["collect", "receipt", "report"]),
        ("knowledge", ("search_knowledge",), ["collect", "receipt", "report"]),
        ("investigate", ("query_events", "search_knowledge"), ["collect", "collect", "receipt", "report"]),
    ],
)
def test_agent_routes_tools_and_preserves_candidates(dataset, monkeypatch, mode, tools, expected_phases):
    settings, ids = dataset
    manual = ids["cv-a-v2"]
    cited = {
        "events": ["window:summary"],
        "knowledge": [manual],
        "investigate": [EVENT_ID, manual],
    }[mode]
    model = use_model(monkeypatch, [*collection(*tools), report(*cited)])

    result = agent.run_investigation(settings, task(mode=mode))

    if mode == "investigate":
        first_input = json.loads(model.requests[0]["messages"][-1].content)
        assert first_input["call_one_of"] == ["query_events"]

        second_input = json.loads(model.requests[1]["messages"][-1].content)
        assert second_input["call_one_of"] == ["search_knowledge"]

        event_result = second_input["observations"]["query_events"]
        assert event_result["status"] == "ok"
        assert event_result["sources"][EVENT_ID]["code"] == "E17"

    assert result["status"] == "answered"
    assert result["answer"]["evidence_ids"] == cited
    assert [(row["tool"], row["status"]) for row in result["tool_trace"]] == [
        (name, "ok") for name in tools
    ]
    assert [row["phase"] for row in result["model_trace"]] == expected_phases
    assert result["model_trace"][-2]["request_tools"][0]["name"] == "CollectionDone"
    assert result["model_trace"][-1]["request_tools"] == []

    final_request = model.requests[-1]
    assert not final_request.get("tools")
    assert final_request["response_format"]["type"] == "json_schema"
    evidence = json.loads(final_request["messages"][-1].content)["sources"]
    if mode == "events":
        assert evidence["window:summary"]["event_count"] == 3
        assert evidence["window:summary"]["warning_count"] == 2
        assert evidence["window:summary"]["latest_event_id"] == "EV-006"
        assert evidence[EVENT_ID]["event_id"] == "EV-002"
    else:
        manual_candidates = [manual, ids["cv-a-maintenance-v1"]]
        assert [key for key in evidence if key.startswith("manual:")] == manual_candidates
        for key in manual_candidates:
            assert evidence[key]["text"] == result["source"][key]["text"]


@pytest.mark.parametrize(
    ("device", "event_id", "reason"),
    [
        ("CV-07", "EV-003", "anchor_not_found"),
        ("PK-02", "EV-005", "no_applicable_manuals"),
    ],
)
def test_evidence_shortfall_returns_reason_without_report(dataset, monkeypatch, device, event_id, reason):
    settings, _ = dataset
    use_model(monkeypatch, collection("query_events", "search_knowledge"))

    result = agent.run_investigation(settings, task(device=device, event_id=event_id))

    assert result["status"] == "insufficient_evidence"
    assert result["reason"] == reason
    assert result["answer"]["missing_information"]
    assert result["tool_trace"][-1]["status"] == reason
    assert [row["phase"] for row in result["model_trace"]] == ["collect", "collect", "receipt"]


def test_missing_required_tool_stops_before_report(dataset, monkeypatch):
    settings, _ = dataset
    use_model(monkeypatch, [
        AIMessage(content="", tool_calls=[tool_call("query_events")]),
        AIMessage(content="I will not look up the manual."),
    ])

    result = agent.run_investigation(settings, task())

    assert result["status"] == "invalid_route"
    assert "search_knowledge" in result["error"]
    assert result["answer"] is None
    assert [row["tool"] for row in result["tool_trace"]] == ["query_events"]
    assert all(row["phase"] == "collect" for row in result["model_trace"])


@pytest.mark.parametrize(
    ("citations", "expected_error"),
    [
        (("manual:unknown",), "未提供的来源"),
        ((), "必须引用这些来源"),
    ],
)
def test_bad_citations_are_rejected_after_one_retry(dataset, monkeypatch, citations, expected_error):
    settings, ids = dataset
    invalid = report(ids["cv-a-v2"], *citations)
    model = use_model(monkeypatch, [
        *collection("query_events", "search_knowledge"), invalid, invalid.model_copy(),
    ])

    result = agent.run_investigation(settings, task())

    assert result["status"] == "invalid_report"
    assert result["answer"] is None
    assert expected_error in result["error"]
    assert [row["phase"] for row in result["model_trace"]].count("report") == 2
    retry = json.loads(model.requests[-1]["messages"][-1].content)
    assert expected_error in retry["previous_validation_error"]


def test_model_budget_preserves_tool_trace(dataset, monkeypatch):
    settings, _ = dataset
    use_model(monkeypatch, [AIMessage(content="", tool_calls=[tool_call("query_events")])])

    result = agent.run_investigation(settings, task(budget=1))

    assert result["status"] == "budget_exceeded"
    assert result["answer"] is None
    assert len(result["model_trace"]) == 1
    assert [(row["tool"], row["status"]) for row in result["tool_trace"]] == [("query_events", "ok")]


def test_manual_evidence_preserves_expiry(dataset):
    settings, ids = dataset
    _, search = build_tools(
        settings, task(mode="knowledge", event_id="EV-HISTORICAL", day="2026-09-20")
    )

    result = search.invoke(tool_call("search_knowledge", topic="E17"))
    evidence = result.artifact["sources"][ids["cv-a-v1"]]

    assert evidence["valid_from"] == "2026-01-01T00:00:00+08:00"
    assert evidence["valid_to"] == "2026-10-01T00:00:00+08:00"


@pytest.mark.parametrize(
    ("budget", "expected_status"),
    [(4, "budget_exceeded"), (5, "answered")],
)
def test_auto_entry_counts_routing_in_budget(
    dataset, monkeypatch, budget, expected_status
):
    settings, ids = dataset
    use_model(monkeypatch, [
        AIMessage(content=json.dumps({"task": "investigate"})),
        *collection("query_events", "search_knowledge"),
        report(EVENT_ID, ids["cv-a-v2"]),
    ])

    inputs = agent_tools.InvestigationInput.model_validate(
        task(budget=budget).model_dump(exclude={"task"})
    )
    result = agent.run_auto_investigation(settings, inputs)

    assert result["status"] == expected_status
    assert len(result["model_trace"]) == budget
    assert result["model_trace"][0]["phase"] == "route"


def test_report_schema_constrains_evidence_ids_enum(dataset, monkeypatch):
    settings, ids = dataset
    model = use_model(monkeypatch, [
        *collection("query_events", "search_knowledge"),
        report(EVENT_ID, ids["cv-a-v2"]),
    ])

    result = agent.run_investigation(settings, task())

    assert result["status"] == "answered"
    schema = model.requests[-1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["evidence_ids"]["items"] == {
        "type": "string",
        "enum": [EVENT_ID, ids["cv-a-v2"], ids["cv-a-maintenance-v1"]],
    }


def test_retry_appends_previous_reply_and_instruction_at_bottom(dataset, monkeypatch):
    settings, ids = dataset
    bad_reply = report(EVENT_ID)
    good_reply = report(EVENT_ID, ids["cv-a-v2"])
    model = use_model(monkeypatch, [
        *collection("query_events", "search_knowledge"),
        bad_reply,
        good_reply,
    ])

    result = agent.run_investigation(settings, task())

    assert result["status"] == "answered"
    assert result["validation_retries"] == 1
    retry_messages = model.requests[-1]["messages"]
    assert isinstance(retry_messages[-2], AIMessage)
    assert retry_messages[-2].content == bad_reply.content
    retry_feedback = json.loads(retry_messages[-1].content)
    assert "至少一份相关手册" in retry_feedback["previous_validation_error"]
    assert ids["cv-a-v2"] in retry_feedback["instruction"]
