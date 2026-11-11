"""Deterministic regression checks for evidence feedback and durable workflows."""

import json

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_ollama")
pytest.importorskip("langgraph.checkpoint.sqlite")
from langchain.messages import AIMessage

from opsdesk import agent, agent_tools, workflow
from opsdesk.agent_tools import InvestigationInput
from opsdesk.retrieval import VectorRetriever

# Reuse the existing synthetic dataset and scripted LangChain model.
from test_agent import dataset, task, collection, report, use_model, tool_call


def inputs(event_id="EV-002", budget=16):
    return InvestigationInput.model_validate(
        task().model_dump(exclude={"task"})
        | {"max_model_calls": budget, "event_id": event_id}
    )


def route(mode="investigate"):
    return AIMessage(content=json.dumps({"task": mode}))


def envelope(*ids, missing=False, query=None):
    return AIMessage(content=json.dumps({
        "answer": {
            "status": "insufficient_evidence" if missing else "answered",
            "explanation": "可控模型输出，用于验证流程。",
            "evidence_ids": list(ids),
            "missing_information": ["具体检查步骤"] if missing else [],
        },
        "retrieval_request": {"missing_information": "具体检查步骤", "query": query} if query else None,
    }))


def test_tool_facts_reach_next_model_call(dataset, monkeypatch):
    settings, ids = dataset
    model = use_model(monkeypatch, [
        *collection("query_events", "search_knowledge"),
        report("event:EV-002", ids["cv-a-v2"]),
    ])
    result = agent.run_investigation(settings, task())
    second_input = json.loads(model.requests[1]["messages"][-1].content)
    observation = second_input["observations"]["query_events"]
    assert observation["status"] == "ok"
    assert observation["sources"]["event:EV-002"]["code"] == "E17"
    assert result["tool_trace"][0]["result"] == observation


@pytest.mark.parametrize("budget", [4, 5])
def test_auto_entry_counts_routing_in_budget(dataset, monkeypatch, budget):
    settings, ids = dataset
    use_model(monkeypatch, [
        route(), *collection("query_events", "search_knowledge"),
        report("event:EV-002", ids["cv-a-v2"]),
    ])
    result = agent.run_auto_investigation(settings, inputs(budget=budget))
    assert len(result["model_trace"]) == budget
    assert result["model_trace"][0]["phase"] == "route"
    assert result["status"] == ("answered" if budget == 5 else "budget_exceeded")


def test_supplement_runs_only_search_and_preserves_all_candidates(dataset, monkeypatch):
    settings, ids = dataset
    embedding_calls = []

    def embed(self, texts):
        embedding_calls.append(texts)
        return [[1.0, 0.0] if len(embedding_calls) == 1 else [0.0, 1.0] for _ in texts]

    monkeypatch.setattr(agent_tools.OllamaEmbedder, "embed_texts", embed)
    use_model(monkeypatch, [
        route(), *collection("query_events", "search_knowledge"),
        envelope("event:EV-002", missing=True, query="补充检查步骤"),
        AIMessage(content="", tool_calls=[tool_call("search_knowledge", topic="补充检查步骤")]),
        AIMessage(content="", tool_calls=[tool_call("CollectionDone", phase="collected")]),
        envelope("event:EV-002", ids["cv-a-v2"]),
    ])
    result = workflow.start_task(settings, inputs())
    assert result["status"] == "completed"
    assert result["supplements"] == 1
    assert [row["tool"] for row in result["bundle"]["tool_trace"]] == [
        "query_events", "search_knowledge", "search_knowledge",
    ]
    assert len(result["bundle"]["model_trace"]) == 8
    searches = [row for row in result["bundle"]["tool_trace"] if row["tool"] == "search_knowledge"]
    expected = set().union(*(set(row["result"]["sources"]) for row in searches))
    actual = {key for key in result["bundle"]["source"] if key.startswith("manual:")}
    assert actual == expected
    assert len(embedding_calls) == 2
    assert result["attempts"][1]["result"]["retrieval_request"]["query"] == "补充检查步骤"
    assert result["attempts"][-1]["result"]["answer"] == result["bundle"]["answer"]
    assert all(
        not ({"source", "observations", "model_trace", "tool_trace"} & attempt["result"].keys())
        for attempt in result["attempts"]
    )


def test_resume_reads_sqlite_and_clears_old_evidence(dataset, monkeypatch):
    settings, ids = dataset
    use_model(monkeypatch, [route(), *collection("query_events", "search_knowledge")])
    first = workflow.start_task(settings, inputs(event_id="EV-003"))
    assert first["status"] == "waiting_for_input"
    assert first["reason"] == "anchor_not_found"
    assert workflow.show_task(settings, first["task_id"])["status"] == "waiting_for_input"
    with pytest.raises(ValueError):
        workflow.resume_task(settings, first["task_id"], {"question": "错误字段"})
    with pytest.raises(ValueError):
        workflow.resume_task(settings, first["task_id"], {"event_id": "EV-003"})
    assert workflow.show_task(settings, first["task_id"])["corrections"] == 0
    use_model(monkeypatch, [
        *collection("query_events", "search_knowledge"),
        envelope("event:EV-002", ids["cv-a-v2"]),
    ])
    final = workflow.resume_task(settings, first["task_id"], {"event_id": "EV-002"})
    assert final["status"] == "completed"
    assert final["original_request"]["event_id"] == "EV-003"
    assert final["current_request"]["event_id"] == "EV-002"
    assert len(final["bundle"]["model_trace"]) == 8
    assert final["attempts"][0]["event_id"] == "EV-003"
    with pytest.raises(ValueError, match="没有等待"):
        workflow.resume_task(settings, first["task_id"], {"event_id": "EV-001"})


def test_missing_event_and_unclear_question_pause_before_tools(dataset, monkeypatch):
    settings, _ = dataset
    use_model(monkeypatch, [route()])
    result = workflow.start_task(settings, inputs(event_id=None))
    assert result["reason"] == "missing_event_id"
    assert not result["bundle"].get("tool_trace")
    use_model(monkeypatch, [route("needs_input")])
    result = workflow.start_task(settings, inputs())
    assert result["waiting_field"] == "question"
    use_model(monkeypatch, [route("events"), *collection("query_events"), envelope("window:summary")])
    result = workflow.resume_task(settings, result["task_id"], {"question": "有几次 Warning？"})
    assert result["status"] == "completed"
    assert [row["phase"] for row in result["bundle"]["model_trace"]].count("route") == 2


def test_format_repair_does_not_repeat_tools(dataset, monkeypatch):
    settings, ids = dataset
    use_model(monkeypatch, [
        route(), *collection("query_events", "search_knowledge"),
        envelope("event:EV-002", "manual:unknown"),
        envelope("event:EV-002", ids["cv-a-v2"]),
    ])
    result = workflow.start_task(settings, inputs())
    assert result["status"] == "completed"
    assert len(result["bundle"]["tool_trace"]) == 2
    assert result["bundle"]["validation_retries"] == 1


def test_budget_is_not_reset_after_resume(dataset, monkeypatch):
    settings, _ = dataset
    use_model(monkeypatch, [route(), *collection("query_events", "search_knowledge")])
    first = workflow.start_task(settings, inputs(event_id="EV-003", budget=4))
    assert first["status"] == "waiting_for_input"
    use_model(monkeypatch, [])
    final = workflow.resume_task(settings, first["task_id"], {"event_id": "EV-002"})
    assert final["status"] == "failed"
    assert final["reason"] == "budget_exceeded"
    assert len(final["bundle"]["model_trace"]) == 4


def test_second_supplement_request_is_rejected(dataset, monkeypatch):
    settings, _ = dataset
    retry = envelope("event:EV-002", missing=True, query="检查步骤")
    use_model(monkeypatch, [
        route(), *collection("query_events", "search_knowledge"), retry,
        *collection("search_knowledge"), retry.model_copy(), retry.model_copy(),
    ])
    result = workflow.start_task(settings, inputs())
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_report"
    assert result["supplements"] == 1
    assert len(result["bundle"]["tool_trace"]) == 3


def test_no_manual_ends_before_answer(dataset, monkeypatch):
    settings, _ = dataset
    embedding_calls = []

    def unexpected_embedding(self, texts):
        embedding_calls.append(texts)
        raise AssertionError("无适用手册时不应调用向量模型")

    monkeypatch.setattr(agent_tools.OllamaEmbedder, "embed_texts", unexpected_embedding)
    use_model(monkeypatch, [route(), *collection("query_events", "search_knowledge")])
    request = inputs(event_id="EV-005").model_copy(update={"device_id": "PK-02"})
    result = workflow.start_task(settings, request)
    assert result["status"] == "insufficient_evidence"
    assert result["reason"] == "no_applicable_manuals"
    assert not result["interrupts"]
    assert embedding_calls == []





def test_manifest_detects_missing_extra_changed_and_duplicate(dataset):
    settings, _ = dataset
    from opsdesk.knowledge import load_manuals
    from test_agent import EXAMPLES

    expected = load_manuals(EXAMPLES / "retrieval" / "manuals")
    retriever = VectorRetriever.load(
        settings.data_dir / "knowledge" / "retrieval-v1.npz", agent_tools.OllamaEmbedder().model
    )
    assert retriever.audit_chunks(expected)["ok"]
    assert retriever.audit_chunks(expected[:-1])["unexpected"] == [expected[-1].chunk_id]
    extra = expected[0].model_copy(update={"chunk_id": "missing:chunk"})
    assert retriever.audit_chunks([*expected, extra])["missing"] == ["missing:chunk"]
    changed = expected[0].model_copy(update={"text": expected[0].text + "\n修改过的正文"})
    assert retriever.audit_chunks([changed, *expected[1:]])["changed"] == [expected[0].chunk_id]
    with pytest.raises(ValueError, match="重复"):
        retriever.audit_chunks([*expected, expected[0]])
