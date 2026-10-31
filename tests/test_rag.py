import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opsdesk.knowledge import load_manual
from opsdesk.llm import GenerationError
from opsdesk.rag import explain_event
from opsdesk.retrieval import SearchHit
from opsdesk.schemas import EventQuery


ROOT = Path(__file__).resolve().parents[1]


def run_case(monkeypatch, hits, evidence_ids):
    report = {
        "device": {"model": "CV-A"},
        "events": [
            {
                "event_id": "EV-002",
                "device_id": "CV-07",
                "occurred_at": "2026-10-19T07:10:00+08:00",
                "code": "E17",
                "level": "warning",
                "message": "设备上报E17，尚未核对原因",
                "source_url": "/api/sources/test/events.json",
                "source_index": 1,
            }
        ],
        "data_quality": {"potentially_incomplete": False},
    }

    monkeypatch.setattr(
        "opsdesk.rag.query_events",
        lambda repository, device_id, query: report,
    )

    calls = []

    def search(**kwargs):
        assert kwargs["device_model"] == "CV-A"
        assert (
            kwargs["at"].isoformat()
            == "2026-10-19T07:10:00+08:00"
        )
        return hits

    def generate(messages, schema):
        calls.append(messages)
        return json.dumps({
            "status": "answered",
            "explanation": "E17 不能单独证明硬件损坏。",
            "evidence_ids": evidence_ids,
            "missing_information": ["上报通道状态"],
        })

    result = explain_event(
        repository=None,
        query=EventQuery(
            start="2026-10-19T07:00:00+08:00",
            end="2026-10-19T08:00:00+08:00",
        ),
        device_id="CV-07",
        event_id="EV-002",
        question="这说明硬件损坏了吗？",
        retriever=SimpleNamespace(search=search),
        embedder=None,
        llm=SimpleNamespace(generate=generate),
    )

    return result, calls


def manual_hit():
    chunk = load_manual(
        ROOT / "examples/retrieval/manuals/cv-a-v2.md"
    )
    return SearchHit(score=0.8, chunk=chunk)


def test_answer_preserves_sources(monkeypatch):
    result, calls = run_case(
        monkeypatch,
        [manual_hit()],
        ["event:EV-002", "manual:1"],
    )

    assert result["status"] == "answered"
    assert len(calls) == 1
    assert (
        result["sources"]["manual:1"]["chunk"]["doc_id"]
        == "cv-a-v2"
    )


def test_no_manual_skips_generation(monkeypatch):
    result, calls = run_case(monkeypatch, [], [])

    assert result["status"] == "no_applicable_manuals"
    assert result["answer"] is None
    assert calls == []


def test_unknown_citation_is_rejected(monkeypatch):
    with pytest.raises(GenerationError, match="不存在"):
        run_case(
            monkeypatch,
            [manual_hit()],
            ["manual:999"],
        )