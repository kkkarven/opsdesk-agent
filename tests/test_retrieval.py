from datetime import datetime
from pathlib import Path

import pytest

from opsdesk.embeddings import OllamaEmbedder
from opsdesk.knowledge import load_manuals
from opsdesk.retrieval import VectorRetriever

MANUAL_DIR = (Path(__file__).resolve().parents[1]/"examples"/"retrieval"/"manuals")

EVENT_TIME = datetime.fromisoformat(
    "2026-10-19T07:10:00+08:00"
)


@pytest.fixture
def retriever():
    chunks = load_manuals(MANUAL_DIR)

    vectors_by_doc = {
        "cv-a-v1": [1000.0, 0.0],
        "cv-b-v1": [1000.0, 0.0],
        "cv-a-v2": [8.0, 6.0],
        "cv-a-e22-v1": [10.0, 100.0],
        "cv-a-maintenance-v1": [0.0, 20.0],
    }

    vectors = [
        vectors_by_doc[chunk.doc_id]
        for chunk in chunks
    ]

    return VectorRetriever(
        chunks=chunks,
        vectors=vectors,
        embedding_model="bge-m3:567m",
    )

@pytest.fixture
def embedder(monkeypatch):
    instance = OllamaEmbedder()

    def fake_embed(texts):
        return [[3.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        instance,
        "embed_texts",
        fake_embed,
    )

    return instance

def test_filters_before_ranking_and_normalizes(retriever, embedder):
    hits = retriever.search(
        question="上报后没有收到确认",
        device_model="CV-A",
        at=EVENT_TIME,
        embedder=embedder,
        top_k=2,
    )

    assert [hit.chunk.doc_id for hit in hits] == [
        "cv-a-v2",
        "cv-a-e22-v1",
    ]
    assert hits[0].score == pytest.approx(0.8, abs=1e-6)

def test_top_k_does_not_exceed_candidates(retriever, embedder):
    hits = retriever.search(
        question="上报后没有收到确认",
        device_model="CV-A",
        at=EVENT_TIME,
        embedder=embedder,
        top_k=10,
    )

    assert len(hits) == 3
    assert {hit.chunk.doc_id for hit in hits} == {
        "cv-a-v2",
        "cv-a-e22-v1",
        "cv-a-maintenance-v1",
    }

def test_no_candidates_skips_model_call(
    retriever,
    embedder,
    monkeypatch,
):
    def unexpected_call(texts):
        pytest.fail("没有合法候选时不应调用模型")

    monkeypatch.setattr(
        embedder,
        "embed_texts",
        unexpected_call,
    )

    hits = retriever.search(
        question="查询资料",
        device_model="PK-A",
        at=EVENT_TIME,
        embedder=embedder,
    )

    assert hits == []

def test_rejects_different_embedding_model(retriever, embedder):
    embedder.model = "another-model"

    with pytest.raises(ValueError, match="模型不一致"):
        retriever.search(
            question="查询资料",
            device_model="CV-A",
            at=EVENT_TIME,
            embedder=embedder,
        )

def test_rejects_query_dimension_mismatch(
    retriever,
    embedder,
    monkeypatch,
):
    monkeypatch.setattr(
        embedder,
        "embed_texts",
        lambda texts: [[1.0, 0.0, 0.0]],
    )

    with pytest.raises(ValueError, match="维度"):
        retriever.search(
            question="查询资料",
            device_model="CV-A",
            at=EVENT_TIME,
            embedder=embedder,
        )