from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from opsdesk.embeddings import OllamaEmbedder
from opsdesk.knowledge import load_manuals
from opsdesk.retrieval import VectorRetriever

MANUAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "retrieval"
    / "manuals"
)

MODEL = "bge-m3:567m"
QUESTION = "上报后没有收到确认"
EVENT_TIME = datetime.fromisoformat(
    "2026-10-19T07:10:00+08:00"
)

def make_retriever():
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
        embedding_model=MODEL,
    )

def test_round_trip_without_reembedding(tmp_path, monkeypatch):
    original = make_retriever()
    path = tmp_path / "knowledge.npz"
    original.save(path)

    def forbidden_embedding(self, texts):
        pytest.fail("加载快照不应调用 Embedding 服务")

    monkeypatch.setattr(
        OllamaEmbedder,
        "embed_texts",
        forbidden_embedding,
    )

    restored = VectorRetriever.load(
        path,
        expected_model=MODEL,
    )

    embedder = OllamaEmbedder()
    requests = []
    def query_embedding(texts):
        requests.append(texts)
        return [[3.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        embedder,
        "embed_texts",
        query_embedding,
    )

    before = original.search(
        QUESTION,
        device_model="CV-A",
        at=EVENT_TIME,
        embedder=embedder,
        top_k=3,
    )
    after = restored.search(
        QUESTION,
        device_model="CV-A",
        at=EVENT_TIME,
        embedder=embedder,
        top_k=3,
    )

    assert requests == [[QUESTION], [QUESTION]]

    assert [hit.chunk.model_dump() for hit in after] == [
        hit.chunk.model_dump() for hit in before
    ]
    assert [hit.score for hit in after] == pytest.approx(
        [hit.score for hit in before],
        abs=1e-6,
    )

def test_rejects_different_model(tmp_path):
    path = tmp_path / "knowledge.npz"
    make_retriever().save(path)

    with pytest.raises(ValueError, match="模型"):
        VectorRetriever.load(
            path,
            expected_model="another-model",
        )

def test_detects_changed_vector_rows(tmp_path):
    path = tmp_path / "knowledge.npz"
    make_retriever().save(path)

    with np.load(path, allow_pickle=False) as saved:
        metadata = saved["metadata"].copy()
        vectors = saved["vectors"].copy()
        checksum = saved["checksum"].copy()

    vectors[[0, 1]] = vectors[[1, 0]]

    np.savez(
        path,
        metadata=metadata,
        vectors=vectors,
        checksum=checksum,
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="完整性校验"):
        VectorRetriever.load(path, expected_model=MODEL)

def test_failed_save_preserves_existing_snapshot(
    tmp_path,
    monkeypatch,
):
    retriever = make_retriever()
    path = tmp_path / "knowledge.npz"
    retriever.save(path)
    previous_bytes = path.read_bytes()

    def broken_save(stream, **kwargs):
        stream.write(b"incomplete")
        raise OSError("模拟写入失败")

    monkeypatch.setattr(
        "opsdesk.retrieval.np.savez",
        broken_save,
    )

    with pytest.raises(OSError, match="模拟写入失败"):
        retriever.save(path)

    assert path.read_bytes() == previous_bytes

    