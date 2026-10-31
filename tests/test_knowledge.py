from datetime import datetime
from pathlib import Path

import pytest

from opsdesk.knowledge import (
    load_manual,
    load_manuals,
    select_applicable,
)

MANUAL_DIR = (
    Path(__file__).resolve().parents[1]/ "examples"/ "normal"/ "manuals"
)

def test_load_manual_preserves_content_and_source():
    path = MANUAL_DIR / "cv-a-v2.md"
    chunk = load_manual(path)

    assert chunk.doc_id == "cv-a-v2"
    assert chunk.model == "CV-A"
    assert chunk.version == "2"
    assert chunk.valid_to is None
    assert "不能单独证明硬件损坏" in chunk.text
    assert "缺少通道记录" in chunk.text

    lines = path.read_text(encoding="utf-8").splitlines()
    source_text = "\n".join(
        lines[chunk.line_start - 1:chunk.line_end]
    )

    assert chunk.text == source_text
    assert Path(chunk.source_path) == path.resolve()

@pytest.mark.parametrize(
    ("model", "at", "expected_ids"),
    [
        (
            "CV-A",
            "2026-09-30T23:59:59+08:00",
            ["cv-a-v1"],
        ),
        (
            "CV-A",
            "2026-10-01T00:00:00+08:00",
            ["cv-a-v2"],
        ),
        (
            "CV-A",
            "2026-10-19T07:10:00+08:00",
            ["cv-a-v2"],
        ),
        (
            "CV-B",
            "2026-10-19T07:20:00+08:00",
            ["cv-b-v1"],
        ),
    ],
)

def test_selects_applicable_manuals(model, at, expected_ids):
    chunks = load_manuals(MANUAL_DIR)

    assert len(chunks) == 3

    selected = select_applicable(
        chunks,
        model=model,
        at=datetime.fromisoformat(at),
    )

    assert [chunk.doc_id for chunk in selected] == expected_ids

def test_distinguishes_missing_model_from_no_matching_manual():
    chunks = load_manuals(MANUAL_DIR)
    at = datetime.fromisoformat("2026-10-19T07:10:00+08:00")

    assert select_applicable(chunks, model="PK-A", at=at) == []

    with pytest.raises(ValueError, match="设备缺少型号"):
        select_applicable(chunks, model=None, at=at)

def test_rejects_time_without_timezone():
    chunks = load_manuals(MANUAL_DIR)

    with pytest.raises(ValueError, match="时间必须包含时区"):
        select_applicable(
            chunks,
            model="CV-A",
            at=datetime(2026, 10, 19, 7, 10),
        )


def test_rejects_invalid_validity_interval(tmp_path):
    original = (MANUAL_DIR / "cv-a-v2.md").read_text(
        encoding="utf-8"
    )

    invalid = original.replace(
        "失效时间：无",
        "失效时间：2026-09-01T00:00:00+08:00",
    )

    path = tmp_path / "invalid.md"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(ValueError, match="失效时间必须晚于生效时间"):
        load_manual(path)