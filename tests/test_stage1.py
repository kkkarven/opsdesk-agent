import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opsdesk.api import create_app
from opsdesk.config import Settings
from opsdesk.ingestion import import_dataset
from opsdesk.storage import Repository

EXAMPLES = Path(__file__).resolve().parents[1]/"examples"
NORMAL = EXAMPLES/"normal"
ENDPOINT = "/api/devices/CV-07/events"

WINDOW = {
    "start": "2026-10-19T07:00:00+08:00",
    "end": "2026-10-19T08:00:00+08:00",
}

def prepare(tmp_path, scenario="normal"):#生成配置
    settings = Settings(tmp_path / "data")
    import_dataset(EXAMPLES / scenario, settings)
    return settings

def write_input(tmp_path, events, devices=None):#生成输入
    directory = tmp_path / "input"
    directory.mkdir(exist_ok=True)

    if devices is None:
        (directory / "devices.json").write_bytes(
            (NORMAL / "devices.json").read_bytes()
        )
    else:
        (directory / "devices.json").write_text(
            json.dumps(devices, ensure_ascii=False),
            encoding="utf-8",
        )

    (directory / "events.json").write_text(
        json.dumps(events, ensure_ascii=False),
        encoding="utf-8",
    )

    return directory

def test_query_and_source(tmp_path):
    settings = prepare(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get(ENDPOINT, params=WINDOW)

        assert response.status_code == 200
        assert response.headers["x-request-id"]#检查头，报错编号

        result = response.json()
        summary = result["summary"]

        assert summary["event_count"] == 3#检查数据
        assert summary["warning_count"] == 2
        assert summary["latest"]["event_id"] == "EV-006"
        assert "code" not in summary["latest"]

        assert [e["event_id"] for e in result["events"]] == [
            "EV-002", "EV-004", "EV-006"
        ]

        first = result["events"][0]
        assert first["source_index"] == 1
        assert (
            first["raw_record"]["occurred_at"]
            == "2026-10-19T07:10:00+08:00"
        )

        source = client.get(first["source_url"])

        assert source.status_code == 200
        assert source.content == (NORMAL / "events.json").read_bytes()
        assert (
            source.headers["x-request-id"]
            != response.headers["x-request-id"]
        )

def test_windows_and_level(tmp_path):
    settings = prepare(tmp_path)

    cases = [#检查时间窗，事件等级
        (
            {
                "start": "2026-10-19T07:10:00+08:00",
                "end": "2026-10-19T07:35:00+08:00",
            },
            ["EV-002", "EV-004"],
        ),
        (
            {
                "start": "2026-10-18T23:00:00Z",
                "end": "2026-10-19T00:00:00Z",
            },
            ["EV-002", "EV-004", "EV-006"],
        ),
        (
            {**WINDOW, "level": "warning"},
            ["EV-002", "EV-004"],
        ),
    ]

    with TestClient(create_app(settings)) as client:
        for parameters, expected_ids in cases:
            response = client.get(ENDPOINT, params=parameters)
            assert response.status_code == 200

            result = response.json()

            assert [e["event_id"] for e in result["events"]] == expected_ids
            assert result["summary"]["event_count"] == len(expected_ids)
            assert (
                result["summary"]["latest"]["event_id"]
                == expected_ids[-1]
            )

def test_limit(tmp_path):#检查limit
    settings = prepare(tmp_path)

    with TestClient(create_app(settings)) as client:
        result = client.get(
            ENDPOINT,
            params={**WINDOW, "limit": 1},
        ).json()

        assert result["returned_count"] == 1
        assert result["truncated"] is True
        assert result["summary"]["event_count"] == 3
        assert result["summary"]["warning_count"] == 2
        assert result["summary"]["latest"]["event_id"] == "EV-006"

def test_query_errors(tmp_path):#query：设备不存在，空结果
    settings = prepare(tmp_path)

    with TestClient(create_app(settings)) as client:
        missing = client.get(
            "/api/devices/UNKNOWN/events",
            params=WINDOW,
        )
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "DEVICE_NOT_FOUND"

        empty = client.get(
            "/api/devices/PK-02/events",
            params={
                "start": WINDOW["start"],
                "end": "2026-10-19T07:30:00+08:00",
            },
        ).json()
        assert empty["status"] == "no_matching_events"
        assert empty["summary"]["latest"] is None

        changes = [
            {"start": "INVALID"},
            {"start": "2026-10-19T07:00:00"},
            {"start": "2026-10-19T09:00:00+08:00"},
            {"limit": 0},
            {"limit": 1001},
            {"unexpected": "field"},
        ]

        for change in changes:
            response = client.get(
                ENDPOINT,
                params={**WINDOW, **change},
            )
            assert response.status_code == 422
            assert response.headers["x-request-id"]

def test_reimport_and_restart(tmp_path):#检查重复导入
    settings = prepare(tmp_path)

    second = import_dataset(NORMAL, settings)
    assert second["already_imported"] is True

    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").status_code == 200
            assert len(client.get("/api/devices").json()["items"]) == 3

            result = client.get(ENDPOINT, params=WINDOW).json()
            assert result["summary"]["event_count"] == 3

def test_bad_timestamp(tmp_path):#检查坏时间数据
    settings = prepare(tmp_path, "invalid_event")

    with TestClient(create_app(settings)) as client:
        result = client.get(ENDPOINT, params=WINDOW).json()

        assert result["summary"]["event_count"] == 2
        assert result["summary"]["warning_count"] == 1
        assert result["data_quality"]["potentially_incomplete"] is True

        issues = client.get("/api/data-issues").json()
        assert issues["total"] == 1

        issue = issues["items"][0]
        assert issue["raw_record"]["occurred_at"] == "INVALID_TIMESTAMP"

        source = client.get(issue["source_url"])
        assert source.content == (
            EXAMPLES / "invalid_event" / "events.json"
        ).read_bytes()

    export = next((settings.data_dir / "quarantine").glob("*.json"))
    assert (
        json.loads(export.read_text(encoding="utf-8"))
        ["raw_record"]["occurred_at"]
        == "INVALID_TIMESTAMP"
    )

    export.unlink()
    result = import_dataset(EXAMPLES / "invalid_event", settings)

    assert result["already_imported"] is True
    assert export.is_file()

def test_event_conflict(tmp_path):#检查事件编号冲突
    settings = prepare(tmp_path)

    events = json.loads(
        (NORMAL / "events.json").read_text(encoding="utf-8")
    )
    original_message = events[1]["message"]
    events[1]["message"] = "changed message"

    result = import_dataset(write_input(tmp_path, events), settings)
    assert result["rejected_records"] == 1

    with TestClient(create_app(settings)) as client:
        found = client.get(ENDPOINT, params=WINDOW).json()
        assert found["events"][0]["message"] == original_message

        issues = client.get("/api/data-issues").json()
        assert issues["items"][0]["raw_record"]["message"] == "changed message"

def test_device_conflict(tmp_path):#检查设备冲突
    settings = prepare(tmp_path)

    devices = json.loads(
        (NORMAL / "devices.json").read_text(encoding="utf-8")
    )
    devices[0]["model"] = "DIFFERENT-MODEL"

    events = json.loads(
        (NORMAL / "events.json").read_text(encoding="utf-8")
    )
    event = events[1]
    event["event_id"] = "NEW-EVENT"

    directory = write_input(tmp_path, [event], devices)
    result = import_dataset(directory, settings)

    assert result["accepted_events"] == 0
    assert result["rejected_records"] == 2

    with TestClient(create_app(settings)) as client:
        issues = client.get("/api/data-issues").json()
        checked = False

        for issue in issues["items"]:
            if issue["record_kind"] == "devices":
                source = client.get(issue["source_url"])
                assert source.content == (
                    directory / "devices.json"
                ).read_bytes()
                checked = True

        assert checked

def test_invalid_document(tmp_path):#检查原文无损坏
    settings = Settings(tmp_path / "data")
    directory = write_input(tmp_path, [])

    for raw in [
        b'[{"event_id": "broken"',
        b'{"not": "an array"}',
        b"[NaN]",
    ]:
        (directory / "events.json").write_bytes(raw)

        with pytest.raises(ValueError):
            import_dataset(directory, settings)

        assert Repository(settings).list_devices() == []

        archived = list(
            (settings.data_dir / "imports").glob("*/events.json")
        )
        assert raw in [path.read_bytes() for path in archived]

def test_record_edges(tmp_path):#检查未知设备
    settings = Settings(tmp_path / "data")

    events = json.loads(
        (NORMAL / "events.json").read_text(encoding="utf-8")
    )
    events[0]["device_id"] = "UNKNOWN"
    events[1]["occurred_at"] = 1792371000
    events[3]["level"] = "vendor_notice"
    events[3]["vendor_payload"] = {"sequence": 123}
    events.append(["not", "an", "event"])

    result = import_dataset(write_input(tmp_path, events), settings)

    assert result["accepted_events"] == 4
    assert result["rejected_records"] == 3

    with TestClient(create_app(settings)) as client:
        found = client.get(ENDPOINT, params=WINDOW).json()

        assert found["summary"]["event_count"] == 2
        assert found["data_quality"]["unattributed_rejected_count"] == 1
        assert found["data_quality"]["unrecognized_levels"] == [
            "vendor_notice"
        ]
        assert (
            found["events"][0]["raw_record"]
            ["vendor_payload"]["sequence"]
            == 123
        )

def test_storage_failure(tmp_path):#检查存储失败
    settings = prepare(tmp_path)

    with TestClient(create_app(settings)) as client:
        with Repository(settings).connection() as db:
            db.execute("DROP TABLE events")

        response = client.get(ENDPOINT, params=WINDOW)

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "STORAGE_UNAVAILABLE"
        assert response.headers["x-request-id"]

        assert client.get("/health").status_code == 200

def test_source_errors(tmp_path):#检查访问限制
    settings = prepare(tmp_path)

    with TestClient(create_app(settings)) as client:
        result = client.get(ENDPOINT, params=WINDOW).json()
        event = result["events"][0]
        batch_id = event["batch_id"]

        assert client.get(
            "/api/sources/not-a-hash/events.json"
        ).status_code == 404

        assert client.get(
            f"/api/sources/{'0' * 64}/events.json"
        ).status_code == 404

        assert client.get(
            f"/api/sources/{batch_id}/opsdesk.sqlite3"
        ).status_code == 422

        archive = (
            settings.data_dir / "imports" / batch_id / "events.json"
        )
        archive.unlink()

        missing = client.get(event["source_url"])
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "SOURCE_FILE_MISSING"