import json

from opsdesk.schemas import EventQuery
from opsdesk.storage import Repository

class DeviceNotFound(Exception):
    pass


def source_url(batch_id:str,filename:str)->str:
    return f"/api/sources/{batch_id}/{filename}"

def query_events(
    repository:Repository,
    device_id:str,
    query:EventQuery,
)->dict:
    found = repository.query(device_id,query)

    if found is None:
        raise DeviceNotFound(device_id)

    level_counts = {}
    event_count = 0

    for row in found["counts"]:
        level =  row["level"]
        count =  row["count"]

        level_counts[level] = count
        event_count +=count

    warning_count = level_counts.get("warning",0)
    latest_row = found["latest"]

    if latest_row is None:
        latest = None
    else:
        latest = dict(latest_row)

    events = []

    for row in found["events"]:
        event = dict(row)

        raw_json = event.pop("raw_json")
        event["raw_record"] = json.loads(raw_json)

        event["source_url"] = source_url(
            event["batch_id"],
            "events.json",
        )

        events.append(event)

    related_count = found["related_rejected_count"]
    unattributed_count  =found["unattributed_rejected_count"]

    potentially_incomplete = (
        related_count > 0 or unattributed_count > 0
    )

    if potentially_incomplete:
        quality_note = (
            "存在相关或设备不明的坏记录；未计入正常统计。"
            "这些记录可能不属于当前时间范围，结果可能不完整。"
        )
    else:
        quality_note =None

    known_levels = {"info","warning","error","critical"}
    unrecognized_levels = []

    for level in level_counts:
        if level not in known_levels:
            unrecognized_levels.append(level)

    unrecognized_levels.sort()

    if event_count == 0:
        status = "no_matching_events"
    else:
        status = "ok"

    return{
        "status":status,
        "device":dict(found["device"]),
        "query":query.model_dump(mode = "json"),
        "summary":{
            "event_count": event_count,
            "warning_count": warning_count,
            "level_counts": level_counts,
            "latest": latest,
        },
        "events": events,
        "returned_count": len(events),
        "truncated": event_count > len(events),
        "data_quality": {
            "potentially_incomplete": potentially_incomplete,
            "related_rejected_count": related_count,
            "unattributed_rejected_count": unattributed_count,
            "unrecognized_levels": unrecognized_levels,
            "note": quality_note,
        },
    }

def list_issues(
    repository:Repository,
    device_id:str|None,
    limit:int,
)->dict:
    found = repository.list_issues(device_id,limit)

    items = []

    for row in found["rows"]:
        issue = dict(row)

        raw_json = issue.pop("raw_json")
        issue["raw_record"] = json.loads(raw_json)

        if issue["record_kind"] == "events":
            filename = "events.json"
        else:
            filename = "devices.json"

        issue["source_url"] = source_url(
            issue["batch_id"],
            filename,
        )

        items.append(issue)

    return{
        "total":found["total"],
        "items": items,
        "truncated":found["total"]>len(items),
    }