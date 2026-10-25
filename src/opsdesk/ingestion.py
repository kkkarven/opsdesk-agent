"""原文归档、逐条校验、数据库导入与异常记录导出。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from opsdesk.config import Settings
from opsdesk.schemas import DeviceInput, EventInput, utc_text
from opsdesk.storage import Repository


def canonical_json(value) -> str:#规范json格式
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )

def archive_bytes(path:Path,payload:bytes):#归档原文,保护原始数据
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                f"archive contents differ: {path.name}"
            )
        return
    with path.open("xb") as stream:
        stream.write(payload)

def export_issues(#导出坏记录
    repository:Repository,
    batch_id:str,
):
    directory = (
        repository.settings.data_dir/"quarantine"
    )
    directory.mkdir(parents =True,exist_ok=True)

    with repository.connection() as db:#查一个批次
        rows = db.execute(
            """
            SELECT *
            FROM data_issues
            WHERE batch_id = ?
            ORDER BY created_at,issue_id 
            """,
            (batch_id,),
        ).fetchall()

    for row in rows:
        item = dict(row)
        item["raw_record"] = json.loads(#整理原始记录
            item.pop("raw_json")
        )

        destination = (
            directory / f"{row['issue_id']}.json"
        )
        payload = json.dumps(
            item,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        if (
            destination.exists()
            and destination.read_bytes() == payload
        ):
            continue

        temproary = directory / (#保护文件读取
            f".{row["issue_id"]}.{uuid4().hex}.tmp"
        )
        temproary.write_bytes(payload)
        temproary.replace(destination)

def import_dataset(
        directory: Path,
        settings:Settings,
) -> dict:
    repository = Repository(settings)
    repository.initialize()

    files = {
        name:(directory / name).read_bytes()
        for name in ("devices.json","events.json")
    }

    fingerprint = hashlib.sha256()
    
    for name,payload in files.items():#哈希码生成
        fingerprint.update(name.encode())
        fingerprint.update(
            len(payload).to_bytes(8,"big")
        )
        fingerprint.update(payload)

    batch_id = fingerprint.hexdigest()

    archive = (
        settings.data_dir / "imports" / batch_id
    )
    archive.mkdir(parents = True,exist_ok =True)

    for name,payload in files.items():
        archive_bytes(archive / name , payload)

    def reject_constant(value):#检查NaN&ininity
        raise ValueError(
            f"non-JSON numeric constant: {value}"
        )

    records = {
        name:json.loads(
            payload.decode("utf-8-sig"),
            parse_constant= reject_constant,
        )
    for name,payload in files.items()
    }

    if any(
        not isinstance(value,list)
        for value in records.values()
    ):
        raise ValueError(
            "devices.json and events.json "
            "must both contain a JSON array"
        )

    created_at = utc_text(datetime.now(UTC))

    result = {
        "batch_id": batch_id,
        "accepted_devices": 0,
        "accepted_events": 0,
        "skipped_duplicates": 0,
        "rejected_records": 0,
        "already_imported": False,
    }

    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")

        existing = db.execute(#检查batch相同，查老数据
            """
            SELECT result_json
            FROM imports
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()

        if existing is not None:#老批次，调取已有数据
            result = {
                **json.loads(existing["result_json"]),
                "already_imported":True,
            }
        else:#新批次，创建批次
            db.execute(
                "INSERT INTO imports VALUES (?, ?, ?)",
                (batch_id, created_at, "{}"),
            )

            blocked_devices = set()

            for filename,model,table,key,counter in (
            (
                "devices.json",
                DeviceInput,
                "devices",
                "device_id",
                "accepted_devices",
            ),
            (
                "events.json",
                EventInput,
                "events",
                "event_id",
                "accepted_events",
            ),#设备+事件
            ):
                for index,raw in enumerate(#编号
                    records[filename]
                ):
                    try:
                        item = model.model_validate(raw)#根据类检查数据合法

                        if table == "events":#只让事件经历下面的动作
                            if(
                                item.device_id
                                in blocked_devices#验证是否是blocked设备
                            ):
                                raise ValueError(
                                    "device metadata conflicts "
                                    "in this import"
                                )
                            known_device = db.execute(#找设备
                                """
                                SELECT 1
                                FROM devices
                                WHERE device_id = ?
                                """,
                                (item.device_id,),
                            ).fetchone()

                            if known_device is None:
                                raise ValueError(
                                    "event references "
                                    "an unknown device"
                                )

                        old = db.execute(#查重编号
                            f"""
                            SELECT raw_json
                            FROM {table}
                            WHERE {key} = ?
                            """,
                            (getattr(item, key),),
                        ).fetchone()

                        if old is not None:
                            if old["raw_json"] == canonical_json(raw):
                                result["skipped_duplicates"] += 1
                                continue

                            raise ValueError(
                                    f"conflicting existing {key}; "
                                     "original retained"         
                            )

                        if table == "devices":#插入设备
                            db.execute(
                                """
                                INSERT INTO devices
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    item.device_id,
                                    item.display_name,
                                    item.model,
                                    item.site,
                                    canonical_json(raw),
                                    batch_id,
                                    index,
                                ),
                            )

                        else:#table == "events"#插入事件
                            db.execute(
                                """
                                INSERT INTO events
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    item.event_id,
                                    item.device_id,
                                    utc_text(item.occurred_at),
                                    item.code,
                                    item.level,
                                    item.message,
                                    canonical_json(raw),
                                    batch_id,
                                    index,
                                )
                            )
                        result[counter] +=1
                    except (ValidationError,ValueError) as error:
                        device_id = (
                            raw.get("device_id")
                            if isinstance(raw,dict)
                            else None
                        )
                        device_id = (
                            device_id
                            if isinstance(device_id, str)
                            else None
                        )
                        if table == "devices" and device_id is not None:
                            blocked_devices.add(device_id)#设备不合法，将整个设备拉黑

                        reason = (#错误原因
                            json.dumps(
                                error.errors(
                                    include_url=False,
                                    include_input=False,
                                ),
                                ensure_ascii=False,
                                default=str,
                            )
                            if isinstance(error,ValidationError)
                            else str(error)
                        )

                        db.execute(
                            """
                            INSERT INTO data_issues
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                uuid4().hex,
                                device_id,
                                table,
                                reason,
                                canonical_json(raw),
                                batch_id,
                                index,
                                created_at,
                            ),
                        )

                        result["rejected_records"] +=1
            db.execute(
                """
                UPDATE imports
                SET result_json = ?
                WHERE batch_id = ?
                """,
                (canonical_json(result),batch_id),
            )
    export_issues(repository,batch_id)
    return result
