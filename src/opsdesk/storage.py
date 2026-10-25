"""数据库连接、表结构和初始化。"""

import sqlite3
from contextlib import contextmanager

from opsdesk.config import Settings
from opsdesk.schemas import EventQuery,utc_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS imports(
    batch_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS devices(
    device_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    model TEXT,
    site TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES imports(batch_id),
    source_index INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL REFERENCES devices(device_id),
    occurred_at TEXT NOT NULL,
    code TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES imports(batch_id),
    source_index INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS events_device_time
ON events(device_id, occurred_at, event_id);

CREATE TABLE IF NOT EXISTS data_issues (
    issue_id TEXT PRIMARY KEY,
    device_id TEXT,
    record_kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES imports(batch_id),
    source_index INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS issues_device
ON data_issues(device_id, created_at);
"""

class Repository:
    def __init__(self,settings:Settings):
        self.settings = settings

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(
            self.settings.db_path,
            timeout= 5
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON") #检查外键连接

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.settings.data_dir.mkdir(
            parents = True,
            exist_ok = True,
        )

        with self.connection() as db:
            version = db.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if version not in (0,1):
                raise RuntimeError(
                    f"unsupported database schema version: {version}"
                )
            db.executescript(SCHEMA)
            db.execute("PRAGMA user_version = 1")

    def list_devices(self):
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT device_id, display_name, model, site
                FROM devices
                ORDER BY device_id
                """
            ).fetchall()
        return rows

    def query(self,device_id:str,query:EventQuery):
        with self.connection() as db:
            db.execute("BEGIN")

            device = db.execute(
                """
                SELECT device_id, display_name, model, site
                FROM devices
                WHERE device_id = ?
                """,
                (device_id,),
            ).fetchone()

            if device is None:
                return None

            where = (
                "device_id = ? "
                "AND occurred_at >= ? "
                "AND occurred_at <= ?"
            )

            parameters = [
                device_id,
                utc_text(query.start),
                utc_text(query.end),
            ]

            if query.level is not None:#判断是否有level筛选
                where += "AND level = ?"
                parameters.append(query.level)

            counts = db.execute(
                f"""
                SELECT level, COUNT(*) AS count
                FROM events
                WHERE {where}
                GROUP BY level
                """,
                parameters,
            ).fetchall()

            latest = db.execute(
                f"""
                SELECT event_id, occurred_at, level
                FROM events
                WHERE {where}
                ORDER BY occurred_at DESC, event_id DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()

            events = db.execute(
                f"""
                SELECT *
                FROM events
                WHERE {where}
                ORDER BY occurred_at ASC, event_id ASC
                LIMIT ?
                """,
                parameters + [query.limit],
            ).fetchall()

            related_rejected_count = db.execute(
                """
                SELECT COUNT(*)
                FROM data_issues
                WHERE device_id = ?
                """,
                (device_id,),
            ).fetchone()[0]

            unattributed_rejected_count = db.execute(
                """
                SELECT COUNT(*)
                FROM data_issues
                WHERE device_id IS NULL
                """
            ).fetchone()[0]

            return{
                "device": device,
                "counts": counts,
                "latest": latest,
                "events": events,
                "related_rejected_count": related_rejected_count,
                "unattributed_rejected_count": unattributed_rejected_count,

            }

    def list_issues(self,device_id:str|None,limit:int):
        with self.connection() as db:
            db.execute("BEGIN")

            if device_id is None:
                where = ""
                parameters = []
            else:
                where = (
                    " WHERE device_id = ? "
                    "OR device_id IS NULL"

                )
                parameters = [device_id]

            total = db.execute(
                f"SELECT COUNT(*) FROM data_issues{where}",
                parameters,
            ).fetchone()[0]

            rows = db.execute(
                f"""
                SELECT *
                FROM data_issues
                {where}
                ORDER BY created_at ASC, issue_id ASC
                LIMIT ?
                """,
                parameters + [limit],
            ).fetchall()

            return{
                "total":total,
                "rows":rows,
            }

    def has_import(self,batch_id:str)-> bool:
        with self.connection() as db:
            row = db.execute(
                """
                SELECT 1
                FROM imports
                WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()

        return row is not None

    def ping(self):
        with self.connection() as db:
            db.execute(
                "SELECT COUNT(*) FROM devices"
            ).fetchone()