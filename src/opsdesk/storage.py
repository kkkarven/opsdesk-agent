"""数据库连接、表结构和初始化。"""

import sqlite3
from contextlib import contextmanager

from opsdesk.config import Settings

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