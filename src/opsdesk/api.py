import logging
import re
import sqlite3
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from opsdesk.request_logging import install_request_logging
from opsdesk.config import Settings
from opsdesk.schemas import EventQuery, Identifier
from opsdesk.service import (
    DeviceNotFound,
    list_issues,
    query_events,
)
from opsdesk.storage import Repository

logger = logging.getLogger("uvicorn.error")

def storage_unavailable():
    logger.exception("数据库操作失败")

    raise HTTPException(
        status_code=503,
        detail={
            "code": "STORAGE_UNAVAILABLE",
            "message": "数据库暂不可用，请查看服务日志。",
        }
    )


def create_app(settings:Settings|None = None) ->FastAPI:
    if settings is None:
        settings = Settings.from_env()

    repository = Repository(settings)
    repository.initialize()

    app = FastAPI(
        title="OpsDesk 设备事件数据服务",
        version="0.1.0",
        description=(
            "支持设备事件查询、统计和原始来源追溯。"
            "示例设备与事件为模拟数据，尚未接入 Agent。"
        ),
    )
    install_request_logging(app)

    @app.get("/health")#访问检查
    def health():
        try:
            repository.ping()
            return{"status":"ok"}
        except sqlite3.Error:
            storage_unavailable()

    @app.get("/api/devices")#设备
    def devices():
        try:
            rows = repository.list_devices()
            items = []

            for row in rows:
                items.append(dict(row))

            return{"items":items}
        except sqlite3.Error:
            storage_unavailable()

    @app.get("/api/devices/{device_id}/events")
    def events(
        device_id: Identifier,
        query:Annotated[EventQuery,Query()],
    ):
        try:
            return query_events(
                repository,
                device_id,
                query,
            )
        except DeviceNotFound:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DEVICE_NOT_FOUND",
                    "message": f"设备 {device_id} 不存在。",
                },
            ) from None
        except sqlite3.Error:
            storage_unavailable()

    @app.get("/api/data-issues")
    def data_issues(
        device_id: Identifier |None = None,
        limit: Annotated[int,Query(ge=1,le=1000)] = 200,
    ):
        try:
            return list_issues(
                repository,
                device_id,
                limit,
            )
        except sqlite3.Error:
            storage_unavailable()

    @app.get("/api/sources/{batch_id}/{filename}")
    def sources(
        batch_id: str,
        filename: Literal["devices.json", "events.json"],
    ):
        if re.fullmatch(r"[a-f0-9]{64}",batch_id) is None:
            raise HTTPException(
                status_code=404,
                detail={"code":"SOURCE_NOT_FOUND"},
            )

        try:
            exists = repository.has_import(batch_id)
        except sqlite3.Error:
            storage_unavailable()

        if not exists:
            raise HTTPException(
                status_code=404,
                detail={"code":"SOURCE_NOT_FOUND"},
            )

        path = (
            settings.data_dir/"imports"/batch_id/filename
        )

        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail={"code":"SOURCE_FILE_MISSING"},
            )

        return FileResponse(
            path,
            media_type="application/json",
        )
    return app