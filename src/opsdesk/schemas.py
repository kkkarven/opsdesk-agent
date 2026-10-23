"""设备、事件与时间格式的输入约束。"""

from datetime import UTC, datetime
from typing import Annotated,TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    Field,
)

Identifier = Annotated[
    str,
    Field(pattern = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
]

def require_iso_datetime(value):#检查时间格式
    if isinstance(value,datetime):
        return value

    if not isinstance(value,str) or "T" not in value:
        raise ValueError(
            "use an ISO datetime with timezone, "
            "e.g. 2026-10-19T07:00:00+08:00"
        )
    return value

Timestamp:TypeAlias = Annotated[
    AwareDatetime,
    BeforeValidator(require_iso_datetime),
]

def utc_text(value:datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")

class DeviceInput(BaseModel):
    device_id : Identifier
    display_name: str = Field(min_length=1,max_length=200)
    model: str | None = None
    site: str =Field(min_length=1,max_length=200)

class EventInput(BaseModel):
    event_id: Identifier
    device_id: Identifier
    occurred_at: Timestamp
    code: str = Field(min_length=1,max_length=100)
    level: str = Field(min_length=1,max_length=40)
    message: str = Field(min_length=1,max_length=10000)