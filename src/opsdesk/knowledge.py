from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from opsdesk.schemas import Timestamp

class KnowledgeChunk(BaseModel):
    doc_id:str = Field(min_length=1,max_length=128,pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",)
    chunk_id:str = Field(min_length=1,max_length=150,pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*:[0-9a-f]{16}:\d{4}$",)
    title:str = Field(min_length=1)
    text:str =Field(min_length=1)

    model:str = Field(min_length=1)
    version:str = Field(min_length=1)
    valid_from:Timestamp
    valid_to:Timestamp|None = None

    source_path:str
    line_start:int = Field(ge=1)
    line_end:int = Field(ge=1)
    content_hash:str

    @model_validator(mode="after")
    def check_validity(self) -> Self:
        if (
            self.valid_to is not None and self.valid_to <= self.valid_from
        ):
            raise ValueError("失效时间必须晚于生效时间")

        if self.line_end < self.line_start:
            raise ValueError("正文结束行不能早于开始行")

        return self

def load_manual(path: Path) -> KnowledgeChunk:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path.name}:第一行必须是一级标题")

    title = lines[0][2:].strip()

    required_fields = {"型号","版本","生效时间","失效时间"}
    metadata:dict[str,str] = {}
    body_start = None

    for index, line in enumerate(lines[1:],start=1):
        stripped = line.strip()

        if not stripped or stripped.startswith("SIMULATED："):
            continue

        key,_,value = stripped.partition("：")
        key =  key.strip()
        value = value.strip()

        if key not in required_fields:
            raise ValueError(
                f"{path.name}:第{index + 1}行不是合法元数据"
            )

        if key in metadata:
            raise ValueError(f"{path.name}:字段重复:{key}")

        if not value:
            raise ValueError(f"{path.name}:字段不能为空:{key}")

        metadata[key] = value

        if required_fields.issubset(metadata):
            body_start = index + 1 
            break

    if body_start is None:
        missing = sorted(required_fields - metadata.keys())
        raise ValueError(f"{path.name}:缺少字段:{missing}")

    body_end = len(lines)

    while body_start < body_end and not lines[body_start].strip():
        body_start += 1
    while body_end > body_start and not lines[body_end - 1].strip():
        body_end -= 1

    if body_start == body_end:
        raise ValueError(f"{path.name}:正文不能为空")

    text = "\n".join(lines[body_start:body_end])

    doc_id = path.stem
    content_hash = sha256(raw.encode("utf-8")).hexdigest()

    valid_to = metadata["失效时间"]
    if valid_to == "无":
        valid_to = None

    return KnowledgeChunk.model_validate(
        {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}:{content_hash[:16]}:0001",
        "title": title,
        "text": text,
        "model": metadata["型号"],
        "version": metadata["版本"],
        "valid_from": metadata["生效时间"],
        "valid_to": valid_to,
        "source_path": path.resolve().as_posix(),
        "line_start": body_start + 1,
        "line_end": body_end,
        "content_hash": content_hash,
        }   
    )

def load_manuals(directory:Path) -> list[KnowledgeChunk]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths = sorted(directory.glob("*.md"))

    if not paths:
        raise ValueError(f"目录中没有 Markdown 手册：{directory}")

    chunks = []

    for path in paths:
        chunk = load_manual(path)
        chunks.append(chunk)

    return chunks

def select_applicable(
        chunks:list[KnowledgeChunk],
        model:str|None,
        at:datetime,
) -> list[KnowledgeChunk]:
    if model is None or not model.strip():
        raise ValueError("设备缺少型号，无法判断手册适用性")

    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("资料匹配时间必须包含时区")

    model = model.strip()
    selected = []

    for chunk in chunks:
        if chunk.model != model:
            continue
        if at < chunk.valid_from:
            continue
        if chunk.valid_to is not None and at >= chunk.valid_to:
            continue

        selected.append(chunk)

    return selected