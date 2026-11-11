"""Build the two read-only tools for one OpsDesk investigation."""

import json
from datetime import datetime
from functools import cache
from typing import Literal, Self

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opsdesk.config import Settings
from opsdesk.embeddings import OllamaEmbedder
from opsdesk.retrieval import VectorRetriever
from opsdesk.schemas import EventQuery, Identifier
from opsdesk.service import query_events as read_events
from opsdesk.storage import Repository


class InvestigationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    device_id: Identifier
    query: EventQuery
    event_id: Identifier | None = None
    question: str = Field(min_length=1, max_length=800)
    max_model_calls: int = Field(default=5, ge=1, le=20)


class InvestigationRequest(InvestigationInput):#检查输入
    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)

    task: Literal["events","knowledge","investigate"]

    @model_validator(mode="after")
    def check_target(self) -> Self:
        if self.task in {"knowledge", "investigate"} and self.event_id is None:
            raise ValueError("查手册或综合调查需要指定 --event-id")
        return self

class NoArguments(BaseModel):#检查query_events
    model_config = ConfigDict(extra="forbid")

class KnowledgeArguments(BaseModel):#检查search_knowledge
    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)

    topic: str = Field(min_length=1,max_length=200,description="需要查找的手册主题")


def tool_result(status:str,sources:dict,**details)->tuple[str,dict]:
    payload = {"status":status,"sources":sources,**details}
    return json.dumps(payload,ensure_ascii=False),payload


def build_tools(settings:Settings,task:InvestigationRequest) -> list:
    repository = Repository(settings)
    @cache
    def read_window() -> dict:
        if not settings.db_path.is_file():
            raise FileNotFoundError("SQLite 数据库不存在，请先运行 opsdesk import")
        lookup_query = task.query.model_copy(update={"limit":1000})
        return read_events(repository=repository,device_id=task.device_id,query=lookup_query)

    def find_target(report:dict) -> dict | None:
        return next(
        (
            event 
            for event in report["events"]
            if event["event_id"] == task.event_id
        ),
            None,
        )
        

    @tool(args_schema = NoArguments,response_format = "content_and_artifact")
    def query_events() -> tuple[str,dict]:
        """查询当前设备和时间窗口内的事件、统计和数据质量。"""
        report = read_window()
        records = report["events"][:8]
        target = find_target(report)
        if target is not None and target not in records:
            records.append(target)

        sources = {
            "window:summary":{
                "kind":"summary",
                "device":report["device"],
                "query":report["query"],
                "summary":report["summary"],
                "data_quality":report["data_quality"],
                "visible_event_count":len(records),
                "detail_is_partial":report["summary"]["event_count"] > len(records),
            }
        }
        for event in records:
            sources[f"event:{event['event_id']}"] = {
                "kind":"event",
                "event_id":event["event_id"],
                "device_id":event["device_id"],
                "occurred_at":event["occurred_at"],
                "code":event["code"],
                "level":event["level"],
                "message":event["message"][:300],
                "message_truncated":len(event["message"]) > 300,
                "source_url":event["source_url"],
                "source_index":event["source_index"],
            }

        return tool_result(report["status"],sources)

    @tool(args_schema = KnowledgeArguments,response_format="content_and_artifact")
    def search_knowledge(topic: str) -> tuple[str,dict]:
        """按当前目标事件的型号和发生时间检索手册。topic 填告警代码、现象或要查的主题。"""
        report = read_window()
        target = find_target(report)
        if target is None:
            return tool_result(
             "anchor_not_found",
             {},
             note = "当前查询未定位到目标事件。请核对事件编号、级别和时间窗口，必要时缩小窗口。", 
            )

        embedder = OllamaEmbedder(timeout=45.0)
        snapshot = settings.data_dir/"knowledge"/"retrieval-v1.npz"
        if not snapshot.is_file():
            raise FileNotFoundError("向量索引不存在，请先运行 opsdesk knowledge-build")
        retriever = VectorRetriever.load(snapshot,expected_model=embedder.model)

        hits = retriever.search(
            question=f"事件码：{target['code']}\n调查问题：{task.question}\n检索主题：{topic}",
            device_model=report["device"]["model"],
            at = datetime.fromisoformat(target["occurred_at"]),
            embedder=embedder,
            top_k= 2,
        )
        if not hits:
            return tool_result(
                "no_applicable_manuals",
                {},
                note="当前资料库没有覆盖该设备型号和事件时间的手册。",
            )
    
        sources = {}    
        for hit in hits:
            chunk = hit.chunk
            sources[f"manual:{chunk.doc_id}:{chunk.line_start}"] = {
                "kind":"manual",
                "chunk_id":chunk.chunk_id,
                "title":chunk.title,
                "model":chunk.model,
                "version":chunk.version,
                "valid_from":chunk.valid_from.isoformat(),
                "valid_to":chunk.valid_to.isoformat() if chunk.valid_to is not None else None,
                "source_path":chunk.source_path,
                "line_start":chunk.line_start,
                "line_end":chunk.line_end,
                "text":chunk.text[:1200],
                "text_truncated":len(chunk.text) > 1200,
                "score":round(hit.score,4),   
            }
        return tool_result("ok",sources)

    return [query_events,search_knowledge]