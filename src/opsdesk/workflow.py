"""Run a resumable OpsDesk investigation."""

import argparse
import json
import sqlite3
from typing import TypedDict
from uuid import uuid4
from contextlib import contextmanager
from copy import deepcopy

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from opsdesk import agent
from opsdesk.agent_tools import InvestigationRequest,InvestigationInput
from opsdesk.config import Settings
from opsdesk.schemas import EventQuery

class InvestigationState(TypedDict):
    version: int
    original_request: dict
    current_request: dict
    plan: dict | None
    bundle: dict
    attempts: list[dict]
    status: str
    reason: str | None
    waiting_field: str | None
    supplements: int
    corrections: int


def initial_state(inputs: InvestigationInput) -> dict:
    request = inputs.model_dump(mode="json")
    return {
        "version": 2,
        "original_request": request,
        "current_request": deepcopy(request),
        "plan": None,
        "bundle": {},
        "attempts": [],
        "status": "running",
        "reason": None,
        "waiting_field": None,
        "supplements": 0,
        "corrections": 0,
    }


def attempt_result(result):
    return {
        "status": result["status"],
        "reason": result.get("reason"),
        "answer": deepcopy(result.get("answer")),
        "retrieval_request": deepcopy(result.get("retrieval_request")),
        "error": result.get("error"),
    }


def build_workflow(settings: Settings, checkpointer):
    def route(state):
        bundle = deepcopy(state["bundle"])
        trace = bundle.setdefault("model_trace", [])
        request = state["current_request"]
        try:
            plan = agent.route_task(
                request["question"], trace, request["max_model_calls"]
            )
        except agent.RunFailure as error:
            bundle["error"] = str(error)
            return {"bundle": bundle, "status": "failed", "reason": error.status}
        return {"bundle": bundle, "plan": plan.model_dump(mode="json")}

    def waiting(state, reason, field):
        if state["corrections"] >= 1:
            return {
                "status": "insufficient_evidence",
                "reason": reason,
                "waiting_field": None,
            }
        return {
            "status": "waiting_for_input",
            "reason": reason,
            "waiting_field": field,
        }

    def check_input(state):
        task = state["plan"]["task"]
        if task == "needs_input":
            return waiting(state, "unclear_or_unsupported_goal", "question")
        if task != "events" and not state["current_request"]["event_id"]:
            return waiting(state, "missing_event_id", "event_id")
        return {"status": "running", "reason": None, "waiting_field": None}

    def request_for(state):
        return InvestigationRequest(
            **state["current_request"], task=state["plan"]["task"]
        )

    def collect(state):
        result = agent.collect_evidence(settings, request_for(state), state["bundle"])
        update = {
            "bundle": result,
            "attempts": [*state["attempts"], {
                "phase": "collect",
                "event_id": state["current_request"]["event_id"],
                "result": attempt_result(result),
            }],
        }
        if result.get("reason") == "anchor_not_found":
            update.update(waiting(state, "anchor_not_found", "event_id"))
        elif result["status"] != "collected":
            update.update(terminal(result))
        return update

    def answer(state):
        result = agent.draft_answer(
            settings,
            request_for(state),
            state["bundle"],
            structured_retrieval=True,
            retrieval_allowed=(
                state["plan"]["task"] != "events" and state["supplements"] == 0
            ),
        )
        return {
            "bundle": result,
            "attempts": [*state["attempts"], {
                "phase": "answer",
                "event_id": state["current_request"]["event_id"],
                "result": attempt_result(result),
            }],
        }

    def decide(state):
        result = state["bundle"]
        if (
            result["status"] == "insufficient_evidence"
            and result.get("retrieval_request")
            and state["supplements"] == 0
            and state["plan"]["task"] != "events"
        ):
            remaining = (
                state["current_request"]["max_model_calls"]
                - len(result["model_trace"])
            )
            if remaining < 3:
                return {"status": "insufficient_evidence", "reason": "budget_exceeded"}
            return {"status": "supplementing", "reason": None}
        return terminal(result)

    def supplement(state):
        before = state["bundle"]
        query = before["retrieval_request"]["query"]
        result = agent.supplement_evidence(settings, request_for(state), before, query)
        update = {
            "bundle": result,
            "supplements": state["supplements"] + 1,
            "attempts": [*state["attempts"], {
                "phase": "supplement",
                "query": query,
                "event_id": state["current_request"]["event_id"],
                "result": attempt_result(result),
            }],
            "status": "running",
        }
        if result["status"] != "collected":
            update.update(terminal(result))
        return update

    def wait_for_input(state):
        value = interrupt({
            "field": state["waiting_field"],
            "reason": state["reason"],
            "device_id": state["current_request"]["device_id"],
            "previous_value": state["current_request"].get(state["waiting_field"]),
        })
        updated = validate_resume(state, value)
        bundle = {
            key: deepcopy(state["bundle"].get(key, default))
            for key, default in (
                ("model_trace", []),
                ("tool_trace", []),
                ("elapsed_ms", 0),
                ("validation_retries", 0),
            )
        }
        plan = None if state["waiting_field"] == "question" else state["plan"]
        return {
            "current_request": updated,
            "plan": plan,
            "bundle": bundle,
            "corrections": state["corrections"] + 1,
            "waiting_field": None,
            "status": "running",
            "reason": None,
        }

    graph = StateGraph(InvestigationState)
    for name, function in (
        ("route", route),
        ("check_input", check_input),
        ("collect", collect),
        ("answer", answer),
        ("decide", decide),
        ("supplement", supplement),
        ("wait_for_input", wait_for_input),
    ):
        graph.add_node(name, function)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: "end" if state["status"] == "failed" else "next",
        {"end": END, "next": "check_input"},
    )
    for name, target in (("check_input", "collect"), ("collect", "answer")):
        graph.add_conditional_edges(
            name,
            route_status,
            {"wait": "wait_for_input", "next": target, "end": END},
        )
    graph.add_edge("answer", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: "retry" if state["status"] == "supplementing" else "end",
        {"retry": "supplement", "end": END},
    )
    graph.add_conditional_edges(
        "supplement",
        route_status,
        {"next": "answer", "wait": "wait_for_input", "end": END},
    )
    graph.add_conditional_edges(
        "wait_for_input",
        lambda state: "route" if state["plan"] is None else "check_input",
        {"route": "route", "check_input": "check_input"},
    )
    return graph.compile(checkpointer=checkpointer)


def terminal(result):
    status = result["status"]
    if status == "answered":
        return {"status": "completed", "reason": None}
    if status == "insufficient_evidence":
        return {"status": status, "reason": result.get("reason") or "missing_information"}
    return {"status": "failed", "reason": result.get("reason") or status}


def route_status(state):
    if state["status"] == "waiting_for_input":
        return "wait"
    return "next" if state["status"] == "running" else "end"

def validate_resume(state, values):
    field = state["waiting_field"]
    if not isinstance(values, dict) or set(values) != {field}:
        raise ValueError(f"当前只接受补充 {field}")

    updated = InvestigationInput.model_validate({
        **state["current_request"],
        **values,
    })
    if getattr(updated, field) == state["current_request"].get(field):
        raise ValueError("补充值必须与原值不同")
    if field == "event_id" and updated.event_id is None:
        raise ValueError("事件编号不能为空")
    return updated.model_dump(mode="json")


@contextmanager
def open_workflow(settings):
    import fcntl

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with (settings.data_dir / "workflow-v2.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = settings.data_dir / "workflow-v2.sqlite3"
        with SqliteSaver.from_conn_string(str(path)) as saver:
            yield build_workflow(settings, saver)


def snapshot_result(graph, task_id):
    config = {"configurable": {"thread_id": task_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise ValueError(f"任务不存在：{task_id}")
    return {
        "task_id": task_id,
        **snapshot.values,
        "interrupts": [
            item.value for task in snapshot.tasks for item in task.interrupts
        ],
    }


def start_task(settings, inputs):
    task_id = str(uuid4())
    config = {"configurable": {"thread_id": task_id}}
    with open_workflow(settings) as graph:
        graph.invoke(initial_state(inputs), config=config, durability="sync")
        return snapshot_result(graph, task_id)


def show_task(settings, task_id):
    with task_graph(settings, task_id) as graph:
        return snapshot_result(graph, task_id)


def resume_task(settings, task_id, values):
    with task_graph(settings, task_id) as graph:
        state = snapshot_result(graph, task_id)
        if state["status"] != "waiting_for_input" or not state["interrupts"]:
            raise ValueError("当前任务没有等待补充的步骤")

        if state.get("version") == 2:
            validate_resume(state, values)
            resumed = values
        else:
            if set(values) != {"event_id"}:
                raise ValueError("旧任务只接受 event_id")
            request = InvestigationRequest.model_validate({
                **state["current_request"],
                **values,
            })
            if request.event_id == state["current_request"]["event_id"]:
                raise ValueError("请提供更正后的事件编号")
            resumed = request.event_id

        config = {"configurable": {"thread_id": task_id}}
        graph.invoke(Command(resume=resumed), config=config, durability="sync")
        return snapshot_result(graph, task_id)


@contextmanager
def task_graph(settings, task_id):
    with open_workflow(settings) as graph:
        yield graph


def main() -> int:
    parser = argparse.ArgumentParser(description="运行或修复OpsDesk调查")
    commands = parser.add_subparsers(dest="command",required=True)

    start = commands.add_parser("start")
    start.add_argument("device_id")
    start.add_argument("--event-id")
    start.add_argument("--level")
    start.add_argument("--max-model-calls", type=int, default=16, choices=range(1, 21))
    start.add_argument("--start", required=True)
    start.add_argument("--end", required=True)
    start.add_argument("--question", required=True)

    show = commands.add_parser("show")
    show.add_argument("task_id")

    resume = commands.add_parser("resume")
    resume.add_argument("task_id")
    fields = resume.add_mutually_exclusive_group(required=True)
    fields.add_argument("--event-id")
    fields.add_argument("--question")
    args = parser.parse_args()

    settings = Settings.from_env()
    try:
        if args.command == "start":
            inputs = InvestigationInput(
                device_id=args.device_id,
                event_id=args.event_id,
                question=args.question,
                query=EventQuery(start=args.start, end=args.end, level=args.level),
                max_model_calls=args.max_model_calls,
            )
            result = start_task(settings, inputs)
        elif args.command == "show":
            result = show_task(settings, args.task_id)
        else:
            values = (
                {"event_id": args.event_id}
                if args.event_id is not None
                else {"question": args.question}
            )
            result = resume_task(settings, args.task_id, values)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())