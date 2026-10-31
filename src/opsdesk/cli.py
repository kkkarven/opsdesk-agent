"""通过命令初始化数据库或导入数据。"""

import argparse
import json
import sqlite3
from pathlib import Path

from opsdesk.config import Settings
from opsdesk.embeddings import OllamaEmbedder
from opsdesk.ingestion import import_dataset
from opsdesk.knowledge import load_manuals
from opsdesk.llm import OllamaChat
from opsdesk.rag import explain_event
from opsdesk.retrieval import VectorRetriever
from opsdesk.schemas import EventQuery
from opsdesk.service import DeviceNotFound
from opsdesk.storage import Repository

def main():
    parser = argparse.ArgumentParser(prog="opsdesk")

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="database and archive directory",
    )

    commands = parser.add_subparsers(
        dest="command",
        required= True
    )

    commands.add_parser(
        "init",
        help="initialize the local database",
    )

    load = commands.add_parser(
        "import",
        help="import devices.json and events.json",
    )

    load.add_argument("directory",type=Path)

    build = commands.add_parser(
        "knowledge-build",
        help="build and save the knowledge snapshot",
    )
    build.add_argument("directory", type=Path)

    explain = commands.add_parser(
        "explain",
        help="explain one event using applicable manuals",
    )
    explain.add_argument("device_id")
    explain.add_argument("--event-id", required=True)
    explain.add_argument("--start", required=True)
    explain.add_argument("--end", required=True)
    explain.add_argument("--question", required=True)

    arguments = parser.parse_args()#保存用户输入值



    settings = (#选定环境路径
        Settings(arguments.data_dir.expanduser().resolve())
        if arguments.data_dir
        else Settings.from_env()
    )

    snapshot_path = (
        settings.data_dir / "knowledge" / "retrieval-v1.npz"
    )

    try:
        if arguments.command == "init":
            Repository(settings).initialize()

            result  = {
                "database":str(settings.db_path),
                "initialized":True,
            }
        elif arguments.command == "import":
            result = import_dataset(
                arguments.directory,
                settings,
            )

        elif arguments.command == "knowledge-build":
            chunks = load_manuals(arguments.directory)
            embedder = OllamaEmbedder()

            retriever = VectorRetriever.from_chunks(
                chunks,
                embedder,
            )
            retriever.save(snapshot_path)

            result = {
                "snapshot": str(snapshot_path),
                "chunk_count": len(chunks),
                "embedding_model": retriever.embedding_model,
                "dimension": retriever.dimension,
            }

        else:
            query = EventQuery(
                start=arguments.start,
                end=arguments.end,
            )

            embedder = OllamaEmbedder()
            retriever = VectorRetriever.load(
                snapshot_path,
                expected_model=embedder.model,
            )

            result = explain_event(
                repository=Repository(settings),
                query=query,
                device_id=arguments.device_id,
                event_id=arguments.event_id,
                question=arguments.question,
                retriever=retriever,
                embedder=embedder,
                llm=OllamaChat(),
            )

    except DeviceNotFound as error:
        parser.exit(1, f"设备不存在：{error}\n")
    except(
        OSError,
        ValueError,
        sqlite3.Error,
        RuntimeError,
    ) as error:
        parser.exit(1, f"操作失败：{error}\n")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )



if __name__ == "__main__":
    main()