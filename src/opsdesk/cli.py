"""通过命令初始化数据库或导入数据。"""

import argparse
import json
import sqlite3
from pathlib import Path

from opsdesk.config import Settings
from opsdesk.ingestion import import_dataset
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

    arguments = parser.parse_args()#保存用户输入值

    settings = (#选定环境路径
        Settings(arguments.data_dir.expanduser().resolve())
        if arguments.data_dir
        else Settings.from_env()
    )

    try:
        if arguments.command == "init":
            Repository(settings).initialize()

            result  = {
                "database":str(settings.db_path),
                "initialized":True,
            }
        else:#import
            result = import_dataset(
                arguments.directory,
                settings,
            )
    except(
        OSError,
        ValueError,
        sqlite3.Error,
        RuntimeError,
    ) as error:
        parser.exit(
            1,
            f"导入/初始化失败：{error}\n",
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )



if __name__ == "__main__":
    main()