import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class Settings:
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            Path(os.environ.get("OPSDESK_DATA_DIR","data"))
            .expanduser()
            .resolve()
        )
    @property
    def db_path(self) -> Path:
        return self.data_dir/"opsdesk.sqlite3"