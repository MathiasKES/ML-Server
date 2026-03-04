from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    host: str
    username: str | None = None
    password: str | None = None
    token: str | None = None
    timeout: float = 10.0
    verify_ssl: bool = True
    offline_mode: str = "queue"
    spool_dir: Path | None = None
