from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    host: str
    username: str | None
    password: str | None = field(default=None, repr=False)
    token: str | None = field(default=None, repr=False)

    timeout: float = 10.0
    verify_ssl: bool = True

    offline_mode: str = "queue"  # "queue" | "raise" | "drop"
    spool_dir: Path | None = None

    flush_interval: float = 2.0