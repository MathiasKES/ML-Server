import json, os, time
from pathlib import Path

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def append_jsonl(path: Path, obj: dict):
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
