import requests, threading, time
from typing import Any, Optional, Dict, List
from .exceptions import RequestFailedError

class Client:

    def __init__(self, host: str, username=None, password=None, token=None):
        self.host = host.rstrip("/")
        self.username = username
        self.password = password
        self.token = token
        self.session = requests.Session()
        self.current_run_id: Optional[str] = None
        self.metrics: Dict[str, Dict[str, List[float]]] = {}
        self.lock = threading.Lock()

    def run(self, name: str, dataset="unknown", methods=None, iterations=0, **extra):
        payload = {
            "name": name,
            "dataset": dataset,
            "methods": methods or [],
            "iterations": iterations
        }
        payload.update(extra)
        r = self.session.post(self.host + "/api/runs", json=payload)
        if r.status_code >= 400:
            raise RequestFailedError(r.text, r.status_code)
        rid = r.json()["run_id"]
        self.current_run_id = rid
        return rid

    def log_metric(self, method: str, metric: str, value: float, run_id=None):
        rid = run_id or self.current_run_id
        if rid is None:
            raise ValueError("No run active")
        with self.lock:
            self.metrics.setdefault(method, {}).setdefault(metric, []).append(value)

    def log_scalar(self, method: str, key: str, value: Any, run_id=None):
        rid = run_id or self.current_run_id
        payload = {"method": method, "scalars": {key: value}}
        self.session.post(f"{self.host}/api/runs/{rid}/metrics", json=payload)

    def flush(self, run_id=None):
        rid = run_id or self.current_run_id
        if not rid:
            return
        for method, series in self.metrics.items():
            payload = {"method": method, "series": series}
            self.session.post(f"{self.host}/api/runs/{rid}/metrics", json=payload)
        self.metrics.clear()

    def post(self, kind: str, path: str, run_id=None):
        rid = run_id or self.current_run_id
        with open(path, "rb") as f:
            files = {"file": (path, f)}
            self.session.post(f"{self.host}/api/runs/{rid}/{kind}", files=files)

    def sync(self, *args, **kwargs):
        pass

    def heartbeat(self, *args, **kwargs):
        pass
