from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import requests

from .exceptions import AuthError, RequestFailedError
from .utils import default_spool_dir, ensure_dir, append_jsonl, read_jsonl, atomic_write_text, now_iso

import io
import os

if TYPE_CHECKING:
    # Only imported for type-checking; matplotlib is an optional dependency
    import matplotlib.figure

class Client:
    """Client for ML-Server.

    - Always fetches a Bearer token from /api/token using username/password.
    - Buffers metrics with log_metric() and uploads full series on flush().
    - If offline_mode='queue', failed requests are persisted to disk and can be replayed by sync().
    """

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        timeout: float = 10.0,
        verify_ssl: bool = True,
        offline_mode: str = "queue",   # "queue" | "drop" | "raise"
        spool_dir: str | None = None,
        flush_interval: float = 2.0,
        start_background_flush: bool = False,
    ) -> None:
        self.host = host.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)
        self.offline_mode = offline_mode
        self.spool_dir = Path(spool_dir) if spool_dir else default_spool_dir()
        self.flush_interval = float(flush_interval)

        self._session = requests.Session()
        self._token: str | None = None

        self.current_run_id: str | None = None

        # metrics_buffer[run_id][method][metric] = list
        self._metrics_buffer: Dict[str, Dict[str, Dict[str, list]]] = {}
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._authenticate()

        if start_background_flush:
            self.start_background_flush()

    # ---- auth ----

    def _authenticate(self) -> None:
        r = self._session.post(
            f"{self.host}/api/token",
            json={"username": self.username, "password": self.password},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        if r.status_code != 200:
            raise AuthError(f"Token request failed: {r.status_code} {r.text}")
        token = r.json().get("token")
        if not token:
            raise AuthError("Server did not return a token.")
        self._token = token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    # ---- spooling ----

    def _spool_paths(self, run_id: str) -> Tuple[Path, Path]:
        base = ensure_dir(self.spool_dir)
        run_dir = ensure_dir(base / "spool" / run_id)
        artifacts_dir = ensure_dir(run_dir / "artifacts")
        events_path = run_dir / "events.jsonl"
        return events_path, artifacts_dir

    def _enqueue(self, run_id: str, event: dict[str, Any]) -> None:
        events_path, _ = self._spool_paths(run_id)
        ev = dict(event)
        ev.setdefault("ts", now_iso())
        append_jsonl(events_path, ev)

    # ---- http helpers ----

    def _request_json(self, method: str, path: str, json_body: dict[str, Any] | None, run_id_for_spool: str | None):
        url = f"{self.host}{path}"
        try:
            r = self._session.request(
                method=method,
                url=url,
                json=json_body,
                headers=self._headers(),
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.RequestException as e:
            if self.offline_mode == "queue" and run_id_for_spool:
                self._enqueue(run_id_for_spool, {"type": "json", "method": method, "path": path, "json": json_body})
                return None
            if self.offline_mode == "drop":
                return None
            raise RequestFailedError(f"Request failed: {e}") from e

        if r.status_code == 401:
            self._authenticate()
            return self._request_json(method, path, json_body, run_id_for_spool)

        if r.status_code >= 500 and self.offline_mode == "queue" and run_id_for_spool:
            self._enqueue(run_id_for_spool, {"type": "json", "method": method, "path": path, "json": json_body})
            return None

        if r.status_code >= 400:
            raise RequestFailedError(f"{method} {path} failed: {r.status_code} {r.text}", status_code=r.status_code)

        return r

    def _request_file(self, path: str, file_path: str, run_id_for_spool: str | None):
        url = f"{self.host}{path}"
        p = Path(file_path)
        try:
            with open(p, "rb") as f:
                files = {"file": (p.name, f)}
                r = self._session.post(
                    url,
                    files=files,
                    headers=self._headers(),
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
        except (requests.RequestException, OSError) as e:
            if self.offline_mode == "queue" and run_id_for_spool:
                events_path, artifacts_dir = self._spool_paths(run_id_for_spool)
                _ = events_path
                copied = artifacts_dir / p.name
                if copied.resolve() != p.resolve():
                    copied.write_bytes(p.read_bytes())
                self._enqueue(run_id_for_spool, {"type": "file", "path": path, "file": str(copied)})
                return None
            if self.offline_mode == "drop":
                return None
            raise RequestFailedError(f"Upload failed: {e}") from e

        if r.status_code == 401:
            self._authenticate()
            return self._request_file(path, file_path, run_id_for_spool)

        if r.status_code >= 500 and self.offline_mode == "queue" and run_id_for_spool:
            events_path, artifacts_dir = self._spool_paths(run_id_for_spool)
            _ = events_path
            copied = artifacts_dir / p.name
            if copied.resolve() != p.resolve():
                copied.write_bytes(p.read_bytes())
            self._enqueue(run_id_for_spool, {"type": "file", "path": path, "file": str(copied)})
            return None

        if r.status_code >= 400:
            raise RequestFailedError(f"POST {path} failed: {r.status_code} {r.text}", status_code=r.status_code)

        return r

    # ---- public API ----

    def run(
        self,
        *,
        name: str,
        dataset: str = "unknown",
        methods: list[str] | None = None,
        num_dummy: int = 1,
        iterations: int = 0,
        lr: float | None = None,
        num_classes: int | None = None,
        gt_label: int | None = None,
        extra: dict[str, Any] | None = None,
        reuse_if_active: bool = True,
    ) -> str:
        if reuse_if_active and self.current_run_id:
            return self.current_run_id

        payload: dict[str, Any] = {
            "name": name,
            "dataset": dataset,
            "methods": methods or [],
            "num_dummy": num_dummy,
            "iterations": iterations,
            "lr": lr,
            "num_classes": num_classes,
            "gt_label": gt_label,
        }
        if extra:
            payload.update(extra)

        r = self._request_json("POST", "/api/runs", payload, run_id_for_spool=None)
        if r is None:
            raise RequestFailedError("Cannot create run while offline (run creation is not queued).")
        rid = r.json().get("run_id")
        if not rid:
            raise RequestFailedError("Server did not return run_id.")
        self.current_run_id = rid
        with self._lock:
            self._metrics_buffer.setdefault(rid, {})
        return rid

    def log_metric(self, *, method: str, metric: str, value: float, step: int | None = None, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        with self._lock:
            self._metrics_buffer.setdefault(rid, {}).setdefault(method, {}).setdefault(metric, []).append(float(value))
            if step is not None:
                self._metrics_buffer.setdefault(rid, {}).setdefault(method, {}).setdefault("step", []).append(int(step))

    def log_scalar(self, *, method: str, key: str, value: Any, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        payload = {"method": method, "scalars": {key: value}}
        self._request_json("POST", f"/api/runs/{rid}/metrics", payload, run_id_for_spool=rid)

    def flush(self, *, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            return
        with self._lock:
            buffered = copy.deepcopy(self._metrics_buffer.get(rid, {}))
            self._metrics_buffer[rid] = {}

        for method, series in buffered.items():
            if not series:
                continue
            payload = {"method": method, "series": series}
            self._request_json("POST", f"/api/runs/{rid}/metrics", payload, run_id_for_spool=rid)

    def post_image(self, *, path: str, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        self._request_file(f"/api/runs/{rid}/images", path, run_id_for_spool=rid)

    def post_image(self, run_id: str, path: "str | None" = None, *, figure: "matplotlib.figure.Figure | None" = None, filename: str = "figure.png", fmt: str = "png") -> None:
        if path is not None and figure is not None:
            raise ValueError("Supply either 'path' or 'figure', not both.")
        if path is None and figure is None:
            raise ValueError("Supply either 'path' or 'figure'.")

        if path is not None:
            # ── original behaviour ────────────────────────────────────────────
            with open(path, "rb") as fh:
                data = fh.read()
            upload_filename = os.path.basename(path)
        else:
            # ── matplotlib figure → in-memory bytes ───────────────────────────
            buf = io.BytesIO()
            figure.savefig(buf, format=fmt)
            buf.seek(0)
            data = buf.read()
            upload_filename = filename

        response = self._session.post(
            f"{self._host}/api/runs/{run_id}/images",
            files={"file": (upload_filename, data, f"image/{fmt}")},
        )
        response.raise_for_status()

        self._upload_image_bytes(run_id, upload_filename, data, fmt)

    def _upload_image_bytes(self, run_id: str, filename: str, data: bytes, fmt: str) -> None:
        """POST raw image bytes to the server.

        This helper centralises the actual HTTP call so that post_image stays
        readable regardless of how many input forms it supports.
        """
        response = self._session.post(
            f"{self._host}/api/runs/{run_id}/images",
            files={"file": (filename, data, f"image/{fmt}")},
        )
        response.raise_for_status()

    def post_data(self, *, path: str, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        self._request_file(f"/api/runs/{rid}/data", path, run_id_for_spool=rid)

    def post_log(self, *, text: str, filename: str = "run.log", append: bool = True, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        payload = {"filename": filename, "content": text, "append": bool(append)}
        self._request_json("POST", f"/api/runs/{rid}/logs", payload, run_id_for_spool=rid)

    def sync(self, *, run_id: str | None = None, max_events: int = 500) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            return
        events_path, _ = self._spool_paths(rid)
        events = read_jsonl(events_path)
        if not events:
            return

        remaining: list[dict[str, Any]] = []
        sent = 0
        for ev in events:
            if sent >= max_events:
                remaining.append(ev)
                continue
            try:
                if ev.get("type") == "json":
                    r = self._request_json(ev.get("method", "POST"), ev.get("path", ""), ev.get("json"), run_id_for_spool=None)
                    if r is None:
                        remaining.append(ev)
                        continue
                elif ev.get("type") == "file":
                    r = self._request_file(ev.get("path", ""), ev.get("file", ""), run_id_for_spool=None)
                    if r is None:
                        remaining.append(ev)
                        continue
                else:
                    remaining.append(ev)
                    continue
                sent += 1
            except Exception:
                remaining.append(ev)

        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in remaining) + ("\n" if remaining else "")
        atomic_write_text(events_path, text)

    def start_background_flush(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_background_flush(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(max(0.25, self.flush_interval))
            if self.current_run_id:
                try:
                    self.flush()
                except Exception:
                    pass
                try:
                    self.sync(max_events=200)
                except Exception:
                    pass

    