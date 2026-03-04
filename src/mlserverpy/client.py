from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .exceptions import AuthError, RequestFailedError
from .utils import (
    append_jsonl, atomic_write_text, default_spool_dir,
    ensure_dir, now_iso, read_jsonl,
)


class Client:
    """Client for ML-Server.

    Buffers metrics locally with log_metric() and flushes full series to the
    server either manually (flush()) or via a background thread when
    flush_interval > 0.

    offline_mode:
        "queue"  — store failed requests on disk; replay with sync()
        "drop"   — silently discard failures
        "raise"  — raise RequestFailedError immediately
    """

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        timeout: float = 10.0,
        verify_ssl: bool = True,
        offline_mode: str = "queue",
        spool_dir: str | None = None,
        flush_interval: float = 0.0,
    ) -> None:
        self.host         = host.rstrip("/")
        self.username     = username
        self.password     = password
        self.timeout      = float(timeout)
        self.verify_ssl   = bool(verify_ssl)
        self.offline_mode = offline_mode
        self.spool_dir    = Path(spool_dir) if spool_dir else default_spool_dir()
        self.flush_interval = float(flush_interval)

        self._session = requests.Session()
        self._token: str | None = None
        self.current_run_id: str | None = None

        # metrics_buffer[run_id][method][metric] = list[float]
        self._metrics_buffer: dict[str, dict[str, dict[str, list]]] = {}
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None

        self._authenticate()

        if self.flush_interval > 0:
            self._start_background_flush()

    # ── Auth ───────────────────────────────────────────────────────────────────

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

    # ── Spooling ───────────────────────────────────────────────────────────────

    def _run_spool_dir(self, run_id: str) -> Path:
        return ensure_dir(self.spool_dir / "spool" / run_id)

    def _events_path(self, run_id: str) -> Path:
        return self._run_spool_dir(run_id) / "events.jsonl"

    def _enqueue(self, run_id: str, event: dict[str, Any]) -> None:
        ev = {**event, "ts": now_iso()}
        append_jsonl(self._events_path(run_id), ev)

    def _spool_file(self, run_id: str, api_path: str, file_path: Path) -> None:
        """Copy a file into the spool directory and record an event for it."""
        artifacts_dir = ensure_dir(self._run_spool_dir(run_id) / "artifacts")
        dest = artifacts_dir / file_path.name
        if dest.resolve() != file_path.resolve():
            dest.write_bytes(file_path.read_bytes())
        self._enqueue(run_id, {"type": "file", "path": api_path, "file": str(dest)})

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        spool_run_id: str | None = None,
    ):
        url = f"{self.host}{path}"
        try:
            r = self._session.request(
                method=method, url=url, json=body,
                headers=self._headers(), timeout=self.timeout, verify=self.verify_ssl,
            )
        except requests.RequestException as e:
            return self._handle_failure(
                lambda: self._enqueue(spool_run_id, {"type": "json", "method": method, "path": path, "json": body}),
                spool_run_id, e,
            )

        if r.status_code == 401:
            self._authenticate()
            return self._request_json(method, path, body, spool_run_id)

        if r.status_code >= 500:
            return self._handle_failure(
                lambda: self._enqueue(spool_run_id, {"type": "json", "method": method, "path": path, "json": body}),
                spool_run_id, RequestFailedError(f"{method} {path} → {r.status_code}", r.status_code),
            )

        if r.status_code >= 400:
            raise RequestFailedError(f"{method} {path} → {r.status_code} {r.text}", r.status_code)

        return r

    def _request_file(self, path: str, file_path: str, spool_run_id: str | None = None):
        url = f"{self.host}{path}"
        p   = Path(file_path)
        try:
            with open(p, "rb") as fh:
                r = self._session.post(
                    url, files={"file": (p.name, fh)},
                    headers=self._headers(), timeout=self.timeout, verify=self.verify_ssl,
                )
        except (requests.RequestException, OSError) as e:
            return self._handle_failure(
                lambda: self._spool_file(spool_run_id, path, p),
                spool_run_id, e,
            )

        if r.status_code == 401:
            self._authenticate()
            return self._request_file(path, file_path, spool_run_id)

        if r.status_code >= 500:
            return self._handle_failure(
                lambda: self._spool_file(spool_run_id, path, p),
                spool_run_id, RequestFailedError(f"POST {path} → {r.status_code}", r.status_code),
            )

        if r.status_code >= 400:
            raise RequestFailedError(f"POST {path} → {r.status_code} {r.text}", r.status_code)

        return r

    def _handle_failure(self, spool_fn, spool_run_id: str | None, exc: Exception):
        """Apply offline_mode policy; returns None on queue/drop, raises on 'raise'."""
        if self.offline_mode == "queue" and spool_run_id:
            spool_fn()
            return None
        if self.offline_mode == "drop":
            return None
        raise RequestFailedError(str(exc)) from exc

    # ── Public API ─────────────────────────────────────────────────────────────

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
            "name": name, "dataset": dataset, "methods": methods or [],
            "num_dummy": num_dummy, "iterations": iterations,
            "lr": lr, "num_classes": num_classes, "gt_label": gt_label,
        }
        if extra:
            payload.update(extra)

        r = self._request_json("POST", "/api/runs", payload)
        if r is None:
            raise RequestFailedError("Cannot create run while offline.")
        rid = r.json().get("run_id")
        if not rid:
            raise RequestFailedError("Server did not return run_id.")
        self.current_run_id = rid
        with self._lock:
            self._metrics_buffer.setdefault(rid, {})
        return rid

    def log_metric(
        self,
        *,
        method: str,
        metric: str,
        value: float,
        step: int | None = None,
        run_id: str | None = None,
    ) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        with self._lock:
            method_buf = (
                self._metrics_buffer
                .setdefault(rid, {})
                .setdefault(method, {})
            )
            method_buf.setdefault(metric, []).append(float(value))
            if step is not None:
                method_buf.setdefault("step", []).append(int(step))

    def log_scalar(self, *, method: str, key: str, value: Any, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        self._request_json("POST", f"/api/runs/{rid}/metrics", {"method": method, "scalars": {key: value}}, rid)

    def flush(self, *, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            return
        with self._lock:
            buffered = copy.deepcopy(self._metrics_buffer.pop(rid, {}))
            self._metrics_buffer[rid] = {}

        for method, series in buffered.items():
            if series:
                self._request_json("POST", f"/api/runs/{rid}/metrics", {"method": method, "series": series}, rid)

    def post_image(
        self,
        *,
        run_id: str | None = None,
        path: str | None = None,
        figure=None,
        filename: str = "figure.png",
        fmt: str = "png",
    ) -> None:
        """Upload an image. Pass either a file path or a matplotlib Figure."""
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        if (path is None) == (figure is None):
            raise ValueError("Supply exactly one of 'path' or 'figure'.")

        if path is not None:
            self._request_file(f"/api/runs/{rid}/images", path, rid)
        else:
            import io
            buf = io.BytesIO()
            figure.savefig(buf, format=fmt)
            buf.seek(0)
            url = f"{self.host}/api/runs/{rid}/images"
            try:
                r = self._session.post(
                    url,
                    files={"file": (filename, buf, f"image/{fmt}")},
                    headers=self._headers(),
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                if r.status_code == 401:
                    self._authenticate()
                    buf.seek(0)
                    r = self._session.post(
                        url,
                        files={"file": (filename, buf, f"image/{fmt}")},
                        headers=self._headers(),
                        timeout=self.timeout,
                        verify=self.verify_ssl,
                    )
                if r.status_code >= 400:
                    raise RequestFailedError(f"POST images → {r.status_code} {r.text}", r.status_code)
            except requests.RequestException as e:
                if self.offline_mode == "raise":
                    raise RequestFailedError(str(e)) from e

    def post_data(self, *, path: str, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        self._request_file(f"/api/runs/{rid}/data", path, rid)

    def post_log(self, *, text: str, filename: str = "run.log", append: bool = True, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No active run. Call client.run(...) first.")
        self._request_json("POST", f"/api/runs/{rid}/logs", {"filename": filename, "content": text, "append": bool(append)}, rid)

    def sync(self, *, run_id: str | None = None, max_events: int = 500) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            return
        events_path = self._events_path(rid)
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
                    r = self._request_json(ev.get("method", "POST"), ev.get("path", ""), ev.get("json"))
                elif ev.get("type") == "file":
                    r = self._request_file(ev.get("path", ""), ev.get("file", ""))
                else:
                    remaining.append(ev)
                    continue
                if r is None:
                    remaining.append(ev)
                    continue
                sent += 1
            except Exception:
                remaining.append(ev)

        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in remaining)
        atomic_write_text(events_path, text + "\n" if remaining else "")

    # ── Background flush ───────────────────────────────────────────────────────

    def _start_background_flush(self) -> None:
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