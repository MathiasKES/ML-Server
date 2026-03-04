from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import requests
from requests import Response

from .config import Settings
from .exceptions import AuthError, RequestFailedError
from .upload import copy_into_spool, figure_to_png_bytes
from .utils import (
    now_iso,
    default_spool_dir,
    ensure_dir,
    append_jsonl,
    read_jsonl,
    atomic_write_text,
)

_logger = logging.getLogger(__name__)


class Client:
    """
    A single client instance that holds:
      - server connection and token
      - current run_id
      - buffered metrics series
      - offline spool queue
      - optional background flush thread
    """

    def __init__(
        self,
        *,
        host: str,
        username: str | None,
        password: str | None,
        token: str | None,
        timeout: float = 10.0,
        verify_ssl: bool = True,
        offline_mode: str = "queue",
        spool_dir: str | None = None,
        flush_interval: float = 2.0,
    ):
        spool_base = Path(spool_dir) if spool_dir else default_spool_dir()

        self.settings = Settings(
            host=host.rstrip("/"),
            username=username,
            password=password,
            token=token,
            timeout=timeout,
            verify_ssl=verify_ssl,
            offline_mode=offline_mode,
            spool_dir=spool_base,
            flush_interval=flush_interval,
        )

        self.current_run_id: str | None = None

        self._session = requests.Session()
        self._lock = threading.Lock()

        # metrics_buffer[run_id][method][metric] = list[float]
        self._metrics_buffer: dict[str, dict[str, dict[str, list[float]]]] = {}
        self._last_flush_at: dict[str, float] = {}

        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None

        if self.settings.token is None:
            self._authenticate()
        else:
            # Validate token format minimally
            if not isinstance(self.settings.token, str) or not self.settings.token:
                raise AuthError("Provided token is empty or invalid.")

    # ---------------------------
    # Auth
    # ---------------------------

    def _authenticate(self) -> None:
        if not self.settings.username or not self.settings.password:
            raise AuthError("username/password required to fetch token.")
        r = self._session.post(
            f"{self.settings.host}/api/token",
            json={"username": self.settings.username, "password": self.settings.password},
            timeout=self.settings.timeout,
            verify=self.settings.verify_ssl,
        )
        if r.status_code != 200:
            raise AuthError(f"Token request failed: {r.status_code} {r.text}")
        self.settings.token = r.json().get("token")
        if not self.settings.token:
            raise AuthError("Server did not return a token.")

    def _headers(self) -> dict[str, str]:
        tok = self.settings.token
        if not tok:
            return {}
        return {"Authorization": f"Bearer {tok}"}

    # ---------------------------
    # Core request with offline handling
    # ---------------------------

    def _spool_paths(self, run_id: str) -> tuple[Path, Path]:
        base = ensure_dir(self.settings.spool_dir or default_spool_dir())
        run_dir = ensure_dir(base / "spool" / run_id)
        artifacts_dir = ensure_dir(run_dir / "artifacts")
        events_path = run_dir / "events.jsonl"
        return events_path, artifacts_dir

    def _enqueue_event(self, run_id: str, event: dict[str, Any]) -> None:
        events_path, _ = self._spool_paths(run_id)
        event = dict(event)
        event.setdefault("ts", now_iso())
        append_jsonl(events_path, event)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        run_id_for_spool: str | None = None,
    ) -> Response:
        url = f"{self.settings.host}{path}"
        try:
            r = self._session.request(
                method=method,
                url=url,
                json=json_body,
                headers=self._headers(),
                timeout=self.settings.timeout,
                verify=self.settings.verify_ssl,
            )
        except requests.RequestException as e:
            if self.settings.offline_mode == "queue" and run_id_for_spool:
                self._enqueue_event(run_id_for_spool, {
                    "type": "json_request",
                    "method": method,
                    "path": path,
                    "json": json_body,
                })
                return _FakeResponse(202, "queued")
            if self.settings.offline_mode == "drop":
                return _FakeResponse(202, "dropped")
            raise RequestFailedError(f"Request failed: {e}") from e

        # Unauthorized: try one re-auth if creds exist
        if r.status_code == 401 and self.settings.username and self.settings.password:
            self._authenticate()
            r = self._session.request(
                method=method,
                url=url,
                json=json_body,
                headers=self._headers(),
                timeout=self.settings.timeout,
                verify=self.settings.verify_ssl,
            )

        if r.status_code >= 500 and self.settings.offline_mode == "queue" and run_id_for_spool:
            self._enqueue_event(run_id_for_spool, {
                "type": "json_request",
                "method": method,
                "path": path,
                "json": json_body,
            })
            return _FakeResponse(202, "queued")

        if r.status_code >= 400:
            raise RequestFailedError(f"{method} {path} failed: {r.status_code} {r.text}", status_code=r.status_code)

        return r

    def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        files: dict[str, Any],
        run_id_for_spool: str | None = None,
        spool_copy_paths: dict[str, str] | None = None,
    ) -> Response:
        url = f"{self.settings.host}{path}"
        try:
            r = self._session.request(
                method=method,
                url=url,
                files=files,
                headers=self._headers(),
                timeout=self.settings.timeout,
                verify=self.settings.verify_ssl,
            )
        except requests.RequestException as e:
            if self.settings.offline_mode == "queue" and run_id_for_spool:
                event: dict[str, Any] = {
                    "type": "multipart_request",
                    "method": method,
                    "path": path,
                    "spool_files": {},
                }
                if spool_copy_paths:
                    events_path, artifacts_dir = self._spool_paths(run_id_for_spool)
                    _ = events_path  # just to ensure dirs exist
                    for field, src in spool_copy_paths.items():
                        copied = copy_into_spool(src, artifacts_dir)
                        event["spool_files"][field] = str(copied)
                self._enqueue_event(run_id_for_spool, event)
                return _FakeResponse(202, "queued")
            if self.settings.offline_mode == "drop":
                return _FakeResponse(202, "dropped")
            raise RequestFailedError(f"Request failed: {e}") from e

        if r.status_code == 401 and self.settings.username and self.settings.password:
            self._authenticate()
            r = self._session.request(
                method=method,
                url=url,
                files=files,
                headers=self._headers(),
                timeout=self.settings.timeout,
                verify=self.settings.verify_ssl,
            )

        if r.status_code >= 500 and self.settings.offline_mode == "queue" and run_id_for_spool:
            event = {
                "type": "multipart_request",
                "method": method,
                "path": path,
                "spool_files": {},
            }
            if spool_copy_paths:
                events_path, artifacts_dir = self._spool_paths(run_id_for_spool)
                _ = events_path
                for field, src in spool_copy_paths.items():
                    copied = copy_into_spool(src, artifacts_dir)
                    event["spool_files"][field] = str(copied)
            self._enqueue_event(run_id_for_spool, event)
            return _FakeResponse(202, "queued")

        if r.status_code >= 400:
            raise RequestFailedError(f"{method} {path} failed: {r.status_code} {r.text}", status_code=r.status_code)

        return r

    # ---------------------------
    # Run lifecycle
    # ---------------------------

    def run(
        self,
        *,
        name: str,
        dataset: str,
        methods: list[str],
        num_dummy: int,
        iterations: int,
        lr: float | None,
        num_classes: int | None,
        gt_label: int | None,
        extra: dict[str, Any] | None,
        reuse_if_active: bool,
    ) -> str:
        with self._lock:
            if reuse_if_active and self.current_run_id:
                return self.current_run_id

        payload: dict[str, Any] = {
            "name": name,
            "dataset": dataset,
            "methods": methods,
            "num_dummy": num_dummy,
            "iterations": iterations,
            "lr": lr,
            "num_classes": num_classes,
            "gt_label": gt_label,
        }
        if extra:
            payload.update(extra)

        r = self._request_json("POST", "/api/runs", json_body=payload, run_id_for_spool=None)
        run_id = r.json().get("run_id")
        if not run_id:
            raise RequestFailedError("Server did not return run_id.")

        with self._lock:
            self.current_run_id = run_id
            self._metrics_buffer.setdefault(run_id, {})
            self._last_flush_at.setdefault(run_id, 0.0)
        return run_id

    # ---------------------------
    # Metrics logging (buffered)
    # ---------------------------

    def log_metric(self, *, run_id: str | None, method: str, metric: str, value: float, step: int | None) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No run_id provided and no active run.")

        with self._lock:
            m = self._metrics_buffer.setdefault(rid, {}).setdefault(method, {}).setdefault(metric, [])
            m.append(float(value))

        # Optional: include step as a parallel series if user supplies it
        if step is not None:
            with self._lock:
                s = self._metrics_buffer.setdefault(rid, {}).setdefault(method, {}).setdefault("step", [])
                s.append(int(step))

    def log_scalar(self, *, run_id: str | None, method: str, key: str, value: Any) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            raise ValueError("No run_id provided and no active run.")
        payload = {"method": method, "scalars": {key: value}}
        self._request_json("POST", f"/api/runs/{rid}/metrics", json_body=payload, run_id_for_spool=rid)

    def flush(self, *, run_id: str | None = None) -> None:
        """
        Flush buffered series (full arrays) for each method.
        """
        rid = run_id or self.current_run_id
        if not rid:
            return

        with self._lock:
            by_method = json.loads(json.dumps(self._metrics_buffer.get(rid, {})))  # deep copy
            self._last_flush_at[rid] = time.time()

        for method, series in by_method.items():
            if not series:
                continue
            payload = {"method": method, "series": series}
            self._request_json(
                "POST",
                f"/api/runs/{rid}/metrics",
                json_body=payload,
                run_id_for_spool=rid,
            )

        # Clear the buffer after successful flush
        with self._lock:
            if rid in self._metrics_buffer:
                self._metrics_buffer[rid] = {}

    # ---------------------------
    # Artifacts and logs
    # ---------------------------

    def post_image(self, *, run_id: str, path: str) -> None:
        p = Path(path)
        with open(p, "rb") as f:
            files = {"file": (p.name, f)}
            self._request_multipart(
                "POST",
                f"/api/runs/{run_id}/images",
                files=files,
                run_id_for_spool=run_id,
                spool_copy_paths={"file": str(p)},
            )

    def post_data(self, *, run_id: str, path: str) -> None:
        p = Path(path)
        with open(p, "rb") as f:
            files = {"file": (p.name, f)}
            self._request_multipart(
                "POST",
                f"/api/runs/{run_id}/data",
                files=files,
                run_id_for_spool=run_id,
                spool_copy_paths={"file": str(p)},
            )

    def post_log(self, *, run_id: str, text: str, filename: str = "run.log", append: bool = True) -> None:
        payload = {"filename": filename, "content": text, "append": bool(append)}
        self._request_json(
            "POST",
            f"/api/runs/{run_id}/logs",
            json_body=payload,
            run_id_for_spool=run_id,
        )

    def upload_figure(self, *, run_id: str, figure: Any, filename: str = "figure.png") -> None:
        png = figure_to_png_bytes(figure)
        files = {"file": (filename, png, "image/png")}
        self._request_multipart(
            "POST",
            f"/api/runs/{run_id}/images",
            files=files,
            run_id_for_spool=run_id,
            spool_copy_paths=None,
        )

    def heartbeat(self, *, run_id: str, status: str = "running", step: int | None = None) -> None:
        scalars: dict[str, Any] = {
            "status": status,
            "heartbeat_at": now_iso(),
        }
        if step is not None:
            scalars["step"] = int(step)
        payload = {"method": "__run__", "scalars": scalars}
        self._request_json("POST", f"/api/runs/{run_id}/metrics", json_body=payload, run_id_for_spool=run_id)

    # ---------------------------
    # Offline spool replay
    # ---------------------------

    def sync(self, *, run_id: str | None = None, max_events: int = 500) -> None:
        rid = run_id or self.current_run_id
        if not rid:
            return

        events_path, artifacts_dir = self._spool_paths(rid)
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
                et = ev.get("type")
                if et == "json_request":
                    self._request_json(
                        ev.get("method", "POST"),
                        ev.get("path", ""),
                        json_body=ev.get("json"),
                        run_id_for_spool=None,  # do not re-enqueue on replay
                    )
                elif et == "multipart_request":
                    spool_files: dict[str, str] = ev.get("spool_files", {}) or {}
                    files: dict[str, Any] = {}
                    file_handles: list[Any] = []
                    try:
                        for field, pstr in spool_files.items():
                            p = Path(pstr)
                            if not p.exists():
                                raise FileNotFoundError(f"Missing spooled file: {p}")
                            fh = open(p, "rb")
                            file_handles.append(fh)
                            files[field] = (p.name, fh)
                        self._request_multipart(
                            ev.get("method", "POST"),
                            ev.get("path", ""),
                            files=files,
                            run_id_for_spool=None,
                            spool_copy_paths=None,
                        )
                    finally:
                        for fh in file_handles:
                            try:
                                fh.close()
                            except Exception:
                                pass
                else:
                    # Unknown event type, keep it
                    remaining.append(ev)
                    continue

                sent += 1
            except Exception:
                remaining.append(ev)

        # rewrite queue with remaining events
        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in remaining) + ("\n" if remaining else "")
        atomic_write_text(events_path, text)

        # If all events were synced, clean up the spool directory
        if not remaining:
            run_spool_dir = events_path.parent
            try:
                shutil.rmtree(run_spool_dir, ignore_errors=True)
            except Exception:
                pass

    # ---------------------------
    # Background flush loop
    # ---------------------------

    def start_background_flush(self) -> None:
        if self._flush_thread and self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop_background_flush(self) -> None:
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(max(0.25, float(self.settings.flush_interval)))
            with self._lock:
                rid = self.current_run_id
            if not rid:
                continue

            # Heartbeat is cheap and also tests connectivity
            try:
                self.heartbeat(run_id=rid, status="running")
            except Exception as e:
                _logger.debug("Heartbeat failed for run %s: %s", rid, e)

            try:
                self.flush(run_id=rid)
            except Exception as e:
                _logger.debug("Flush failed for run %s: %s", rid, e)

            try:
                self.sync(run_id=rid, max_events=200)
            except Exception as e:
                _logger.debug("Sync failed for run %s: %s", rid, e)


class _FakeResponse:
    """
    Minimal Response-like object used when offline_mode queues or drops.
    Implements common Response interface methods to prevent AttributeErrors.
    """
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.encoding = "utf-8"
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return {"status": self.text}

    def raise_for_status(self) -> None:
        """No-op for offline responses."""
        pass