from __future__ import annotations

from typing import Any, Optional

from .client import Client
from .exceptions import NotInitializedError

_client: Optional[Client] = None


def init(
    host: str,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
    *,
    timeout: float = 10.0,
    verify_ssl: bool = True,
    offline_mode: str = "queue",  # "queue" | "raise" | "drop"
    spool_dir: str | None = None,
    flush_interval: float = 2.0,
    start_flush_thread: bool = True,
) -> Client:
    """
    Initialize the global client.

    You can provide either:
      - username + password (token fetched via /api/token), or
      - token directly.
    """
    global _client
    _client = Client(
        host=host,
        username=username,
        password=password,
        token=token,
        timeout=timeout,
        verify_ssl=verify_ssl,
        offline_mode=offline_mode,
        spool_dir=spool_dir,
        flush_interval=flush_interval,
    )
    if start_flush_thread:
        _client.start_background_flush()
    return _client


def _c() -> Client:
    if _client is None:
        raise NotInitializedError("mlserverpy.init(...) must be called first.")
    return _client


def run(
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
    """
    Create (or reuse) an active run and set it as current.

    reuse_if_active:
      - True: if a run is already active in this process, return it.
      - False: always create a new run.
    """
    return _c().run(
        name=name,
        dataset=dataset,
        methods=methods or [],
        num_dummy=num_dummy,
        iterations=iterations,
        lr=lr,
        num_classes=num_classes,
        gt_label=gt_label,
        extra=extra,
        reuse_if_active=reuse_if_active,
    )


def log_metric(
    *,
    method: str,
    metric: str,
    value: float,
    step: int | None = None,
    run_id: str | None = None,
) -> None:
    """
    High-level metric logging.

    Buffers points locally and periodically flushes full series to the server.
    """
    _c().log_metric(run_id=run_id, method=method, metric=metric, value=value, step=step)


def log_scalar(
    *,
    method: str,
    key: str,
    value: Any,
    run_id: str | None = None,
) -> None:
    """
    Store scalar metadata under metrics[method][key] on the server.
    Example: final_loss, final_mse, predicted_label, etc.
    """
    _c().log_scalar(run_id=run_id, method=method, key=key, value=value)


def post(
    *,
    kind: str,
    run_id: str | None = None,
    path: str | None = None,
    filename: str | None = None,
    text: str | None = None,
    figure: Any | None = None,
) -> None:
    """
    Generic artifact/log upload dispatcher.

    kind:
      - "image": requires path (image file)
      - "data":  requires path (data file)
      - "log":   requires text (and optional filename)
      - "figure": requires figure (matplotlib Figure) and optional filename
    """
    c = _c()
    rid = run_id or c.current_run_id
    if not rid:
        raise ValueError("No run_id provided and no active run. Call mlserverpy.run(...) first.")

    kind = kind.lower().strip()
    if kind == "image":
        if not path:
            raise ValueError("post(kind='image') requires path=...")
        c.post_image(run_id=rid, path=path)
    elif kind == "data":
        if not path:
            raise ValueError("post(kind='data') requires path=...")
        c.post_data(run_id=rid, path=path)
    elif kind == "log":
        if text is None:
            raise ValueError("post(kind='log') requires text=...")
        c.post_log(run_id=rid, text=text, filename=filename or "run.log")
    elif kind == "figure":
        if figure is None:
            raise ValueError("post(kind='figure') requires figure=...")
        c.upload_figure(run_id=rid, figure=figure, filename=filename or "figure.png")
    else:
        raise ValueError(f"Unknown kind={kind!r}")


def post_image(path: str, *, run_id: str | None = None) -> None:
    c = _c()
    rid = run_id or c.current_run_id
    if not rid:
        raise ValueError("No active run. Call mlserverpy.run(...) first.")
    c.post_image(run_id=rid, path=path)


def post_data(path: str, *, run_id: str | None = None) -> None:
    c = _c()
    rid = run_id or c.current_run_id
    if not rid:
        raise ValueError("No active run. Call mlserverpy.run(...) first.")
    c.post_data(run_id=rid, path=path)


def post_log(text: str, *, filename: str = "run.log", run_id: str | None = None) -> None:
    c = _c()
    rid = run_id or c.current_run_id
    if not rid:
        raise ValueError("No active run. Call mlserverpy.run(...) first.")
    c.post_log(run_id=rid, text=text, filename=filename)


def upload_figure(figure: Any, *, filename: str = "figure.png", run_id: str | None = None) -> None:
    c = _c()
    rid = run_id or c.current_run_id
    if not rid:
        raise ValueError("No active run. Call mlserverpy.run(...) first.")
    c.upload_figure(run_id=rid, figure=figure, filename=filename)


def heartbeat(
    *,
    status: str = "running",
    step: int | None = None,
    run_id: str | None = None,
) -> None:
    """
    Heartbeat without requiring a server endpoint change.
    Stores scalars on a special method name "__run__".
    """
    c = _c()
    rid = run_id or c.current_run_id
    if not rid:
        raise ValueError("No active run. Call mlserverpy.run(...) first.")
    c.heartbeat(run_id=rid, status=status, step=step)


def flush(*, run_id: str | None = None) -> None:
    """
    Force flush buffered metrics + attempt to drain spool queue.
    """
    c = _c()
    c.flush(run_id=run_id)
    c.sync(run_id=run_id)


def sync(*, run_id: str | None = None) -> None:
    """
    Replay queued offline events from disk.
    """
    _c().sync(run_id=run_id)


def get(*args: Any, **kwargs: Any) -> None:
    """
    Blank template for future retrieval functionality.
    """
    raise NotImplementedError("mlserverpy.get(...) is not implemented yet.")