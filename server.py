"""
Machine Learning Experiment Server
Single-user, authenticated. Receives and displays experiment data.

File layout:
    server.py          — this file
    html/
        base.html      — shared layout, CSS, sidebar, lightbox
        login.html     — sign-in page
        dashboard.html — overview & recent runs
        jobs_list.html — all experiments table
        job_detail.html— per-run view (config, charts, images, files, logs)
        api_docs.html  — endpoint reference & Python client snippet

Run:
    pip install flask
    python server.py [--host 0.0.0.0] [--port 5000] [--debug]

Credentials (defaults admin / password):
    ML_USER=admin  ML_PASS=yourpassword  python server.py
"""

import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Flask, Response, abort, jsonify, redirect,
    request, send_from_directory, session, url_for,
)
from jinja2 import Environment, FileSystemLoader
from werkzeug.utils import secure_filename

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("experiment_data")
RUNS_DIR    = DATA_DIR / "runs"
SECRET_FILE = DATA_DIR / ".secret_key"

USERNAME = os.environ.get("ML_USER", "admin")
PASSWORD = os.environ.get("ML_PASS", "password")

ALLOWED_IMG  = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_DATA = {".pkl", ".npy", ".npz", ".json", ".csv", ".txt", ".log"}

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

for _d in [DATA_DIR, RUNS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Secret key — persisted so sessions survive restarts
if SECRET_FILE.exists():
    app.secret_key = SECRET_FILE.read_bytes()
else:
    _key = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(_key)
    app.secret_key = _key

# Precompute password hash once at startup
_PW_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

# Jinja2 environment pointing at the project root (templates live in html/)
_jinja = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    autoescape=True,
)

def render(template_name: str, **ctx) -> str:
    ctx.setdefault("username", USERNAME)
    ctx.setdefault("active_nav", "")
    ctx.setdefault("breadcrumb", "")
    return _jinja.get_template(f"html/{template_name}").render(**ctx)

# In-memory bearer tokens — cleared on restart
_tokens: dict[str, bool] = {}

# ── Auth ───────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def dec(*args, **kwargs):
        if not session.get("authenticated"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return dec

@app.before_request
def bearer_auth():
    """Allow API access via Authorization: Bearer <token>."""
    if request.path.startswith("/api/") and not session.get("authenticated"):
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and _tokens.get(auth[7:]):
            session["authenticated"] = True

# ── Helpers ────────────────────────────────────────────────────────────────────

def human_size(n: int) -> str:
    """Convert a byte count to a human-readable string."""
    v = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if v < 1024:
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} PB"

def get_meta(run_id: str) -> dict | None:
    f = RUNS_DIR / run_id / "meta.json"
    return json.loads(f.read_text()) if f.exists() else None

def save_meta(run_id: str, meta: dict) -> None:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            m = get_meta(d.name)
            if m:
                runs.append(m)
    return runs

def fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts or "—"

def split_metrics(raw: Any) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, Any]]]:
    """
    Split meta["metrics"] into (series, scalars).
      series[method][metric_name]  = list
      scalars[method][key]         = non-list value
    """
    series:  dict[str, dict[str, list]] = {}
    scalars: dict[str, dict[str, Any]]  = {}

    if not isinstance(raw, dict):
        return series, scalars

    for method, md in raw.items():
        if not isinstance(method, str) or not isinstance(md, dict):
            continue
        series[method]  = {k: v for k, v in md.items() if isinstance(k, str) and isinstance(v, list)}
        scalars[method] = {k: v for k, v in md.items() if isinstance(k, str) and not isinstance(v, list)}

    return series, scalars

def run_summary_loss(meta: dict) -> str:
    """Return a short loss summary string, e.g. 'DLG:0.1234  iDLG:0.9876'."""
    ms, mc = split_metrics(meta.get("metrics", {}))
    parts: list[str] = []
    for method in sorted(ms):
        loss = mc.get(method, {}).get("final_loss") or (ms[method].get("loss") or [None])[-1]
        if loss is None:
            continue
        try:
            parts.append(f"{method}:{float(loss):.4f}")
        except Exception:
            parts.append(f"{method}:{loss}")
    return "  ".join(parts)

# ── Auth routes ────────────────────────────────────────────────────────────────

@app.get("/login")
def login_page():
    if session.get("authenticated"):
        return redirect(url_for("dashboard"))
    return render("login.html", error=bool(request.args.get("error")))

@app.post("/login")
def do_login():
    u = request.form.get("username", "")
    p = request.form.get("password", "")
    if u == USERNAME and hashlib.sha256(p.encode()).hexdigest() == _PW_HASH:
        session["authenticated"] = True
        session.permanent = True
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page") + "?error=1")

@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/")
@login_required
def dashboard():
    runs = list_runs()
    dsets = sorted({r.get("dataset", "?") for r in runs})
    meths = sorted({
        m
        for r in runs
        for m in (r.get("methods") or list((r.get("metrics") or {}).keys()))
    })
    display_runs = []
    for r in runs[:8]:
        rr = dict(r)
        rr["created_fmt"] = fmt_ts(rr.get("created_at", ""))
        display_runs.append(rr)
    return render(
        "dashboard.html",
        active_nav="dashboard",
        runs=display_runs,
        total_runs=len(runs),
        num_datasets=len(dsets),
        datasets_str=", ".join(dsets),
        num_methods=len(meths),
        methods_str=", ".join(meths),
    )

@app.get("/jobs")
@login_required
def jobs_list():
    runs = list_runs()
    display = []
    for r in runs:
        rr = dict(r)
        rr["created_fmt"] = fmt_ts(rr.get("created_at", ""))
        rr["loss_str"] = run_summary_loss(rr)
        display.append(rr)
    return render("jobs_list.html", active_nav="jobs", breadcrumb="/ experiments", runs=display)

@app.get("/jobs/<run_id>")
@login_required
def job_detail(run_id: str):
    meta = get_meta(run_id)
    if not meta:
        abort(404)

    rd = RUNS_DIR / run_id

    images     = sorted((rd / "images").glob("*")) if (rd / "images").exists() else []
    data_files = sorted((rd / "data").glob("*"))   if (rd / "data").exists()   else []
    log_files  = sorted((rd / "logs").glob("*"))   if (rd / "logs").exists()   else []

    data_display = [type("F", (), {"name": f.name, "size_str": human_size(f.stat().st_size)})() for f in data_files]

    log_content = ""
    for lf in log_files[:3]:
        try:
            txt = lf.read_text(errors="replace")[-8000:]
        except Exception:
            txt = "(unreadable)"
        log_content += f"=== {lf.name} ===\n{txt}\n\n"

    metrics_series, metrics_scalars = split_metrics(meta.get("metrics", {}))
    methods      = sorted(metrics_series)
    metric_names = sorted({k for md in metrics_series.values() for k in md})
    has_metrics  = any(bool(v) for md in metrics_series.values() for v in md.values())

    name = meta.get("name", run_id)
    return render(
        "job_detail.html",
        active_nav="jobs",
        breadcrumb=f"/ experiments / {name}",
        run_id=run_id,
        name=name,
        meta=meta,
        created_fmt=fmt_ts(meta.get("created_at", "")),
        images=images,
        data_files=data_display,
        log_files=log_files,
        log_content=log_content,
        has_metrics=has_metrics,
        metrics_series=metrics_series,
        metrics_scalars=metrics_scalars,
        methods=methods,
        metric_names=metric_names,
    )

@app.get("/api-docs")
@login_required
def api_docs():
    return render("api_docs.html", active_nav="api", breadcrumb="/ api-docs")

# ── File serving ───────────────────────────────────────────────────────────────

@app.get("/files/<run_id>/<subdir>/<filename>")
@login_required
def serve_file(run_id: str, subdir: str, filename: str):
    if subdir not in {"images", "data", "logs"}:
        abort(404)
    return send_from_directory(RUNS_DIR / run_id / subdir, secure_filename(filename))

# ── API ────────────────────────────────────────────────────────────────────────

@app.post("/api/token")
def api_token():
    d = request.get_json(force=True, silent=True) or {}
    if d.get("username") == USERNAME and hashlib.sha256(d.get("password", "").encode()).hexdigest() == _PW_HASH:
        tok = secrets.token_hex(32)
        _tokens[tok] = True
        return jsonify({"token": tok})
    return jsonify({"error": "Invalid credentials"}), 401

@app.get("/api/storage")
@login_required
def api_storage():
    total = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    return jsonify({"size_bytes": total, "size_human": human_size(total)})

@app.get("/api/runs")
@login_required
def api_list_runs():
    return jsonify(list_runs())

@app.get("/api/runs/<run_id>")
@login_required
def api_get_run(run_id: str):
    m = get_meta(run_id)
    if not m:
        abort(404)
    return jsonify(m)

@app.delete("/api/runs/<run_id>")
@login_required
def api_delete_run(run_id: str):
    rd = RUNS_DIR / run_id
    if not rd.exists():
        abort(404)
    shutil.rmtree(rd)
    return jsonify({"status": "deleted"}), 200

@app.post("/api/runs")
@login_required
def api_create_run():
    d = request.get_json(force=True, silent=True) or {}
    run_id = str(uuid.uuid4())[:12]
    meta = {
        "run_id":          run_id,
        "name":            d.get("name", f"run-{run_id}"),
        "dataset":         d.get("dataset", "unknown"),
        "methods":         d.get("methods", []),
        "num_dummy":       d.get("num_dummy", 1),
        "iterations":      d.get("iterations", 0),
        "lr":              d.get("lr"),
        "num_classes":     d.get("num_classes"),
        "gt_label":        d.get("gt_label"),
        "created_at":      datetime.now().isoformat(),
        "image_count":     0,
        "data_file_count": 0,
        "metrics":         {},
    }
    save_meta(run_id, meta)
    for sub in ["images", "data", "logs"]:
        (RUNS_DIR / run_id / sub).mkdir(parents=True, exist_ok=True)
    return jsonify({"run_id": run_id, "name": meta["name"]}), 201

@app.post("/api/runs/<run_id>/images")
@login_required
def api_upload_image(run_id: str):
    """Multipart (field: file)  OR  JSON { filename, data: <base64> }."""
    meta = get_meta(run_id)
    if not meta:
        abort(404)
    img_dir = RUNS_DIR / run_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if request.files:
        for _, f in request.files.items():
            fname = secure_filename(f.filename or "image.png")
            if Path(fname).suffix.lower() not in ALLOWED_IMG:
                return jsonify({"error": "Extension not allowed"}), 400
            f.save(img_dir / fname)
            saved.append(fname)
    else:
        d     = request.get_json(force=True, silent=True) or {}
        fname = secure_filename(d.get("filename", "image.png"))
        if Path(fname).suffix.lower() not in ALLOWED_IMG:
            return jsonify({"error": "Extension not allowed"}), 400
        (img_dir / fname).write_bytes(base64.b64decode(d.get("data", "")))
        saved.append(fname)

    meta["image_count"] = len(list(img_dir.iterdir()))
    save_meta(run_id, meta)
    return jsonify({"saved": saved}), 201

@app.post("/api/runs/<run_id>/data")
@login_required
def api_upload_data(run_id: str):
    meta = get_meta(run_id)
    if not meta:
        abort(404)
    data_dir = RUNS_DIR / run_id / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for _, f in request.files.items():
        fname = secure_filename(f.filename or "data.bin")
        if Path(fname).suffix.lower() not in ALLOWED_DATA:
            return jsonify({"error": "Extension not allowed"}), 400
        f.save(data_dir / fname)
        saved.append(fname)
    meta["data_file_count"] = len(list(data_dir.iterdir()))
    save_meta(run_id, meta)
    return jsonify({"saved": saved}), 201

@app.post("/api/runs/<run_id>/logs")
@login_required
def api_upload_log(run_id: str):
    """JSON { filename, content, append? }  OR  multipart file upload."""
    meta = get_meta(run_id)
    if not meta:
        abort(404)
    log_dir = RUNS_DIR / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if request.files:
        for _, f in request.files.items():
            f.save(log_dir / secure_filename(f.filename or "log.txt"))
    else:
        d       = request.get_json(force=True, silent=True) or {}
        fname   = secure_filename(d.get("filename", "run.log"))
        content = d.get("content", "")
        mode    = "a" if d.get("append", True) else "w"
        with open(log_dir / fname, mode) as lf:
            lf.write(content)
            if content and not content.endswith("\n"):
                lf.write("\n")
    return jsonify({"status": "ok"}), 200

@app.post("/api/runs/<run_id>/metrics")
@login_required
def api_upload_metrics(run_id: str):
    """
    Payload:
        method   str
        series   dict[str, list]    — full series replacement
        append   dict[str, float]   — append single point per metric
        scalars  dict[str, Any]     — scalar metadata
    """
    meta = get_meta(run_id)
    if not meta:
        abort(404)

    d      = request.get_json(force=True, silent=True) or {}
    method = d.get("method", "")

    if not (isinstance(method, str) and method):
        return jsonify({"error": "method is required"}), 400

    meta.setdefault("metrics", {}).setdefault(method, {})
    m = meta["metrics"][method]

    if isinstance(d.get("series"), dict):
        for k, v in d["series"].items():
            if isinstance(k, str) and isinstance(v, list):
                m[k] = v

    if isinstance(d.get("append"), dict):
        for k, v in d["append"].items():
            if isinstance(k, str):
                m.setdefault(k, [])
                if isinstance(m[k], list):
                    m[k].append(float(v))

    if isinstance(d.get("scalars"), dict):
        for k, v in d["scalars"].items():
            if isinstance(k, str):
                m[k] = v

    if "gt_label" in d:
        meta["gt_label"] = d["gt_label"]

    save_meta(run_id, meta)
    return jsonify({"status": "ok"}), 200

@app.get("/api/runs/<run_id>/stream")
@login_required
def api_stream_run(run_id: str):
    def event_stream():
        last_mtime = None
        while True:
            meta_path = RUNS_DIR / run_id / "meta.json"
            if not meta_path.exists():
                yield "event: error\ndata: {}\n\n"
                break
            mtime = meta_path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                yield f"data: {meta_path.read_text()}\n\n"
            time.sleep(2)
    return Response(event_stream(), mimetype="text/event-stream")

# ── 404 ────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render("base.html"), 404

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GradInv Experiment Server")
    p.add_argument("--host",  default="0.0.0.0", help="Bind address")
    p.add_argument("--port",  default=5001, type=int, help="Port")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = p.parse_args()
    print(f"""
  ◈ GradInv Server
  ──────────────────────────────────────
  URL:      http://localhost:{args.port}
  User:     {USERNAME}
  Data dir: {DATA_DIR.resolve()}
  ──────────────────────────────────────
  Override credentials via env vars:
    ML_USER=admin  ML_PASS=yourpassword
""")
    app.run(host=args.host, port=args.port, debug=args.debug)