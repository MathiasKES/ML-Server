"""
Gradient Inversion Experiment Server
Single-user, authenticated. Receives and displays DLG/iDLG experiment data.

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

Credentials (defaults admin / changeme):
    GRAD_USER=admin  GRAD_PASS=yourpassword  python server.py
"""

import os
import json
import uuid
import hashlib
import secrets
import base64
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask, request, jsonify, session,
    redirect, url_for, send_from_directory, abort,
)
from werkzeug.utils import secure_filename
from jinja2 import FileSystemLoader, Environment

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("experiment_data")
RUNS_DIR    = DATA_DIR / "runs"
SECRET_FILE = DATA_DIR / ".secret_key"
HTML_DIR    = Path(__file__).parent / "html"

USERNAME = os.environ.get("GRAD_USER", "admin")
PASSWORD = os.environ.get("GRAD_PASS", "password")

ALLOWED_IMG  = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_DATA = {".pkl", ".npy", ".npz", ".json", ".csv", ".txt", ".log"}

# ── App & Jinja2 setup ─────────────────────────────────────────────────────────

app = Flask(__name__)

for d in [DATA_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Secret key (persisted so sessions survive restarts)
if SECRET_FILE.exists():
    app.secret_key = SECRET_FILE.read_bytes()
else:
    key = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(key)
    app.secret_key = key

# Jinja2 environment pointing at the html/ folder
_jinja = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    autoescape=True,
)

def render(template_name: str, **ctx) -> str:
    """Render a template from the html/ folder with common context injected."""
    ctx.setdefault("username", USERNAME)
    ctx.setdefault("active_nav", "")
    ctx.setdefault("breadcrumb", "")
    return _jinja.get_template(f"html/{template_name}").render(**ctx)

# In-memory bearer tokens (cleared on restart)
_tokens: dict[str, bool] = {}

# ── Auth helpers ───────────────────────────────────────────────────────────────

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

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
    """Allow API access via Authorization: Bearer <token> header."""
    if request.path.startswith("/api/") and not session.get("authenticated"):
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and _tokens.get(auth[7:]):
            session["authenticated"] = True

# ── Metadata helpers ───────────────────────────────────────────────────────────

def get_meta(run_id: str) -> dict | None:
    f = RUNS_DIR / run_id / "meta.json"
    return json.loads(f.read_text()) if f.exists() else None

def save_meta(run_id: str, meta: dict):
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

def fsize(p: Path) -> str:
    s = p.stat().st_size
    for u in ["B", "KB", "MB", "GB"]:
        if s < 1024:
            return f"{s:.1f} {u}"
        s /= 1024
    return f"{s:.1f} TB"

# ── Auth routes ────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET"])
def login_page():
    if session.get("authenticated"):
        return redirect(url_for("dashboard"))
    return render("login.html", error=bool(request.args.get("error")))

@app.route("/login", methods=["POST"])
def do_login():
    u = request.form.get("username", "")
    p = request.form.get("password", "")
    if u == USERNAME and hash_pw(p) == hash_pw(PASSWORD):
        session["authenticated"] = True
        session.permanent = True
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page") + "?error=1")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def dashboard():
    runs   = list_runs()
    dsets  = sorted({r.get("dataset", "?") for r in runs})
    meths  = sorted({m for r in runs for m in r.get("methods", [])})

    # Attach formatted timestamp for template
    display_runs = []
    for r in runs[:8]:
        r = dict(r)
        r["created_fmt"] = fmt_ts(r.get("created_at", ""))
        display_runs.append(r)

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

# ── Experiments list ───────────────────────────────────────────────────────────

@app.route("/jobs")
@login_required
def jobs_list():
    runs = list_runs()
    display = []
    for r in runs:
        r = dict(r)
        r["created_fmt"] = fmt_ts(r.get("created_at", ""))
        dl = r.get("final_loss_DLG")
        il = r.get("final_loss_iDLG")
        parts = []
        if dl is not None: parts.append(f"DLG:{dl:.4f}")
        if il is not None: parts.append(f"iDLG:{il:.4f}")
        r["loss_str"] = "  ".join(parts)
        display.append(r)
    return render("jobs_list.html", active_nav="jobs", breadcrumb="/ experiments", runs=display)

# ── Experiment detail ──────────────────────────────────────────────────────────

@app.route("/jobs/<run_id>")
@login_required
def job_detail(run_id: str):
    meta = get_meta(run_id)
    if not meta:
        abort(404)
    rd = RUNS_DIR / run_id

    images     = sorted((rd / "images").glob("*")) if (rd / "images").exists() else []
    data_files = sorted((rd / "data").glob("*"))   if (rd / "data").exists()   else []
    log_files  = sorted((rd / "logs").glob("*"))   if (rd / "logs").exists()   else []

    # Attach human-readable size to each data file
    data_display = [type("F", (), {"name": f.name, "size_str": fsize(f)})() for f in data_files]

    # Read up to 3 log files into a single string
    log_content = ""
    for lf in log_files[:3]:
        try:
            txt = lf.read_text(errors="replace")[-8000:]
        except Exception:
            txt = "(unreadable)"
        log_content += f"=== {lf.name} ===\n{txt}\n\n"

    M = meta.get("metrics", {})
    dlg_loss  = M.get("DLG",  {}).get("loss", [])
    idlg_loss = M.get("iDLG", {}).get("loss", [])
    dlg_mse   = M.get("DLG",  {}).get("mse",  [])
    idlg_mse  = M.get("iDLG", {}).get("mse",  [])
    has_metrics = any([dlg_loss, idlg_loss, dlg_mse, idlg_mse])

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
        dlg_loss=dlg_loss,
        idlg_loss=idlg_loss,
        dlg_mse=dlg_mse,
        idlg_mse=idlg_mse,
    )

# ── File serving ───────────────────────────────────────────────────────────────

@app.route("/files/<run_id>/<subdir>/<filename>")
@login_required
def serve_file(run_id: str, subdir: str, filename: str):
    if subdir not in {"images", "data", "logs"}:
        abort(404)
    return send_from_directory(RUNS_DIR / run_id / subdir, secure_filename(filename))

# ── API: token ─────────────────────────────────────────────────────────────────

@app.route("/api/token", methods=["POST"])
def api_token():
    d = request.get_json(force=True, silent=True) or {}
    if d.get("username") == USERNAME and hash_pw(d.get("password", "")) == hash_pw(PASSWORD):
        tok = secrets.token_hex(32)
        _tokens[tok] = True
        return jsonify({"token": tok})
    return jsonify({"error": "Invalid credentials"}), 401

# ── API: storage info ──────────────────────────────────────────────────────────

@app.route("/api/storage")
@login_required
def api_storage():
    total = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    n = float(total)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return jsonify({"size_bytes": total, "size_human": f"{n:.1f} {u}"})
        n /= 1024
    return jsonify({"size_bytes": total, "size_human": f"{n:.1f} PB"})

# ── API: list / get runs ───────────────────────────────────────────────────────

@app.route("/api/runs", methods=["GET"])
@login_required
def api_list_runs():
    return jsonify(list_runs())

@app.route("/api/runs/<run_id>", methods=["GET"])
@login_required
def api_get_run(run_id: str):
    m = get_meta(run_id)
    if not m:
        abort(404)
    return jsonify(m)

# ── API: create run ────────────────────────────────────────────────────────────

@app.route("/api/runs", methods=["POST"])
@login_required
def api_create_run():
    """
    Create a new experiment run.
    Body (JSON):
        name        str
        dataset     str   e.g. "cifar100"
        methods     list  e.g. ["DLG", "iDLG"]
        num_dummy   int
        iterations  int
        lr          float
        num_classes int
    Returns: { run_id, name }
    """
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

# ── API: upload image ──────────────────────────────────────────────────────────

@app.route("/api/runs/<run_id>/images", methods=["POST"])
@login_required
def api_upload_image(run_id: str):
    """
    Multipart form (field: file)  OR  JSON { filename, data: <base64> }
    """
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
        d = request.get_json(force=True, silent=True) or {}
        fname = secure_filename(d.get("filename", "image.png"))
        if Path(fname).suffix.lower() not in ALLOWED_IMG:
            return jsonify({"error": "Extension not allowed"}), 400
        (img_dir / fname).write_bytes(base64.b64decode(d.get("data", "")))
        saved.append(fname)

    meta["image_count"] = len(list(img_dir.iterdir()))
    save_meta(run_id, meta)
    return jsonify({"saved": saved}), 201

# ── API: upload data file ──────────────────────────────────────────────────────

@app.route("/api/runs/<run_id>/data", methods=["POST"])
@login_required
def api_upload_data(run_id: str):
    """Multipart form upload (field: file)."""
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

# ── API: upload log ────────────────────────────────────────────────────────────

@app.route("/api/runs/<run_id>/logs", methods=["POST"])
@login_required
def api_upload_log(run_id: str):
    """
    JSON  { filename, content, append? }   OR   multipart file upload.
    append defaults to true.
    """
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
            if not content.endswith("\n"):
                lf.write("\n")
    return jsonify({"status": "ok"}), 200

# ── API: upload metrics ────────────────────────────────────────────────────────

@app.route("/api/runs/<run_id>/metrics", methods=["POST"])
@login_required
def api_upload_metrics(run_id: str):
    """
    Body fields:
        method           str    "DLG" | "iDLG"
        loss             list   full loss array
        mse              list   full mse array
        append_loss      float  single value appended to loss list
        append_mse       float  single value appended to mse list
        final_loss_DLG   float
        final_loss_iDLG  float
        final_mse_DLG    float
        final_mse_iDLG   float
        gt_label         int
        label_DLG        int
        label_iDLG       int
    """
    meta = get_meta(run_id)
    if not meta:
        abort(404)
    d = request.get_json(force=True, silent=True) or {}
    method = d.get("method", "")

    if method:
        if method not in meta["metrics"]:
            meta["metrics"][method] = {"loss": [], "mse": []}
        m = meta["metrics"][method]
        if "loss"        in d: m["loss"] = d["loss"]
        if "mse"         in d: m["mse"]  = d["mse"]
        if "append_loss" in d: m["loss"].append(float(d["append_loss"]))
        if "append_mse"  in d: m["mse"].append(float(d["append_mse"]))

    for key in ["gt_label", "label_DLG", "label_iDLG",
                "final_loss_DLG", "final_loss_iDLG",
                "final_mse_DLG",  "final_mse_iDLG"]:
        if key in d:
            meta[key] = d[key]

    save_meta(run_id, meta)
    return jsonify({"status": "ok"}), 200

# ── API Docs ───────────────────────────────────────────────────────────────────

@app.route("/api-docs")
@login_required
def api_docs():
    return render("api_docs.html", active_nav="api", breadcrumb="/ api-docs")

# ── 404 ────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render("base.html",
                  content_block="<div style='text-align:center;padding:80px 20px;color:var(--text3)'>"
                                "<div style='font-size:48px;margin-bottom:16px'>&#x2298;</div>"
                                "<p>Page not found.</p></div>"), 404

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
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
  HTML dir: {HTML_DIR.resolve()}
  ──────────────────────────────────────
  Override credentials via env vars:
    GRAD_USER=admin  GRAD_PASS=yourpassword
""")
    app.run(host=args.host, port=args.port, debug=args.debug)
