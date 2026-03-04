from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def copy_into_spool(src: str, spool_artifacts_dir: Path) -> Path:
    """
    Copy a file into the spool artifacts directory so it can be replayed later.
    Returns the new path.
    """
    ensure_dir(spool_artifacts_dir)
    src_p = Path(src)
    dst = spool_artifacts_dir / src_p.name
    if src_p.resolve() != dst.resolve():
        shutil.copy2(src_p, dst)
    return dst


def figure_to_png_bytes(figure: Any) -> bytes:
    """
    Convert a matplotlib Figure to PNG bytes without requiring a temp file.
    """
    buf = io.BytesIO()
    # matplotlib Figure has savefig
    figure.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    return buf.getvalue()