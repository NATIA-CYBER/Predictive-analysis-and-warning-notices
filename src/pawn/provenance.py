from __future__ import annotations
import hashlib, json, platform, subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable, Dict, Any

def _git(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git"] + cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _pip_freeze(limit: int = 2000) -> str:
    try:
        out = subprocess.check_output(["python", "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
        return out[:limit]
    except Exception:
        return ""

def write_provenance(out_dir: Path, data_paths: Iterable[Path], model_params: Dict[str, Any] = None, extra: Dict[str, Any] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git": {
            "sha": _git(["rev-parse", "--short", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"])),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pip_freeze_head": _pip_freeze(),
        },
        "data_checksums": {str(p): _sha256(p) for p in data_paths if Path(p).exists()},
        "model_params": model_params or {},
        "extra": extra or {},
    }
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(prov, indent=2))
    return path
