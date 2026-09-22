"""Release input-artifact helpers used by reproducibility tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FULL_DB_MIN_BYTES = 100_000_000
DB_RELATIVE_PATH = Path("src/chemfast/ff/opls/opls_db/resources/opls.db")
MODEL_GLOBS = (
    "src/chemfast/ff/opls/opls_ml/models/*.pt",
    "src/chemfast/ff/opls/opls_ml/models/*.pkl",
    "src/chemfast/cg/cg_ff/models/*.pt",
)

DATABASE_DOWNLOAD_HINT = (
    "Download the full OPLS database from:\n"
    "  https://github.com/DoMD-toolkit/ChemFAST/releases/download/ChemFAST-v1.0.0/opls.db"
)

def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def require_full_database(project_root: Path) -> Path:
    path = Path(project_root) / DB_RELATIVE_PATH
    if not path.is_file():
        raise AssertionError(
            f"Full OPLS database is missing: {path}\n\n"
            f"{DATABASE_DOWNLOAD_HINT}"
        )
    if path.stat().st_size < FULL_DB_MIN_BYTES:
        raise AssertionError(
            f"OPLS database is only {path.stat().st_size:,} bytes; "
            f"expected the full release database.\n\n"
            f"{DATABASE_DOWNLOAD_HINT}"
        )
    return path


def model_files(project_root: Path) -> list[Path]:
    root = Path(project_root).resolve()
    files = []
    for pattern in MODEL_GLOBS:
        files.extend(root.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def model_manifest(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in model_files(root)
    ]
    if not rows:
        raise AssertionError("No packaged ChemFAST model files were found")
    combined = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"files": rows, "combined_sha256": combined}


def database_manifest(project_root: Path, *, with_sha256: bool = True) -> dict[str, Any]:
    import sqlite3

    path = require_full_database(project_root)
    tables = {
        "atom": "atom_type",
        "bond": "bond_type",
        "angle": "angle_type",
        "dihedral": "dihedral_type",
        "improper": "improper_type",
    }

    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5) as conn:
        counts = {}
        for name, table in tables.items():
            print(f"[manifest] counting {table} ...", flush=True)
            counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            print(f"[manifest] {table}: {counts[name]:,}", flush=True)

    out = {
        "path": DB_RELATIVE_PATH.as_posix(),
        "size_bytes": path.stat().st_size,
        "counts": counts,
    }
    if with_sha256:
        print("[manifest] hashing OPLS database ...", flush=True)
        out["sha256"] = sha256_file(path)
    return out
