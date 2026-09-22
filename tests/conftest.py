from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
# Keep this path short on Windows: Numba appends long module/hash filenames.
NUMBA_CACHE_DIR = Path(tempfile.gettempdir()) / "chemfast_numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
DATA_DIR = TESTS_DIR / "data"
GOLDEN_ROOT = TESTS_DIR / "golden" / "reproducibility"
TOLERANCES_PATH = TESTS_DIR / "golden" / "tolerances.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests using packaged resources or integrated workflows")
    config.addinivalue_line("markers", "reproducibility: file-level golden reproducibility tests")
    config.addinivalue_line("markers", "requires_database: requires the full release opls.db")
    config.addinivalue_line("markers", "requires_ml: requires packaged production ML checkpoints")
    config.addinivalue_line("markers", "release_artifact: exact SHA256 validation of release artifacts")


@pytest.fixture
def project_root(): return PROJECT_ROOT

@pytest.fixture
def tests_dir(): return TESTS_DIR

@pytest.fixture
def golden_root(): return GOLDEN_ROOT

@pytest.fixture
def tolerances():
    return json.loads(TOLERANCES_PATH.read_text(encoding="utf-8"))

@pytest.fixture
def minimal_pi_dir(): return DATA_DIR / "minimal_pi"

@pytest.fixture
def au_peo_dir(): return DATA_DIR / "golden_cases" / "au_peo"

@pytest.fixture
def spe_network_dir(): return DATA_DIR / "golden_cases" / "spe_network"

@pytest.fixture
def cg_params_dir(): return DATA_DIR / "golden_cases" / "cg_params"

@pytest.fixture
def db_cases_path(): return DATA_DIR / "db_cases.json"

@pytest.fixture
def ml_cases_path(): return DATA_DIR / "ml_cases.json"

@pytest.fixture
def minimal_pi_config(minimal_pi_dir):
    return json.loads((minimal_pi_dir / "config.json").read_text(encoding="utf-8"))

@pytest.fixture
def full_database_path(project_root):
    from tests._release_artifacts import require_full_database
    return require_full_database(project_root)


@pytest.fixture
def run_repro_case(project_root):
    def run(case: str, output: Path) -> None:
        subprocess.run(
            [sys.executable, "-m", "tests._repro_case", case, str(output),
             "--repo-root", str(project_root)],
            cwd=project_root,
            check=True,
        )
    return run
