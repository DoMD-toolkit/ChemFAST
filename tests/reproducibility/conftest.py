import os
import warnings
from pathlib import Path

import pytest

from tests._release_artifacts import DB_RELATIVE_PATH, FULL_DB_MIN_BYTES, DATABASE_DOWNLOAD_HINT

_FULL_DB_TESTS = {
    "test_db_forcefield_repro.py",
    "test_minimal_pi_repro.py",
    "test_release_artifacts_repro.py",
    "test_spe_network_repro.py",
}


@pytest.fixture(autouse=True)
def skip_reproducibility_tests_without_full_opls_db(request):
    if request.node.path.name not in _FULL_DB_TESTS:
        return

    root = Path(__file__).resolve().parents[2]
    db = root / DB_RELATIVE_PATH
    size = db.stat().st_size if db.is_file() else 0

    if size >= FULL_DB_MIN_BYTES:
        return

    message = (
        f"Full OPLS database is unavailable ({size:,} bytes). "
        "Skipping reproducibility tests that require the full release database. "
        + DATABASE_DOWNLOAD_HINT
    )

    if os.environ.get("CHEMFAST_REQUIRE_FULL_DB") == "1":
        pytest.fail(message)

    warnings.warn(message, RuntimeWarning, stacklevel=2)
    pytest.skip(message)