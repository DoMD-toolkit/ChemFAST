"""Identity checks for the database and production model files used to make the goldens."""

from __future__ import annotations

import json
import warnings

import pytest

from tests._release_artifacts import (
    DATABASE_DOWNLOAD_HINT,
    database_manifest,
    model_manifest,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.reproducibility,
    pytest.mark.release_artifact,
    pytest.mark.requires_database,
    pytest.mark.requires_ml,
]


def test_release_artifacts_match_reference(
    project_root,
    golden_root,
    full_database_path,
):
    manifest_path = golden_root / "RELEASE_MANIFEST.json"

    if not manifest_path.is_file():
        raise AssertionError(
            f"Missing {manifest_path}; generate approved goldens first with "
            "tools/generate_reproducibility_goldens.py"
        )

    expected = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )["release_artifacts"]

    current_database = database_manifest(
        project_root,
        with_sha256=True,
    )

    if current_database != expected["database"]:
        warnings.warn(
            "The installed OPLS database does not match the database used "
            "to generate the approved ChemFAST reference files.\n\n"
            f"Expected database:\n{expected['database']}\n\n"
            f"Current database:\n{current_database}\n\n"
            "Database-dependent results may therefore differ from the "
            "published reproducibility reference.\n\n"
            f"{DATABASE_DOWNLOAD_HINT}",
            RuntimeWarning,
            stacklevel=2,
        )

    assert model_manifest(project_root) == expected["models"]