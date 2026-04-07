"""Security regression tests for admin data sync uploads."""

import pytest

from trader_koo.backend.utils import resolve_child_filename


def test_model_upload_path_stays_inside_model_dir(tmp_path):
    safe_name, dest_path = resolve_child_filename(tmp_path, "baseline_model.joblib")

    assert safe_name == "baseline_model.joblib"
    assert dest_path == (tmp_path / "baseline_model.joblib").resolve()


@pytest.mark.parametrize(
    "filename",
    [
        "../../escape.joblib",
        "/tmp/escape.joblib",
        "..\\escape.joblib",
        "nested/model.joblib",
    ],
)
def test_model_upload_path_rejects_directory_traversal(tmp_path, filename):
    with pytest.raises(ValueError):
        resolve_child_filename(tmp_path, filename)
