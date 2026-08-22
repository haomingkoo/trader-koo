from __future__ import annotations

from trader_koo.backend.routers.admin import ml as admin_ml
from trader_koo.ml import scorer

from tests.test_ml_scorer import _Model, _meta, _verified_conn


def test_admin_model_status_exposes_current_basis_mismatch(monkeypatch):
    monkeypatch.setattr(
        scorer,
        "load_model",
        lambda: (_Model(), _meta(adjustment_version="old-v0")),
    )
    monkeypatch.setattr(admin_ml, "get_conn", _verified_conn)

    result = admin_ml.ml_model_status(None)

    assert result["loaded"] is True
    assert result["basis_compatible"] is False
    assert "adjustment_version" in result["basis_error"]
    assert result["model_price_contract"]["adjustment_version"] == "old-v0"
    assert result["current_price_contract"]["version"] == "test-v1"


def test_admin_shap_fails_before_feature_extraction_on_basis_mismatch(monkeypatch):
    from trader_koo.ml import features

    monkeypatch.setattr(
        scorer,
        "load_model",
        lambda: (_Model(), _meta(adjustment_version="old-v0")),
    )
    monkeypatch.setattr(admin_ml, "get_conn", _verified_conn)
    monkeypatch.setattr(
        features,
        "extract_features_for_universe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("features must not run on a basis mismatch")
        ),
    )

    result = admin_ml.ml_shap_analysis(None)

    assert result["ok"] is False
    assert "adjustment_version" in result["error"]
