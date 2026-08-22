"""Tests for the framework-native admin authentication seam."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import router as production_admin_router
from trader_koo.middleware.auth import (
    AdminAuthConfig,
    AdminAuthenticator,
    admin_route_inventory,
    require_admin,
    route_uses_admin_dependency,
)


def _app(
    config: AdminAuthConfig,
    *,
    audit_recorder=None,
) -> FastAPI:
    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(
        config,
        audit_recorder=audit_recorder,
    )
    router = APIRouter(
        prefix="/api/admin",
        dependencies=[Depends(require_admin)],
    )

    @router.get("/ping")
    def ping(request: Request):
        return {"identity": request.state.admin_identity}

    app.include_router(router)

    @app.get("/api/public/ping")
    def public_ping():
        return {"ok": True}

    return app


def test_missing_and_invalid_keys_are_rejected_and_valid_key_is_accepted():
    app = _app(AdminAuthConfig(api_key="x" * 32, username="operator"))
    with TestClient(app) as client:
        assert client.get("/api/admin/ping").status_code == 401
        assert client.get(
            "/api/admin/ping", headers={"X-API-Key": "wrong"}
        ).status_code == 401
        response = client.get(
            "/api/admin/ping", headers={"X-API-Key": "x" * 32}
        )
        assert response.status_code == 200
        assert response.json()["identity"] == {
            "username": "operator",
            "mode": "api_key",
            "user_id": "operator",
        }


def test_missing_server_key_fails_closed_outside_explicit_development_mode():
    strict = _app(AdminAuthConfig(api_key="", strict_api_key=True))
    non_strict_production = _app(
        AdminAuthConfig(
            api_key="",
            strict_api_key=False,
            development_mode=False,
        )
    )
    assert TestClient(strict).get("/api/admin/ping").status_code == 503
    assert TestClient(non_strict_production).get("/api/admin/ping").status_code == 503


def test_explicit_development_mode_sets_a_visible_identity():
    app = _app(
        AdminAuthConfig(
            api_key="",
            strict_api_key=False,
            development_mode=True,
        )
    )
    response = TestClient(app).get("/api/admin/ping")
    assert response.status_code == 200
    assert response.json()["identity"] == {
        "username": "local-dev",
        "mode": "open-admin",
    }


def test_failed_attempts_are_throttled_and_valid_auth_clears_failure_state():
    app = _app(
        AdminAuthConfig(
            api_key="x" * 32,
            max_failures=3,
            failure_window_sec=300,
            block_sec=600,
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/admin/ping").status_code == 401
        assert client.get(
            "/api/admin/ping", headers={"X-API-Key": "x" * 32}
        ).status_code == 200
        assert client.get("/api/admin/ping").status_code == 401
        assert client.get("/api/admin/ping").status_code == 401
        blocked = client.get("/api/admin/ping")
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) > 0


def test_authentication_attempts_use_the_injected_audit_recorder():
    events: list[dict] = []
    app = _app(
        AdminAuthConfig(api_key="x" * 32),
        audit_recorder=lambda **payload: events.append(payload),
    )
    with TestClient(app) as client:
        client.get("/api/admin/ping")
        client.get("/api/admin/ping", headers={"X-API-Key": "x" * 32})
    assert [event["success"] for event in events] == [False, True]
    assert events[0]["reason"] == "invalid_api_key"


def test_runtime_inventory_reads_resolved_fastapi_dependencies():
    app = _app(AdminAuthConfig(api_key="x" * 32))

    @app.get("/api/admin/unsafe")
    def unsafe():
        return {"unsafe": True}

    inventory = admin_route_inventory(app)
    assert inventory == [
        {
            "method": "GET",
            "path": "/api/admin/ping",
            "has_auth": True,
            "key": "GET:/api/admin/ping",
        },
        {
            "method": "GET",
            "path": "/api/admin/unsafe",
            "has_auth": False,
            "key": "GET:/api/admin/unsafe",
        },
    ]


def test_every_production_admin_route_has_the_native_dependency():
    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(
        AdminAuthConfig(api_key="x" * 32)
    )
    app.include_router(production_admin_router)
    admin_routes = [
        route
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/admin/")
    ]
    assert admin_routes
    assert all(route_uses_admin_dependency(route) for route in admin_routes)
    assert all(row["has_auth"] for row in admin_route_inventory(app))
    schema = app.openapi()
    operations = [
        operation
        for path, path_item in schema["paths"].items()
        if path.startswith("/api/admin/")
        for method, operation in path_item.items()
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    ]
    assert operations
    assert all(operation["security"] == [{"APIKeyHeader": []}] for operation in operations)

    with TestClient(app) as client:
        assert client.get("/api/admin/routes").status_code == 401
        assert client.get(
            "/api/admin/routes", headers={"X-API-Key": "x" * 32}
        ).status_code == 200


def test_openapi_marks_the_admin_surface_with_the_api_key_scheme():
    app = _app(AdminAuthConfig(api_key="x" * 32))
    schema = app.openapi()
    operation = schema["paths"]["/api/admin/ping"]["get"]
    assert operation["security"] == [{"APIKeyHeader": []}]
    assert schema["paths"]["/api/public/ping"]["get"].get("security") is None
