"""Security response headers added to every HTTP response."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from trader_koo.middleware.security_headers import SECURITY_HEADERS, SecurityHeadersMiddleware


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/ping")
    def ping():
        return {"ok": True}

    @app.get("/framed")
    def framed():
        return PlainTextResponse("x", headers={"X-Frame-Options": "SAMEORIGIN"})

    return app


def test_security_headers_are_set_on_api_responses():
    with TestClient(_app()) as client:
        response = client.get("/api/ping")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in response.headers["permissions-policy"]
    assert "geolocation=()" in response.headers["permissions-policy"]


def test_security_headers_are_set_on_errors():
    with TestClient(_app()) as client:
        response = client.get("/missing")

    assert response.status_code == 404
    assert response.headers["x-frame-options"] == "DENY"


def test_route_specific_header_is_not_overwritten():
    with TestClient(_app()) as client:
        response = client.get("/framed")

    assert response.headers["x-frame-options"] == "SAMEORIGIN"
    assert response.headers.get_list("x-frame-options") == ["SAMEORIGIN"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_security_header_set_is_complete():
    assert {name.lower() for name, _ in SECURITY_HEADERS} == {
        "x-content-type-options",
        "x-frame-options",
        "content-security-policy",
        "referrer-policy",
        "permissions-policy",
    }
