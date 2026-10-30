"""Security response headers added to every HTTP response."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from trader_koo.middleware.security_headers import SECURITY_HEADERS, SecurityHeadersMiddleware

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    for name, value in SECURITY_HEADERS:
        assert response.headers[name] == value


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


def test_spa_script_policy_has_no_unsafe_inline():
    """Vite emits only external module scripts, so inline script is not needed."""
    html = (REPO_ROOT / "trader_koo/frontend-v2/index.html").read_text()
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert csp is not None
    directives = {
        tokens[0]: tokens[1:]
        for tokens in (item.split() for item in csp.group(1).split(";"))
        if tokens
    }
    script_src = set(directives["script-src"])
    assert "'unsafe-inline'" not in script_src
    assert {"'self'", "https://static.cloudflareinsights.com"} <= script_src

    scripts: list[dict[str, str | None]] = []

    class ScriptTags(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                scripts.append(dict(attrs))

    ScriptTags().feed(html)
    assert scripts
    for attrs in scripts:
        assert "src" in attrs or attrs.get("type") == "application/ld+json", attrs
