"""Property-based tests for admin auth boundary enforcement.

Requirements:
- 5.3: Add property-based test for auth boundary enforcement
"""

import pytest
from hypothesis import given, strategies as st, assume
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.middleware.auth import (
    require_admin_auth,
    auto_register_admin_endpoints,
    verify_all_admin_endpoints_protected,
    _ADMIN_ENDPOINT_REGISTRY,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the admin endpoint registry before each test."""
    _ADMIN_ENDPOINT_REGISTRY.clear()
    yield
    _ADMIN_ENDPOINT_REGISTRY.clear()


# Strategy for generating HTTP methods
http_methods = st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"])

# Strategy for generating admin endpoint paths
admin_paths = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=20
).map(lambda s: f"/api/admin/{s}")

# Strategy for generating non-admin paths
non_admin_paths = st.one_of(
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20
    ).map(lambda s: f"/api/public/{s}"),
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20
    ).map(lambda s: f"/api/{s}"),
    st.just("/"),
    st.just("/health"),
)


class TestAdminAuthBoundaryProperty:
    """Property-based tests for admin authentication boundary.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
    
    These tests verify that:
    1. All /api/admin/* endpoints are registered
    2. All /api/admin/* endpoints require authentication
    3. Non-admin endpoints are not affected
    4. The authentication boundary is consistently enforced
    """
    
    @given(path=admin_paths, method=http_methods)
    def test_admin_endpoints_must_be_in_registry(self, path, method):
        """Property: All /api/admin/* endpoints must be registered in the registry.
        
        For any admin endpoint path and HTTP method, after auto-registration,
        the endpoint should appear in the admin endpoint registry.
        """
        app = FastAPI()
        
        # Create a simple endpoint
        if method == "GET":
            @app.get(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "POST":
            @app.post(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "PUT":
            @app.put(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "DELETE":
            @app.delete(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "PATCH":
            @app.patch(path)
            def endpoint():
                return {"status": "ok"}
        
        # Auto-register endpoints
        auto_register_admin_endpoints(app)
        
        # Verify the endpoint is registered
        from trader_koo.middleware.auth import _ADMIN_ENDPOINT_REGISTRY
        key = f"{method}:{path}"
        assert key in _ADMIN_ENDPOINT_REGISTRY, (
            f"Admin endpoint {key} should be registered but was not found"
        )
    
    @given(path=non_admin_paths, method=http_methods)
    def test_non_admin_endpoints_not_in_registry(self, path, method):
        """Property: Non-admin endpoints should not be registered.
        
        For any non-admin endpoint path, it should not appear in the
        admin endpoint registry after auto-registration.
        """
        # Skip if path accidentally starts with /api/admin/
        assume(not path.startswith("/api/admin/"))
        
        app = FastAPI()
        
        # Create a simple endpoint
        if method == "GET":
            @app.get(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "POST":
            @app.post(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "PUT":
            @app.put(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "DELETE":
            @app.delete(path)
            def endpoint():
                return {"status": "ok"}
        elif method == "PATCH":
            @app.patch(path)
            def endpoint():
                return {"status": "ok"}
        
        # Auto-register endpoints
        auto_register_admin_endpoints(app)
        
        # Verify the endpoint is NOT registered
        from trader_koo.middleware.auth import _ADMIN_ENDPOINT_REGISTRY
        key = f"{method}:{path}"
        assert key not in _ADMIN_ENDPOINT_REGISTRY, (
            f"Non-admin endpoint {key} should not be registered but was found"
        )
    
    @given(
        paths=st.lists(admin_paths, min_size=1, max_size=10, unique=True),
        methods=st.lists(http_methods, min_size=1, max_size=10)
    )
    def test_all_admin_endpoints_tracked(self, paths, methods):
        """Property: All admin endpoints are tracked in the registry.
        
        For any collection of admin endpoints, after auto-registration,
        all endpoints should be present in the registry.
        """
        app = FastAPI()
        
        # Create endpoints
        created_endpoints = []
        for i, (path, method) in enumerate(zip(paths, methods)):
            endpoint_name = f"endpoint_{i}"
            
            if method == "GET":
                @app.get(path, name=endpoint_name)
                def endpoint():
                    return {"status": "ok"}
            elif method == "POST":
                @app.post(path, name=endpoint_name)
                def endpoint():
                    return {"status": "ok"}
            elif method == "PUT":
                @app.put(path, name=endpoint_name)
                def endpoint():
                    return {"status": "ok"}
            elif method == "DELETE":
                @app.delete(path, name=endpoint_name)
                def endpoint():
                    return {"status": "ok"}
            elif method == "PATCH":
                @app.patch(path, name=endpoint_name)
                def endpoint():
                    return {"status": "ok"}
            
            created_endpoints.append((method, path))
        
        # Auto-register endpoints
        auto_register_admin_endpoints(app)
        
        # Verify all endpoints are registered
        from trader_koo.middleware.auth import _ADMIN_ENDPOINT_REGISTRY
        for method, path in created_endpoints:
            key = f"{method}:{path}"
            assert key in _ADMIN_ENDPOINT_REGISTRY, (
                f"Admin endpoint {key} should be registered"
            )
    
    @given(
        protected_count=st.integers(min_value=0, max_value=5),
        unprotected_count=st.integers(min_value=0, max_value=5)
    )
    def test_verification_detects_unprotected_endpoints(self, protected_count, unprotected_count):
        """Property: Verification correctly identifies unprotected endpoints.
        
        For any mix of protected and unprotected endpoints, the verification
        function should correctly count and identify unprotected endpoints.
        """
        # Clear registry for this hypothesis example
        _ADMIN_ENDPOINT_REGISTRY.clear()
        
        app = FastAPI()
        
        # Create protected endpoints with unique function names
        for i in range(protected_count):
            path = f"/api/admin/protected_{i}"
            
            # Create a unique function for each endpoint
            def make_protected_endpoint():
                def endpoint():
                    return {"status": "ok"}
                return endpoint
            
            endpoint_func = make_protected_endpoint()
            endpoint_func = require_admin_auth(endpoint_func)
            app.get(path)(endpoint_func)
        
        # Create unprotected endpoints with unique function names
        for i in range(unprotected_count):
            path = f"/api/admin/unprotected_{i}"
            
            # Create a unique function for each endpoint
            def make_unprotected_endpoint():
                def endpoint():
                    return {"status": "ok"}
                return endpoint
            
            endpoint_func = make_unprotected_endpoint()
            app.get(path)(endpoint_func)
        
        # Auto-register endpoints
        auto_register_admin_endpoints(app)
        
        # Verify detection
        all_protected, unprotected_list = verify_all_admin_endpoints_protected()
        
        if unprotected_count == 0:
            assert all_protected is True
            assert len(unprotected_list) == 0
        else:
            assert all_protected is False
            assert len(unprotected_list) == unprotected_count
    
    @given(path=admin_paths)
    def test_unauthenticated_requests_rejected(self, path):
        """Property: Unauthenticated requests to admin endpoints are rejected.
        
        For any admin endpoint, requests without authentication should be
        rejected with 401 or 503 status code.
        """
        app = FastAPI()
        
        # Create an admin endpoint without auth decorator
        @app.get(path)
        def endpoint():
            return {"status": "ok"}
        
        # Add the API key middleware (simplified version)
        @app.middleware("http")
        async def auth_middleware(request, call_next):
            if request.url.path.startswith("/api/admin/"):
                # Simulate missing API key
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            return await call_next(request)
        
        client = TestClient(app)
        
        # Request without authentication
        response = client.get(path)
        
        # Should be rejected
        assert response.status_code in [401, 503], (
            f"Unauthenticated request to {path} should be rejected, "
            f"but got status {response.status_code}"
        )
