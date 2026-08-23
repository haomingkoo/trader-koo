from trader_koo.backend.frontend_routes import SPA_ROUTE_PATHS


def test_direct_navigation_routes_include_all_research_surfaces() -> None:
    assert {"experiments", "agent-observability"}.issubset(SPA_ROUTE_PATHS)
