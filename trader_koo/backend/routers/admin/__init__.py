"""Admin router package — combines all admin sub-routers into a single ``router``.

Consumers import this exactly as before::

    from trader_koo.backend.routers.admin import router as admin_router
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from trader_koo.backend.routers.admin.alert_quality import (
    router as alert_quality_router,
)
from trader_koo.backend.routers.admin.agents import router as agents_router
from trader_koo.backend.routers.admin.backups import (
    router as backups_router,
)
from trader_koo.backend.routers.admin.crypto import (
    router as crypto_router,
)
from trader_koo.backend.routers.admin.data import router as data_router
from trader_koo.backend.routers.admin.email_admin import (
    router as email_router,
)
from trader_koo.backend.routers.admin.market_monitor import (
    router as market_monitor_router,
)
from trader_koo.backend.routers.admin.ml import router as ml_router
from trader_koo.backend.routers.admin.pipeline import (
    router as pipeline_router,
)
from trader_koo.backend.routers.admin.paper_campaigns import (
    router as paper_campaigns_router,
)
from trader_koo.backend.routers.admin.system import router as system_router
from trader_koo.backend.routers.admin.telegram import (
    router as telegram_router,
)
from trader_koo.backend.routers.data_sync import router as data_sync_router
from trader_koo.middleware.auth import require_admin
from trader_koo.ratelimit.api import router as rate_limit_router

router = APIRouter(dependencies=[Depends(require_admin)])

router.include_router(alert_quality_router)
router.include_router(agents_router)
router.include_router(pipeline_router)
router.include_router(paper_campaigns_router)
router.include_router(ml_router)
router.include_router(data_router)
router.include_router(email_router)
router.include_router(system_router)
router.include_router(telegram_router)
router.include_router(backups_router)
router.include_router(market_monitor_router)
router.include_router(crypto_router)
router.include_router(data_sync_router)
router.include_router(rate_limit_router)

__all__ = ["router"]
