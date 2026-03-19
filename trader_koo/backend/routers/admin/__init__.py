"""Admin router package — combines all admin sub-routers into a single ``router``.

Consumers import this exactly as before::

    from trader_koo.backend.routers.admin import router as admin_router
"""
from __future__ import annotations

from fastapi import APIRouter

from trader_koo.backend.routers.admin.data import router as data_router
from trader_koo.backend.routers.admin.email_admin import (
    router as email_router,
)
from trader_koo.backend.routers.admin.ml import router as ml_router
from trader_koo.backend.routers.admin.pipeline import (
    router as pipeline_router,
)
from trader_koo.backend.routers.admin.system import router as system_router

router = APIRouter()

router.include_router(pipeline_router)
router.include_router(ml_router)
router.include_router(data_router)
router.include_router(email_router)
router.include_router(system_router)

__all__ = ["router"]
