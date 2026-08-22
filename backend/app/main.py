from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.state import AppContext


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.ctx = AppContext.bootstrap()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="DeFi Yield Research API",
        description=(
            "Academic research backend. The lending environment is an educational "
            "Aave-inspired simulation, not the Aave protocol. ML artifacts are trained "
            "on synthetic local data unless a historical pipeline is configured."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url, "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
