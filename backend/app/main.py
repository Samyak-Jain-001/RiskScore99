import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.db.base import Base
from app.db.session import engine, SessionLocal
from app.utils.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="RiskScore99 API",
        version="0.2.0",
        description=(
            "RiskScore99 – IEEE-CIS based fraud risk scoring MVP. "
            "Provides calibrated 0–99 risk scores, agentic routing decisions, "
            "and audit logging over a SQLite backend."
        ),
    )

    # ── CORS (needed for frontend) ──────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def on_startup() -> None:
        Base.metadata.create_all(bind=engine)
        logger.info("RiskScore99 API started — tables initialized")

    # ── Enhanced health check ───────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check() -> dict:
        """
        Deep health check: verifies DB connectivity and model availability.
        """
        checks = {"api": "ok"}

        # DB check
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1"))  # type: ignore[arg-type]
            db.close()
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc}"

        # Model check
        try:
            from app.db.models import ModelRegistry
            db = SessionLocal()
            latest = (
                db.query(ModelRegistry)
                .filter(ModelRegistry.is_active == True)  # noqa: E712
                .order_by(ModelRegistry.created_at.desc())
                .first()
            )
            db.close()
            if latest:
                checks["model"] = f"ok (version={latest.model_version})"
            else:
                checks["model"] = "warning: no active model"
        except Exception as exc:
            checks["model"] = f"error: {exc}"

        overall = "ok" if all("ok" in str(v) for v in checks.values()) else "degraded"

        return {"status": overall, "checks": checks}

    app.include_router(api_router)
    return app


app = create_app()
