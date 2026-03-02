from fastapi import FastAPI

from app.api.routes import router as api_router
from app.db.base import Base
from app.db.session import engine


def create_app() -> FastAPI:
    app = FastAPI(
        title="RiskScore99 API",
        version="0.1.0",
        description=(
            "RiskScore99 – IEEE-CIS based fraud risk scoring MVP. "
            "Provides calibrated 0–99 risk scores, agentic routing decisions, "
            "and audit logging over a SQLite backend."
        ),
    )

    @app.on_event("startup")
    def on_startup() -> None:
        Base.metadata.create_all(bind=engine)

    @app.get("/health", tags=["system"])
    async def health_check() -> dict:
        return {"status": "ok"}

    app.include_router(api_router)
    return app


app = create_app()

