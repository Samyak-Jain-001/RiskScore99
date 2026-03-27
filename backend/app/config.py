from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_ROOT = DATA_ROOT / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RISKSCORE99_", extra="ignore", protected_namespaces=("settings_",)
    )

    # Paths
    data_root: Path = DATA_ROOT
    artifacts_root: Path = ARTIFACTS_ROOT
    sqlite_path: Path = PROJECT_ROOT / "backend" / "app" / "riskscore99.db"

    # Model + thresholds
    model_type: Literal["logreg", "hgb", "lightgbm"] = "hgb"
    threshold_t1: float = 0.20
    threshold_t2: float = 0.50
    threshold_t3: float = 0.80

    friction_cost: float = 1.0
    fraud_loss_multiplier: float = 2.0

    # Optional: fixed model version to load; if None, load latest from registry
    fixed_model_version: Optional[str] = None

    # ── NEW: Agentic behavior settings ──────────────────────────────
    # Ambiguous score band — triggers re-evaluation loop
    ambiguous_score_low: int = 35
    ambiguous_score_high: int = 65

    # Enable the agentic re-evaluation loop (set False for deterministic-only mode)
    enable_adaptive_loop: bool = True

    # Enhanced explanation for high-risk decisions (BLOCK / REVIEW)
    enable_enhanced_explanations: bool = True

    # ── NEW: Drift / feedback thresholds ────────────────────────────
    drift_alert_threshold: int = 500       # outcomes before first drift check
    drift_retrain_threshold: int = 2000    # outcomes before retrain recommendation

    # ── NEW: Cost model – real avg transaction amount (updated by training) ─
    avg_transaction_amt: float = 150.0     # default; overwritten during training


settings = Settings()
