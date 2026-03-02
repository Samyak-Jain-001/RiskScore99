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
    model_type: Literal["logreg", "hgb", "lightgbm"] = "logreg"
    threshold_t1: float = 0.20
    threshold_t2: float = 0.50
    threshold_t3: float = 0.80

    friction_cost: float = 1.0
    fraud_loss_multiplier: float = 2.0

    # Optional: fixed model version to load; if None, load latest from registry
    fixed_model_version: Optional[str] = None


settings = Settings()

