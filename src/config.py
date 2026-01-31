"""Global configuration and constants."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration with env overrides (prefix: KODIK_)."""

    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"
    COLOR_PRIMARY: str = "#ff4b4b"
    ANIMATION_DURATION_MS: int = 150

    DEFAULT_SEED: int = 42
    RICCI_CUTOFF: float = 8.0
    RICCI_MAX_SUPPORT: int = 60
    APPROX_EFFICIENCY_K: int = 32

    CRITICAL_JUMP_THRESHOLD: float = 0.35

    DEFAULT_DAMPING: float = 0.98
    DEFAULT_INJECTION: float = 1.0
    DEFAULT_LEAK: float = 0.005

    model_config = SettingsConfigDict(env_prefix="KODIK_")


settings = Settings()
