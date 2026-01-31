"""Global configuration powered by pydantic-settings (12-factor friendly)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables when available."""

    # Visualization defaults
    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"
    COLOR_PRIMARY: str = "#ff4b4b"
    ANIMATION_DURATION_MS: int = 150

    # Calculation defaults
    DEFAULT_SEED: int = 42
    RICCI_CUTOFF: float = 8.0
    RICCI_MAX_SUPPORT: int = 60
    APPROX_EFFICIENCY_K: int = 32

    # Phase transition heuristics
    CRITICAL_JUMP_THRESHOLD: float = 0.35  # Fraction of range to consider "abrupt"

    # UI defaults
    DEFAULT_DAMPING: float = 0.98
    DEFAULT_INJECTION: float = 1.0
    DEFAULT_LEAK: float = 0.005

    # Позволяет переопределять через env vars: KODIK_PLOT_HEIGHT=1000
    class Config:
        env_prefix = "KODIK_"


settings = Settings()
