from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]      # .../backend
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """
    All settings can be overridden with environment variables (case-insensitive)
    or in backend/.env, e.g.  DATA_DIR=/data  MAX_UPLOAD_MB=200
    """
    model_config = SettingsConfigDict(env_file=str(BACKEND_DIR / ".env"), extra="ignore")

    anthropic_api_key: str = ""
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Where uploads/, outputs/ and references/ live. Default keeps the old layout (inside backend/).
    data_dir: Path = BACKEND_DIR

    # Pro reference video for the serve. First existing path wins: REF_SERVE_PATH,
    # backend/assets/reference/serve.mov (recommended), then the original Windows dev path.
    ref_serve_path: str = ""
    ref_serve_label: str = "Elena Rybakina"

    # Built frontend (npm run build). If the folder exists, FastAPI serves it at "/".
    frontend_dist: Path = PROJECT_DIR / "frontend" / "dist"

    # Limits & housekeeping
    max_upload_mb: int = 100            # larger uploads are rejected with 413
    max_concurrent_jobs: int = 1        # analyses run one at a time; others wait in "processing"
    retention_days: float = 7           # uploads/outputs older than this are deleted
    cleanup_interval_hours: float = 6


settings = Settings()


def serve_reference_candidates() -> list[Path]:
    paths = []
    if settings.ref_serve_path:
        paths.append(Path(settings.ref_serve_path))
    ref_dir = BACKEND_DIR / "assets" / "reference"
    paths += [ref_dir / name for name in ("serve.mov", "serve.mp4")]
    paths.append(Path(r"C:\Users\Elvis\Documents\raybaserve.mov"))   # original local dev location
    return paths
