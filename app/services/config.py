import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _positive_integer(name: str, fallback: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return fallback
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class Settings:
    colqwen_model: str = os.getenv(
        "COLQWEN_MODEL",
        "vidore/colqwen2-v1.0-hf",
    )

    qwen_model: str = os.getenv(
        "QWEN_MODEL",
        "Qwen/Qwen3-VL-2B-Instruct",
    )

    yolo_model_path: str = os.getenv(
        "YOLO_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "best.pt"),
    )

    port: int = _positive_integer("PORT", 3000)
    frontend_origin: str = _required("FRONTEND_ORIGIN")
    supabase_url: str = _required("SUPABASE_URL").rstrip("/")
    supabase_publishable_key: str = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )
    supabase_service_role_key: str = os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY", ""
    ).strip()
    storage_bucket: str = os.getenv(
        "SUPABASE_STORAGE_BUCKET", "handwritten-files"
    ).strip()
    max_image_count: int = _positive_integer("MAX_IMAGE_COUNT", 5)
    max_image_size_bytes: int = _positive_integer(
        "MAX_IMAGE_SIZE_BYTES", 10 * 1024 * 1024
    )

    def __post_init__(self) -> None:
        if not self.supabase_publishable_key:
            raise ValueError(
                "Set SUPABASE_PUBLISHABLE_KEY (or SUPABASE_ANON_KEY) in the environment"
            )
        for name, value in (
            ("FRONTEND_ORIGIN", self.frontend_origin),
            ("SUPABASE_URL", self.supabase_url),
        ):
            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"{name} must be a valid URL")


settings = Settings()
