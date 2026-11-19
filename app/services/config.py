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


def _bounded_float(name: str, fallback: float, minimum: float, maximum: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return fallback
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    jina_api_key: str = os.getenv("JINA_API_KEY", "").strip()
    jina_embedding_model: str = os.getenv(
        "JINA_EMBEDDING_MODEL",
        "jina-embeddings-v4",
    ).strip()
    jina_base_url: str = os.getenv(
        "JINA_BASE_URL",
        "https://api.jina.ai/v1",
    ).rstrip("/")
    jina_max_image_dimension: int = _positive_integer(
        "JINA_MAX_IMAGE_DIMENSION",
        2048,
    )

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    ).strip()
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "qwen/qwen3-vl-8b-instruct",
    ).strip()
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ).rstrip("/")
    openrouter_max_tokens: int = _positive_integer("OPENROUTER_MAX_TOKENS", 1024)
    openrouter_temperature: float = _bounded_float(
        "OPENROUTER_TEMPERATURE",
        0.2,
        0.0,
        2.0,
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
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION", "handwritten_pages"
    ).strip()
    jina_qdrant_collection: str = os.getenv(
        "JINA_QDRANT_COLLECTION", "handwritten_pages_jina"
    ).strip()
    def __post_init__(self) -> None:
        if self.llm_provider not in {"gemini", "openrouter"}:
            raise ValueError("LLM_PROVIDER must be either gemini or openrouter")
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
        if self.qdrant_url:
            parsed = urlparse(self.qdrant_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("QDRANT_URL must be a valid URL")
        if not self.qdrant_collection:
            raise ValueError("QDRANT_COLLECTION must not be empty")
        if not self.jina_qdrant_collection:
            raise ValueError("JINA_QDRANT_COLLECTION must not be empty")
        parsed = urlparse(self.openrouter_base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("OPENROUTER_BASE_URL must be a valid URL")


settings = Settings()
