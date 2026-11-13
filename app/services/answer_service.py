from app.services.config import settings
from app.services.errors import ApiError
from app.services.gemini_service import gemini_service
from app.services.openrouter_service import openrouter_service


def get_answer_service():
    if settings.llm_provider == "openrouter":
        return openrouter_service
    if settings.llm_provider == "gemini":
        return gemini_service
    raise ApiError(
        500,
        "The configured answer-generation provider is unavailable.",
        "LLM_PROVIDER_UNAVAILABLE",
    )
