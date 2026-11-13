from __future__ import annotations

import base64
import io
import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from PIL import Image

from app.services.config import settings
from app.services.errors import ApiError

logger = logging.getLogger(__name__)

_GROUNDING_PROMPT = (
    "You are an AI assistant that answers questions strictly from the provided "
    "handwritten notes. Treat the images as your sole source of truth.\n\n"
    "Carefully read the handwriting, equations, diagrams, tables, and headings. "
    "Combine information across images when necessary. Answer clearly using only "
    "information supported by the notes. Do not introduce outside knowledge or "
    "guess. If the answer is not present or cannot be reliably determined, say "
    "so explicitly. Preserve the terminology, equations, and meaning in the notes.\n\n"
    "Every factual claim must be supported by the provided images."
)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenRouterService:
    """OpenRouter multimodal answer-generation service."""

    def answer(self, images: list[Image.Image], question: str) -> str:
        if not images:
            raise ApiError(
                400,
                "At least one retrieved page image is required.",
                "OPENROUTER_IMAGES_REQUIRED",
            )
        if not question.strip():
            raise ApiError(400, "A question is required.", "OPENROUTER_QUESTION_REQUIRED")
        if not settings.openrouter_api_key:
            raise ApiError(
                503,
                "OpenRouter inference is not configured.",
                "OPENROUTER_NOT_CONFIGURED",
            )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": _GROUNDING_PROMPT + "\n\nQuestion: " + question.strip()}
        ]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": self._image_data_url(image)}})

        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": settings.openrouter_max_tokens,
            "temperature": settings.openrouter_temperature,
        }
        return self._request(payload)

    def _request(self, payload: dict[str, Any]) -> str:
        url = f"{settings.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)

        for attempt in range(3):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise ApiError(
                    504,
                    "The OpenRouter request timed out.",
                    "OPENROUTER_TIMEOUT",
                ) from exc
            except httpx.RequestError as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise ApiError(
                    503,
                    "The OpenRouter service could not be reached.",
                    "OPENROUTER_UNAVAILABLE",
                ) from exc

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < 2:
                time.sleep(self._retry_delay(response, attempt))
                continue
            if response.is_error:
                raise self._api_error(response)
            return self._extract_answer(response)

        raise ApiError(503, "The OpenRouter service is unavailable.", "OPENROUTER_UNAVAILABLE")

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                try:
                    seconds = (
                        parsedate_to_datetime(retry_after).timestamp() - time.time()
                    )
                    return min(max(seconds, 0.0), 30.0)
                except (TypeError, ValueError, OverflowError):
                    pass
        return float(2**attempt)

    @staticmethod
    def _api_error(response: httpx.Response) -> ApiError:
        status = response.status_code
        if status in (401, 403):
            return ApiError(502, "OpenRouter rejected the API credentials.", "OPENROUTER_AUTH_FAILED")
        if status == 402:
            return ApiError(402, "OpenRouter account payment or credit is required.", "OPENROUTER_PAYMENT_REQUIRED")
        if status == 400:
            return ApiError(502, "OpenRouter rejected the request or model.", "OPENROUTER_INVALID_REQUEST")
        if status == 429:
            return ApiError(429, "The OpenRouter rate limit was reached.", "OPENROUTER_RATE_LIMITED")
        if status >= 500:
            return ApiError(503, "OpenRouter is temporarily unavailable.", "OPENROUTER_PROVIDER_ERROR")
        logger.warning("OpenRouter request failed with status %s.", status)
        return ApiError(502, "The OpenRouter request failed.", "OPENROUTER_REQUEST_FAILED")

    @staticmethod
    def _extract_answer(response: httpx.Response) -> str:
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                answer = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                ).strip()
            else:
                answer = str(content).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ApiError(
                502,
                "OpenRouter returned an invalid response.",
                "OPENROUTER_INVALID_RESPONSE",
            ) from exc
        if not answer:
            raise ApiError(502, "OpenRouter returned an empty answer.", "OPENROUTER_EMPTY_RESPONSE")
        return answer

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        try:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        except (OSError, ValueError) as exc:
            raise ApiError(
                422,
                "A retrieved page image could not be encoded.",
                "OPENROUTER_INVALID_IMAGE",
            ) from exc
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


openrouter_service = OpenRouterService()
