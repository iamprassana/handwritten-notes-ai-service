from __future__ import annotations

import base64
import io
import time
from typing import Any

import httpx
from PIL import Image

from app.services.config import settings
from app.services.errors import ApiError


class JinaEmbeddingService:
    """Jina Embeddings v4 multimodal embedding service."""

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(120.0, connect=10.0)

    def embed_image(self, image: Image.Image) -> list[list[float]]:
        try:
            image = image.convert("RGB")
            if max(image.size) > settings.jina_max_image_dimension:
                scale = settings.jina_max_image_dimension / max(image.size)
                resized = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                image = image.resize(resized, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        except (OSError, ValueError) as exc:
            raise ApiError(
                422,
                "The page image could not be encoded for embedding.",
                "INVALID_EMBEDDING_IMAGE",
            ) from exc
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return self._embed(
            input_value={
                "image": f"data:image/png;base64,{encoded}",
            },
            task="retrieval.passage",
        )

    def embed_question(self, question: str) -> list[list[float]]:
        if not question.strip():
            raise ApiError(400, "A question is required.", "INVALID_QUERY")
        return self._embed(input_value=question.strip(), task="retrieval.query")

    def _embed(self, input_value: str | dict[str, str], task: str) -> list[list[float]]:
        if not settings.jina_api_key:
            raise ApiError(
                503,
                "Jina embedding service is not configured.",
                "JINA_NOT_CONFIGURED",
            )
        payload = {
            "model": settings.jina_embedding_model,
            "task": task,
            "input": [input_value],
            "embedding_type": "float",
            "return_multivector": True,
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(
                    base_url=settings.jina_base_url,
                    headers={
                        "Authorization": f"Bearer {settings.jina_api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    timeout=self._timeout,
                ) as client:
                    response = client.post("/embeddings", json=payload)
                    response.raise_for_status()
                    body: Any = response.json()
                break
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise ApiError(
                    504,
                    "The Jina embedding request timed out.",
                    "JINA_TIMEOUT",
                ) from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in (401, 403):
                    raise ApiError(502, "Jina rejected the API credentials.", "JINA_AUTH_FAILED") from exc
                if status_code == 429:
                    raise ApiError(429, "The Jina embedding rate limit was reached.", "JINA_RATE_LIMITED") from exc
                raise ApiError(502, "The Jina embedding request failed.", "JINA_PROVIDER_ERROR") from exc
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise ApiError(502, "The Jina embedding service is unavailable.", "JINA_PROVIDER_ERROR") from exc
        else:
            raise ApiError(502, "The Jina embedding service is unavailable.", "JINA_PROVIDER_ERROR") from last_error

        try:
            embedding = body["data"][0]["embeddings"]
            if (
                not isinstance(embedding, list)
                or not embedding
                or not isinstance(embedding[0], list)
                or not embedding[0]
                or any(
                    not isinstance(vector, list) or len(vector) != 128
                    for vector in embedding
                )
            ):
                raise ValueError
            return embedding
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ApiError(
                502,
                "Jina returned an invalid multivector embedding.",
                "INVALID_EMBEDDING_RESPONSE",
            ) from exc


embedding_service = JinaEmbeddingService()
