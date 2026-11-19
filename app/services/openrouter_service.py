from __future__ import annotations

import base64
import io
import json
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
    "Answer the user's exact question directly in complete sentence(s). Include "
    "the relevant explanation, definition, comparison, or calculation from the "
    "notes instead of replying with a one-word or bare yes/no answer. Only use "
    "a short yes/no answer when the question explicitly asks for yes or no. "
    "If the supplied notes do not answer the question, state that clearly.\n\n"
    "Every factual claim must be supported by the provided images.\n\n"
    "Return ONLY one JSON object with exactly these fields: answer (string), "
    "broad_subject (string), specific_topic (string), subject_confidence (number "
    "from 0 to 1), evidence_found (boolean), and evidence (array). Use "
    "\"Unknown\" for either subject when the page content does not support a "
    "reliable classification. The confidence is model-reported, not a calibrated "
    "probability.\n\n"
    "Each evidence item must contain page_index (zero-based index of one supplied "
    "image), bbox ([x_min, y_min, x_max, y_max]), and evidence_description. "
    "The bbox must be the smallest rectangle containing the supporting handwritten "
    "text, equation, table, or diagram, with integer coordinates normalized from "
    "0 to 1000, where (0,0) is the image top-left. Use multiple items for "
    "multiple pages or regions. If reliable evidence cannot be located, use an "
    "empty evidence array and set evidence_found to false. Never invent page "
    "indices or coordinates."
)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenRouterService:
    """OpenRouter multimodal answer-generation service."""

    def answer(self, images: list[Image.Image], question: str) -> dict[str, Any]:
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
        return self._parse_response(self._request(payload), len(images))

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

    @classmethod
    def _parse_response(cls, raw_response: str, image_count: int) -> dict[str, Any]:
        try:
            decoder = json.JSONDecoder()
            candidate = raw_response.strip()
            if candidate.startswith("```"):
                candidate = candidate.split("\n", 1)[1]
                candidate = candidate.rsplit("```", 1)[0].strip()
            start = candidate.find("{")
            if start < 0:
                raise ValueError("JSON object not found")
            body, _ = decoder.raw_decode(candidate[start:])
            if not isinstance(body, dict):
                raise ValueError("JSON response must be an object")

            required = {
                "answer",
                "broad_subject",
                "specific_topic",
                "subject_confidence",
                "evidence_found",
                "evidence",
            }
            if not required.issubset(body):
                raise ValueError("Required response fields are missing")
            if not all(isinstance(body[field], str) and body[field].strip() for field in (
                "answer", "broad_subject", "specific_topic"
            )):
                raise ValueError("Answer and classification fields must be non-empty strings")
            confidence = body["subject_confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("Subject confidence must be a number")
            if not 0 <= confidence <= 1:
                raise ValueError("Subject confidence must be between 0 and 1")
            if not isinstance(body["evidence_found"], bool) or not isinstance(body["evidence"], list):
                raise ValueError("Evidence fields have invalid types")

            evidence: list[dict[str, Any]] = []
            for item in body["evidence"]:
                if not isinstance(item, dict):
                    raise ValueError("Evidence items must be objects")
                page_index = item.get("page_index")
                bbox = item.get("bbox")
                description = item.get("evidence_description")
                if (
                    isinstance(page_index, bool)
                    or not isinstance(page_index, int)
                    or not 0 <= page_index < image_count
                    or not isinstance(bbox, list)
                    or len(bbox) != 4
                    or not all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000 for value in bbox)
                    or bbox[0] >= bbox[2]
                    or bbox[1] >= bbox[3]
                    or not isinstance(description, str)
                    or not description.strip()
                ):
                    raise ValueError("Evidence contains an invalid page index or bounding box")
                evidence.append({
                    "page_index": page_index,
                    "bbox": bbox,
                    "evidence_description": description.strip(),
                })
            if body["evidence_found"] != bool(evidence):
                raise ValueError("evidence_found must match the evidence list")
            return {
                "answer": body["answer"].strip(),
                "broad_subject": body["broad_subject"].strip(),
                "specific_topic": body["specific_topic"].strip(),
                "subject_confidence": float(confidence),
                "evidence_found": bool(evidence),
                "evidence": evidence,
            }
        except (json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
            raise ApiError(
                502,
                "OpenRouter returned an invalid structured response.",
                "OPENROUTER_INVALID_RESPONSE",
            ) from exc

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
