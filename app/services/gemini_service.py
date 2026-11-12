from __future__ import annotations

import io
import logging

from google import genai
from google.genai import errors, types
from PIL import Image

from app.services.config import settings
from app.services.errors import ApiError

logger = logging.getLogger(__name__)

_GROUNDING_PROMPT = (
    "You are an AI assistant that answers questions strictly from the provided "
    "handwritten notes. Treat the images as your sole source of truth.\n\n"

    "Your responsibilities:\n"
    "1. Carefully read and interpret the handwriting, printed text, equations, "
    "diagrams, tables, headings, and other visual information in the images.\n"
    "2. Combine information across multiple note images when necessary to answer "
    "the question.\n"
    "3. Answer clearly and directly using only information supported by the notes.\n"
    "4. Do not introduce outside knowledge, assumptions, or facts that are not "
    "present in the provided notes.\n"
    "5. If the answer is not present or cannot be reliably determined from the "
    "images, say so explicitly instead of guessing.\n"
    "6. If the notes contain an incomplete explanation, acknowledge the limitation "
    "rather than filling the gap with external knowledge.\n"
    "7. Preserve the terminology, definitions, equations, and meaning used in "
    "the original notes.\n"
    "8. When useful, explain the answer step-by-step using the relevant information "
    "from the notes.\n\n"

    "Grounding rule:\n"
    "Every factual claim in your answer must be supported by the provided images. "
    "If you cannot find evidence for a claim in the images, do not make that claim.\n\n"
)


class GeminiService:
    """Gemini Flash answer-generation service."""

    def answer(
        self,
        images: list[Image.Image],
        question: str,
    ) -> str:
        if not images:
            raise ApiError(
                400,
                "At least one retrieved page image is required.",
                "GEMINI_IMAGES_REQUIRED",
            )
        if not question.strip():
            raise ApiError(400, "A question is required.", "GEMINI_QUESTION_REQUIRED")
        if not settings.gemini_api_key:
            raise ApiError(
                503,
                "Gemini cloud inference is not configured.",
                "GEMINI_NOT_CONFIGURED",
            )
        return self._answer(images, question.strip())

    def _answer(
        self,
        images: list[Image.Image],
        question: str,
    ) -> str:
        contents = []
        for image in images:
            contents.append(self._image_part(image))
        contents.append(
            _GROUNDING_PROMPT
            + f"The question asked by the user is: {question}"
        )

        try:
            with genai.Client(api_key=settings.gemini_api_key) as client:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=512,
                        temperature=0,
                    ),
                )
        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code in (401, 403):
                raise ApiError(
                    502,
                    "Gemini rejected the API credentials.",
                    "GEMINI_AUTH_FAILED",
                ) from exc
            if status_code == 429:
                raise ApiError(
                    429,
                    "The Gemini API quota or rate limit was reached.",
                    "GEMINI_RATE_LIMITED",
                ) from exc
            if status_code in (408, 504):
                raise ApiError(
                    504,
                    "The Gemini API request timed out.",
                    "GEMINI_TIMEOUT",
                ) from exc
            if status_code == 404:
                raise ApiError(
                    502,
                    "The configured Gemini model is unavailable for this API project.",
                    "GEMINI_MODEL_UNAVAILABLE",
                ) from exc
            if status_code == 503:
                raise ApiError(
                    503,
                    "The configured Gemini model is temporarily unavailable.",
                    "GEMINI_PROVIDER_UNAVAILABLE",
                ) from exc
            logger.warning("Gemini API request failed with status %s.", status_code)
            raise ApiError(
                502,
                "The Gemini API request failed.",
                "GEMINI_PROVIDER_ERROR",
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise ApiError(
                504,
                "The Gemini API request timed out or could not be completed.",
                "GEMINI_TIMEOUT",
            ) from exc

        try:
            answer = response.text.strip()
        except (AttributeError, ValueError) as exc:
            raise ApiError(
                502,
                "Gemini returned an invalid response.",
                "GEMINI_INVALID_RESPONSE",
            ) from exc
        if not answer:
            raise ApiError(
                502,
                "Gemini returned an empty answer.",
                "GEMINI_EMPTY_RESPONSE",
            )
        return answer

    @staticmethod
    def _image_part(image: Image.Image) -> types.Part:
        try:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        except (OSError, ValueError) as exc:
            raise ApiError(
                422,
                "A retrieved page image could not be encoded.",
                "GEMINI_INVALID_IMAGE",
            ) from exc
        return types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/png",
        )


gemini_service = GeminiService()
