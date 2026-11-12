import io
from typing import Annotated

from fastapi import APIRouter, Header, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.services.auth import auth_service, extract_access_token
from app.services.config import settings
from app.services.errors import ApiError
from app.services.gemini_service import gemini_service

router = APIRouter(prefix="/handwritten", tags=["Handwritten Notes"])

_SIGNATURES = {
    "image/jpeg": lambda content: (
        len(content) >= 3 and content[:3] == b"\xff\xd8\xff"
    ),
    "image/png": lambda content: (
        len(content) >= 8
        and content[:8] == b"\x89PNG\r\n\x1a\n"
    ),
    "image/webp": lambda content: (
        len(content) >= 12
        and content[:4] == b"RIFF"
        and content[8:12] == b"WEBP"
    ),
}


@router.post("/process")
async def process_handwritten_notes(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    token = extract_access_token(authorization)
    await run_in_threadpool(auth_service.get_user, token)

    try:
        form = await request.form(
            max_files=settings.max_image_count,
            max_fields=1,
            max_part_size=max(settings.max_image_size_bytes, 4000),
        )
    except HTTPException as exc:
        raise ApiError(
            400,
            "The uploaded files are invalid.",
            "INVALID_UPLOAD",
        ) from exc

    try:
        question = form.get("question")
        images = form.getlist("images")
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ApiError(
                400,
                "A question between 1 and 4000 characters is required.",
                "INVALID_QUESTION",
            )
        if not images:
            raise ApiError(
                400,
                "At least one image is required in the 'images' field.",
                "IMAGES_REQUIRED",
            )
        if len(images) > settings.max_image_count:
            raise ApiError(
                400,
                "The uploaded files are invalid.",
                "INVALID_UPLOAD",
            )

        pil_images = []
        for image_file in images:
            if not isinstance(image_file, UploadFile):
                raise ApiError(
                    400,
                    "At least one image is required in the 'images' field.",
                    "IMAGES_REQUIRED",
                )
            if image_file.content_type not in _SIGNATURES:
                raise ApiError(
                    415,
                    "Only JPEG, PNG, and WebP images are supported.",
                    "UNSUPPORTED_IMAGE",
                )
            content = await image_file.read(settings.max_image_size_bytes + 1)
            if len(content) > settings.max_image_size_bytes:
                raise ApiError(
                    413,
                    "An image exceeds the maximum allowed file size.",
                    "FILE_TOO_LARGE",
                )
            if not _SIGNATURES[image_file.content_type](content):
                raise ApiError(
                    415,
                    "Only valid JPEG, PNG, and WebP images are supported.",
                    "UNSUPPORTED_IMAGE",
                )
            try:
                pil_images.append(Image.open(io.BytesIO(content)).convert("RGB"))
            except (UnidentifiedImageError, OSError) as exc:
                raise ApiError(
                    415,
                    "Only valid JPEG, PNG, and WebP images are supported.",
                    "UNSUPPORTED_IMAGE",
                ) from exc

    finally:
        await form.close()

    answer = gemini_service.answer(pil_images, question.strip())
    return {"answer": answer}
