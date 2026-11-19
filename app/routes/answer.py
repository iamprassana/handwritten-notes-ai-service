import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.services.answer_service import get_answer_service
from app.services.errors import ApiError


router = APIRouter(
    prefix="/answer",
    tags=["Question Answering"],
)


@router.post("")
async def answer_question(
    question: str = Form(...),
    images: list[UploadFile] = File(...),
):
    """
    Node.js sends:
        - the user's question
        - only the images selected by Node.js retrieval

    FastAPI does not retrieve anything.
    It simply passes those images to the configured answer provider.
    """

    try:
        if not images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required.",
            )

        pil_images = []

        for image_file in images:
            image_bytes = await image_file.read()

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            pil_images.append(image)

        generated = get_answer_service().answer(
            images=pil_images,
            question=question,
        )

        return generated if isinstance(generated, dict) else {"answer": generated}

    except HTTPException:
        raise

    except ApiError:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer.",
        ) from exc
