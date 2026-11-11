import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.models.schemas import QuestionEmbeddingRequest
from app.services.embedding import embedding_service
from app.services.page_detection import page_detection_service
from app.services.pre_processing import preprocess_handwritten_image


router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"],
)


@router.post("/image")
async def generate_image_embedding(
    image: UploadFile = File(...),
):
    """
    Node.js sends the actual image.

    FastAPI:
        image
          -> YOLO
          -> preprocessing
          -> ColQwen2
          -> embedding response
    """

    try:
        image_bytes = await image.read()

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        try:
            page = page_detection_service.detect_page(pil_image)
        finally:
            page_detection_service.unload()

        processed = preprocess_handwritten_image(
            page
        )

        embedding = embedding_service.embed_image(
            processed
        )

        return {
            "embedding": embedding.tolist(),
            "shape": list(embedding.shape),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate image embedding.",
        ) from exc


@router.post("/question")
async def generate_question_embedding(
    request: QuestionEmbeddingRequest,
):
    """
    Generate a ColQwen question multivector.
    """

    try:
        embedding = embedding_service.embed_question(
            request.question
        )

        return {
            "embedding": embedding.tolist(),
            "shape": list(embedding.shape),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate question embedding.",
        ) from exc
