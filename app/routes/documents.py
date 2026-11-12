from __future__ import annotations

import io
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from app.services.document_storage import document_image_storage
from app.services.embedding import embedding_service
from app.services.errors import ApiError
from app.services.page_detection import page_detection_service
from app.services.pre_processing import preprocess_handwritten_image
from app.services.qwen import qwen_service
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None
    notebook_id: str | None = None


def _read_image(content: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ApiError(415, "The uploaded file is not a valid image.", "INVALID_DOCUMENT") from exc


def _search(request: DocumentSearchRequest) -> list[dict]:
    embedding = embedding_service.embed_question(request.query)
    points = get_vector_store().search(
        embedding,
        request.top_k,
        request.document_id,
        request.notebook_id,
    )
    results = []
    for point in points:
        payload = point.payload or {}
        key = payload.get("image_storage_key")
        if not key:
            continue
        try:
            document_image_storage.read(key)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ApiError(500, "A stored page image is missing.", "PAGE_IMAGE_NOT_FOUND") from exc
        results.append(
            {
                "page_id": payload.get("page_id", str(point.id)),
                "document_id": payload.get("document_id"),
                "page_number": payload.get("page_number"),
                "image_storage_key": key,
                "score": point.score,
            }
        )
    return results


@router.post("/index")
async def index_document(
    document: UploadFile = File(...),
    notebook_id: str | None = Form(default=None),
) -> dict:
    content = await document.read()
    source = _read_image(content)
    document_id = str(uuid4())
    page_id = str(uuid4())

    try:
        page = await run_in_threadpool(page_detection_service.detect_page, source)
    except ValueError as exc:
        page_detection_service.unload()
        raise ApiError(422, str(exc), "NO_PAGE_DETECTED") from exc
    finally:
        page_detection_service.unload()

    processed = await run_in_threadpool(preprocess_handwritten_image, page)
    key = document_image_storage.save(document_id, page_id, processed)
    try:
        embedding = await run_in_threadpool(embedding_service.embed_image, processed)
        payload = {
            "document_id": document_id,
            "page_id": page_id,
            "page_number": 1,
            "notebook_id": notebook_id,
            "original_filename": document.filename or "document",
            "image_storage_key": key,
            "preprocessing_applied": True,
            "embedding_model": "ColQwen2",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await run_in_threadpool(get_vector_store().upsert_page, page_id, embedding, payload)
    except Exception:
        document_image_storage.delete_key(key)
        raise

    return {
        "document_id": document_id,
        "status": "indexed",
        "total_pages": 1,
        "indexed_pages": 1,
        "failed_pages": 0,
    }


@router.post("/search")
async def search_documents(request: DocumentSearchRequest) -> dict:
    results = await run_in_threadpool(_search, request)
    return {"results": results}


@router.post("/ask")
async def ask_documents(request: DocumentSearchRequest) -> dict:
    results = await run_in_threadpool(_search, request)
    images = [
        document_image_storage.read(result["image_storage_key"])
        for result in results
    ]
    if not images:
        return {"answer": "The requested information was not found in the indexed notes.", "references": []}
    answer = await run_in_threadpool(qwen_service.answer, images, request.query)
    return {
        "answer": answer,
        "references": [
            {
                "document_id": result["document_id"],
                "page_id": result["page_id"],
                "page_number": result["page_number"],
            }
            for result in results
        ],
    }


@router.get("")
async def list_documents() -> dict:
    points = await run_in_threadpool(get_vector_store().scroll_pages)
    documents: dict[str, dict] = {}
    for point in points:
        payload = point.payload or {}
        document_id = payload.get("document_id")
        if document_id:
            entry = documents.setdefault(
                document_id,
                {"document_id": document_id, "page_count": 0, "pages": []},
            )
            entry["page_count"] += 1
            entry["pages"].append(payload)
    return {"documents": list(documents.values())}


@router.get("/{document_id}/pages")
async def get_document_pages(document_id: str) -> dict:
    points = await run_in_threadpool(get_vector_store().scroll_pages, document_id)
    if not points:
        raise ApiError(404, "The document was not found.", "DOCUMENT_NOT_FOUND")
    return {"pages": [point.payload or {} for point in points]}


@router.get("/{document_id}")
async def get_document(document_id: str) -> dict:
    pages = await get_document_pages(document_id)
    return {
        "document_id": document_id,
        "page_count": len(pages["pages"]),
        "pages": pages["pages"],
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    pages = await run_in_threadpool(get_vector_store().scroll_pages, document_id)
    if not pages:
        raise ApiError(404, "The document was not found.", "DOCUMENT_NOT_FOUND")
    await run_in_threadpool(get_vector_store().delete_document, document_id)
    document_image_storage.delete_document(document_id)
    return {"document_id": document_id, "status": "deleted"}
