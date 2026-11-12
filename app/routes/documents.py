from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.services.document_service import document_service
from app.services.document_storage import document_image_storage
from app.services.embedding import embedding_service
from app.services.errors import ApiError
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None
    notebook_id: str | None = None


@router.post("/index")
async def index_document(
    document: UploadFile = File(...),
    notebook_id: str | None = Form(default=None),
) -> dict:
    content = await document.read()
    return await run_in_threadpool(
        document_service.index,
        content,
        document.filename,
        notebook_id,
    )


@router.post("/search")
async def search_documents(request: DocumentSearchRequest) -> dict:
    results = await run_in_threadpool(
        document_service.search,
        request.query,
        request.top_k,
        request.document_id,
        request.notebook_id,
    )
    return {"results": results}


@router.post("/ask")
async def ask_documents(request: DocumentSearchRequest) -> dict:
    return await run_in_threadpool(
        document_service.answer,
        request.query,
        request.top_k,
        request.document_id,
        request.notebook_id,
    )


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


@router.post("/{document_id}/reindex")
async def reindex_document(document_id: str) -> dict:
    legacy_points = await run_in_threadpool(
        get_vector_store().scroll_legacy_pages,
        document_id,
    )
    if not legacy_points:
        raise ApiError(
            404,
            "The document was not found in the legacy embedding collection.",
            "DOCUMENT_NOT_FOUND",
        )

    for point in legacy_points:
        payload = dict(point.payload or {})
        key = payload.get("image_storage_key")
        if not key:
            raise ApiError(
                500,
                "A legacy page is missing its stored image.",
                "PAGE_IMAGE_NOT_FOUND",
            )
        image = document_image_storage.read(key)
        embedding = await run_in_threadpool(embedding_service.embed_image, image)
        payload["embedding_model"] = "Jina Embeddings v4"
        await run_in_threadpool(
            get_vector_store().upsert_page,
            str(point.id),
            embedding,
            payload,
        )
    return {
        "document_id": document_id,
        "status": "reindexed",
        "indexed_pages": len(legacy_points),
    }


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
