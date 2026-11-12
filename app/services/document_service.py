from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.services.document_storage import document_image_storage
from app.services.embedding import embedding_service
from app.services.errors import ApiError
from app.services.gemini_service import gemini_service
from app.services.page_detection import page_detection_service
from app.services.pre_processing import preprocess_handwritten_image
from app.services.vector_store import get_vector_store


class DocumentService:
    """Orchestrates page processing, embedding, retrieval, and answering."""

    def index(
        self,
        content: bytes,
        filename: str | None,
        notebook_id: str | None,
    ) -> dict[str, Any]:
        source = self._read_image(content)
        document_id = str(uuid4())
        page_id = str(uuid4())

        try:
            page = page_detection_service.detect_page(source)
        except ValueError as exc:
            raise ApiError(422, str(exc), "NO_PAGE_DETECTED") from exc
        finally:
            page_detection_service.unload()

        processed = preprocess_handwritten_image(page)
        key = document_image_storage.save(document_id, page_id, processed)
        try:
            embedding = embedding_service.embed_image(processed)
            payload = {
                "document_id": document_id,
                "page_id": page_id,
                "page_number": 1,
                "notebook_id": notebook_id,
                "original_filename": filename or "document",
                "image_storage_key": key,
                "preprocessing_applied": True,
                "embedding_model": "Jina Embeddings v4",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            get_vector_store().upsert_page(page_id, embedding, payload)
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

    def search(
        self,
        query: str,
        limit: int,
        document_id: str | None = None,
        notebook_id: str | None = None,
    ) -> list[dict[str, Any]]:
        embedding = embedding_service.embed_question(query)
        points = get_vector_store().search(
            embedding,
            limit,
            document_id,
            notebook_id,
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
                raise ApiError(
                    500,
                    "A stored page image is missing.",
                    "PAGE_IMAGE_NOT_FOUND",
                ) from exc
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

    def answer(
        self,
        query: str,
        limit: int,
        document_id: str | None = None,
        notebook_id: str | None = None,
    ) -> dict[str, Any]:
        results = self.search(query, limit, document_id, notebook_id)
        images = [
            document_image_storage.read(result["image_storage_key"])
            for result in results
        ]
        if not images:
            return {
                "answer": "The requested information was not found in the indexed notes.",
                "references": [],
            }
        return {
            "answer": gemini_service.answer(images, query),
            "references": [
                {
                    "document_id": result["document_id"],
                    "page_id": result["page_id"],
                    "page_number": result["page_number"],
                }
                for result in results
            ],
        }

    @staticmethod
    def _read_image(content: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ApiError(
                415,
                "The uploaded file is not a valid image.",
                "INVALID_DOCUMENT",
            ) from exc


document_service = DocumentService()
