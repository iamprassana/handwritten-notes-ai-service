from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.services.config import settings
from app.services.errors import ApiError

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Persistent Jina Embeddings v4 multivector storage."""

    _VECTOR_NAME = "jina_v4"

    def __init__(self) -> None:
        if not settings.qdrant_url or not settings.qdrant_api_key:
            raise RuntimeError(
                "QDRANT_URL and QDRANT_API_KEY are required for document indexing"
            )
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection = settings.jina_qdrant_collection
        self._embedding_size: int | None = None

    def close(self) -> None:
        self.client.close()

    def ensure_collection(self, embedding_size: int) -> None:
        if embedding_size <= 0:
            raise ApiError(422, "The embedding has no vector dimensions.", "INVALID_EMBEDDING")
        try:
            exists = self.client.collection_exists(self.collection)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        self._VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                            datatype=models.Datatype.FLOAT32,
                            hnsw_config=models.HnswConfigDiff(m=0),
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                        ),
                    },
                )
            info = self.client.get_collection(self.collection)
        except Exception as exc:
            raise ApiError(502, "Qdrant collection is unavailable.", "VECTOR_STORE_UNAVAILABLE") from exc

        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, dict):
            raise ApiError(
                409,
                "The Jina Qdrant collection must use a named jina_v4 vector.",
                "VECTOR_COLLECTION_MISMATCH",
            )
        vector_config = vectors_config.get(self._VECTOR_NAME)
        if vector_config is None:
            raise ApiError(
                409,
                "The Jina Qdrant collection is missing the jina_v4 vector.",
                "VECTOR_COLLECTION_MISMATCH",
            )
        if vector_config.size != embedding_size:
            raise ApiError(
                409,
                "The Qdrant collection dimension does not match Jina Embeddings v4.",
                "VECTOR_COLLECTION_MISMATCH",
            )
        if vector_config.datatype != models.Datatype.FLOAT32:
            raise ApiError(
                409,
                "The Qdrant collection must use float32 vectors.",
                "VECTOR_COLLECTION_MISMATCH",
            )
        if (
            vector_config.distance != models.Distance.COSINE
            or vector_config.multivector_config is None
            or vector_config.multivector_config.comparator
            != models.MultiVectorComparator.MAX_SIM
        ):
            raise ApiError(
                409,
                "The Qdrant collection is not configured for Jina MAX_SIM multivectors.",
                "VECTOR_COLLECTION_MISMATCH",
            )
        for field_name, field_schema in (
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("notebook_id", models.PayloadSchemaType.KEYWORD),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            except Exception as exc:
                raise ApiError(
                    502,
                    f"The Qdrant payload index for {field_name} could not be created.",
                    "VECTOR_INDEX_FAILED",
                ) from exc
        self._embedding_size = embedding_size

    @staticmethod
    def _as_vector(embedding: Any) -> list[list[float]]:
        shape = getattr(embedding, "shape", None)
        if shape is not None:
            if len(shape) != 2:
                raise ApiError(
                    422,
                    "Jina must return a two-dimensional multivector.",
                    "INVALID_EMBEDDING",
                )
            vectors = embedding.tolist()
        else:
            vectors = embedding
        if (
            not isinstance(vectors, list)
            or not vectors
            or not all(isinstance(vector, list) for vector in vectors)
            or not all(vector for vector in vectors)
            or len({len(vector) for vector in vectors}) != 1
        ):
            raise ApiError(
                422,
                "Jina must return a two-dimensional multivector.",
                "INVALID_EMBEDDING",
            )
        if len(vectors[0]) != 128:
            raise ApiError(
                422,
                "Jina returned a multivector with an unsupported dimension.",
                "INVALID_EMBEDDING",
            )
        return vectors

    def upsert_page(self, page_id: str, embedding: Any, payload: dict[str, Any]) -> None:
        vectors = self._as_vector(embedding)
        self.ensure_collection(len(vectors[0]))
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=page_id,
                        vector={self._VECTOR_NAME: vectors},
                        payload=payload,
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            logger.exception(
                "Qdrant upsert failed for page %s in collection %s.",
                page_id,
                self.collection,
            )
            raise ApiError(502, "The page embedding could not be stored.", "VECTOR_UPSERT_FAILED") from exc

    def search(
        self,
        embedding: Any,
        limit: int,
        document_id: str | None = None,
        notebook_id: str | None = None,
    ) -> list[Any]:
        vectors = self._as_vector(embedding)
        self.ensure_collection(len(vectors[0]))
        conditions = []
        for key, value in (("document_id", document_id), ("notebook_id", notebook_id)):
            if value:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        query_filter = models.Filter(must=conditions) if conditions else None
        try:
            result = self.client.query_points(
                collection_name=self.collection,
                query=vectors,
                using=self._VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise ApiError(502, "The page embeddings could not be searched.", "VECTOR_SEARCH_FAILED") from exc
        return result.points

    def scroll_pages(self, document_id: str | None = None) -> list[Any]:
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                with_payload=True,
                limit=10000,
            )
        except Exception as exc:
            raise ApiError(502, "The indexed documents could not be listed.", "VECTOR_LIST_FAILED") from exc
        return points

    def scroll_legacy_pages(self, document_id: str) -> list[Any]:
        """Read legacy ColQwen2 payloads without modifying the old collection."""
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )
        try:
            points, _ = self.client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=query_filter,
                with_payload=True,
                limit=10000,
            )
        except Exception as exc:
            raise ApiError(
                502,
                "The legacy document embeddings could not be read.",
                "LEGACY_VECTOR_LIST_FAILED",
            ) from exc
        return points

    def delete_document(self, document_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise ApiError(502, "The document embeddings could not be deleted.", "VECTOR_DELETE_FAILED") from exc


vector_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    if vector_store is None:
        raise RuntimeError("The Qdrant vector store has not been initialized.")
    return vector_store
