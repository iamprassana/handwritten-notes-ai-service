from __future__ import annotations

import io

from PIL import Image
from storage3.exceptions import StorageException
from supabase import Client, create_client

from app.services.config import settings
from app.services.errors import ApiError


class DocumentImageStorage:
    def __init__(self, client: Client | None = None) -> None:
        if client is None:
            if not settings.supabase_service_role_key:
                raise RuntimeError(
                    "SUPABASE_SERVICE_ROLE_KEY is required for document image storage"
                )
            client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
            )
        self.client = client
        self.bucket = settings.storage_bucket

    def save(self, document_id: str, page_id: str, image: Image.Image) -> str:
        key = f"documents/{document_id}/{page_id}.png"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        try:
            self.client.storage.from_(self.bucket).upload(
                key,
                buffer.getvalue(),
                {"content-type": "image/png", "upsert": "false"},
            )
        except StorageException as exc:
            raise ApiError(
                502,
                "The page image could not be uploaded to storage.",
                "STORAGE_UPLOAD_FAILED",
            ) from exc
        return key

    def read(self, key: str) -> Image.Image:
        return Image.open(io.BytesIO(self.read_bytes(key))).convert("RGB")

    def read_bytes(self, key: str) -> bytes:
        self._validate_key(key)
        try:
            return self.client.storage.from_(self.bucket).download(key)
        except StorageException as exc:
            if self._is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise ApiError(
                502,
                "The page image could not be retrieved from storage.",
                "STORAGE_DOWNLOAD_FAILED",
            ) from exc

    def delete_document(self, document_id: str) -> None:
        prefix = f"documents/{document_id}"
        try:
            files = self.client.storage.from_(self.bucket).list(
                prefix,
                {"limit": 1000},
            )
            keys = [
                f"{prefix}/{item['name']}"
                for item in files
                if isinstance(item.get("name"), str)
                and item["name"].lower().endswith(".png")
            ]
            if keys:
                self.client.storage.from_(self.bucket).remove(keys)
        except StorageException as exc:
            raise ApiError(
                502,
                "The document images could not be deleted from storage.",
                "STORAGE_DELETE_FAILED",
            ) from exc

    def delete_key(self, key: str) -> None:
        self._validate_key(key)
        try:
            self.client.storage.from_(self.bucket).remove([key])
        except StorageException as exc:
            if not self._is_not_found(exc):
                raise ApiError(
                    502,
                    "The page image could not be deleted from storage.",
                    "STORAGE_DELETE_FAILED",
                ) from exc

    @staticmethod
    def _validate_key(key: str) -> None:
        if (
            not key
            or "\\" in key
            or key.startswith("/")
            or any(part in ("", ".", "..") for part in key.split("/"))
            or not key.lower().endswith(".png")
        ):
            raise ValueError("Invalid document image storage key.")

    @staticmethod
    def _is_not_found(exc: StorageException) -> bool:
        status_code = getattr(exc, "status_code", None)
        return str(status_code) == "404" or "not found" in str(exc).lower()


document_image_storage = DocumentImageStorage()
