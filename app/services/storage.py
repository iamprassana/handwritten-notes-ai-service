from typing import Any
from uuid import UUID

from supabase import Client, create_client
from storage3.exceptions import StorageException

from app.services.config import settings
from app.services.errors import ApiError


class SupabaseStorageService:
    def __init__(self, client: Client | None = None) -> None:
        if client is None:
            if not settings.supabase_service_role_key:
                raise RuntimeError(
                    "SUPABASE_SERVICE_ROLE_KEY is required to use the storage service"
                )
            client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
            )
        self.client = client
        self.bucket = settings.storage_bucket

    def upload(
        self,
        user_id: str,
        path: str,
        body: bytes,
        content_type: str,
    ) -> dict[str, str]:
        safe_path = self._user_path(user_id, path)
        try:
            result = self.client.storage.from_(self.bucket).upload(
                safe_path,
                body,
                {"content-type": content_type, "upsert": "false"},
            )
        except StorageException as exc:
            raise ApiError(
                502,
                "File could not be uploaded to storage.",
                "STORAGE_UPLOAD_FAILED",
            ) from exc
        result_path = result.get("path", safe_path)
        return {"path": result_path}

    def get_metadata(self, user_id: str, path: str) -> dict[str, Any]:
        safe_path = self._user_path(user_id, path)
        directory, filename = safe_path.rsplit("/", 1)
        try:
            items = self.client.storage.from_(self.bucket).list(
                directory,
                {"search": filename, "limit": 100},
            )
        except StorageException as exc:
            raise ApiError(
                502,
                "File metadata could not be retrieved.",
                "STORAGE_METADATA_FAILED",
            ) from exc
        metadata = next(
            (item for item in items if item.get("name") == filename),
            None,
        )
        if metadata is None:
            raise ApiError(404, "Stored file was not found.", "STORAGE_FILE_NOT_FOUND")
        return {"path": safe_path, **metadata}

    def delete(self, user_id: str, path: str) -> None:
        safe_path = self._user_path(user_id, path)
        try:
            self.client.storage.from_(self.bucket).remove([safe_path])
        except StorageException as exc:
            raise ApiError(
                502,
                "File could not be deleted from storage.",
                "STORAGE_DELETE_FAILED",
            ) from exc

    @staticmethod
    def _user_path(user_id: str, path: str) -> str:
        try:
            UUID(user_id)
        except (ValueError, AttributeError) as exc:
            raise ApiError(
                400,
                "A valid user and relative storage path are required.",
                "INVALID_STORAGE_PATH",
            ) from exc
        segments = path.split("/")
        if (
            not path
            or path.startswith("/")
            or "\\" in path
            or any(segment in ("", ".", "..") for segment in segments)
        ):
            raise ApiError(
                400,
                "Storage paths must not contain empty or traversal segments.",
                "INVALID_STORAGE_PATH",
            )
        return f"{user_id}/{path}"
