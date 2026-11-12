from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.services.config import settings


class DocumentImageStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, page_id: str, image: Image.Image) -> str:
        key = f"documents/{document_id}/{page_id}.png"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        return key

    def read(self, key: str) -> Image.Image:
        return Image.open(self._path(key)).convert("RGB")

    def delete_document(self, document_id: str) -> None:
        directory = self.root / "documents" / document_id
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
            directory.rmdir()

    def delete_key(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents or path.suffix.lower() != ".png":
            raise ValueError("Invalid document image storage key.")
        return path


document_image_storage = DocumentImageStorage(settings.document_storage_dir)
