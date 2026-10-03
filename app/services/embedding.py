import torch
from PIL import Image
from transformers import AutoProcessor, ColQwen2ForRetrieval

from app.services.config import settings


class ColQwenEmbeddingService:
    """
    Stateless inference service for ColQwen2.

    FastAPI does not store embeddings, access databases,
    access object storage, or access Qdrant. It only receives
    data and returns the generated embedding.
    """

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self) -> None:
        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(
            settings.colqwen_model
        )

        self.model = (
            ColQwen2ForRetrieval.from_pretrained(
                settings.colqwen_model,
                torch_dtype=dtype,
            )
            .to(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            .eval()
        )

        print("ColQwen2 loaded.")

    def _device(self):
        return next(self.model.parameters()).device

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        if self.model is None or self.processor is None:
            raise RuntimeError("ColQwen2 is not loaded.")

        inputs = self.processor(
            images=[image],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device())
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model(**inputs)

        return output.embeddings[0].detach().cpu()

    def embed_question(self, question: str) -> torch.Tensor:
        if self.model is None or self.processor is None:
            raise RuntimeError("ColQwen2 is not loaded.")

        inputs = self.processor(
            text=[question],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device())
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model(**inputs)

        return output.embeddings[0].detach().cpu()


embedding_service = ColQwenEmbeddingService()
