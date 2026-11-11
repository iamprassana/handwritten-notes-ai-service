import torch
from PIL import Image
from transformers import AutoProcessor, ColQwen2ForRetrieval

from app.services.config import settings
from app.services.model_runtime import model_runtime


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
        if self.model is not None and self.processor is not None:
            return

        processor = AutoProcessor.from_pretrained(
            settings.colqwen_model
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = ColQwen2ForRetrieval.from_pretrained(
            settings.colqwen_model,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.processor = processor
        self.model = model.eval()

        print("ColQwen2 loaded.")

    def unload(self) -> None:
        self.model = None
        self.processor = None

    def _device(self):
        return next(self.model.parameters()).device

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        with model_runtime.use(self):
            return self._embed_image(image)

    def _embed_image(self, image: Image.Image) -> torch.Tensor:
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
        with model_runtime.use(self):
            return self._embed_question(question)

    def _embed_question(self, question: str) -> torch.Tensor:
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
