import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    colqwen_model: str = os.getenv(
        "COLQWEN_MODEL",
        "vidore/colqwen2-v1.0-hf",
    )

    qwen_model: str = os.getenv(
        "QWEN_MODEL",
        "Qwen/Qwen3-VL-2B-Instruct",
    )

    yolo_model_path: str = os.getenv(
        "YOLO_MODEL_PATH",
        "models/best.pt",
    )


settings = Settings()
