from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.services.config import settings


class PageDetectionService:
    """Detect and crop the notebook page using YOLO segmentation."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        self.model = YOLO(model_path)

    def detect_page(self, image: Image.Image) -> Image.Image:
        image_np = np.asarray(image)

        results = self.model.predict(
            source=image_np,
            imgsz=640,
            conf=0.5,
            verbose=False,
        )

        if not results or results[0].masks is None:
            raise ValueError("No page segmentation mask was detected.")

        masks = results[0].masks.data

        if len(masks) == 0:
            raise ValueError("No page was detected.")

        # Use the largest detected mask.
        areas = [int(mask.sum().item()) for mask in masks]
        mask = masks[int(np.argmax(areas))].detach().cpu().numpy()

        mask = ((mask > 0.5).astype(np.uint8) * 255)

        mask_image = Image.fromarray(mask).resize(image.size)
        mask_np = np.asarray(mask_image)

        ys, xs = np.where(mask_np > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Detected page mask contains no pixels.")

        return image.crop(
            (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            )
        )


page_detection_service = PageDetectionService(
    settings.yolo_model_path
)
