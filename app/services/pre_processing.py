import cv2
import numpy as np
from PIL import Image


def preprocess_handwritten_image(
    image: Image.Image,
    denoise_strength: int = 10,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: tuple[int, int] = (8, 8),
    sharpening_strength: float = 1.5,
) -> Image.Image:
    """
    Preprocess a handwritten notebook page.

    Keep these parameters aligned with the preprocessing used
    when your production embeddings were generated.
    """

    image_np = np.asarray(image)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR,
    )

    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        denoise_strength,
        denoise_strength,
        7,
        21,
    )

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_grid_size,
    )

    l_channel = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1 + sharpening_strength,
        blurred,
        -sharpening_strength,
        0,
    )

    result = cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(result)
