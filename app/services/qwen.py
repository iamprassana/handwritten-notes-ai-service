import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from app.services.config import settings


class QwenService:
    """Stateless Qwen3-VL inference service."""

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            settings.qwen_model
        )

        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                settings.qwen_model,
                torch_dtype="auto",
                device_map="auto",
            )
            .eval()
        )

        print("Qwen3-VL loaded.")

    def answer(
        self,
        images: list[Image.Image],
        question: str,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL is not loaded.")

        content = []

        for image in images:
            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    "You are an AI assistant that answers questions strictly from the provided "
                    "handwritten notes. Treat the images as your sole source of truth.\n\n"

                    "Your responsibilities:\n"
                    "1. Carefully read and interpret the handwriting, printed text, equations, "
                    "diagrams, tables, headings, and other visual information in the images.\n"
                    "2. Combine information across multiple note images when necessary to answer "
                    "the question.\n"
                    "3. Answer clearly and directly using only information supported by the notes.\n"
                    "4. Do not introduce outside knowledge, assumptions, or facts that are not "
                    "present in the provided notes.\n"
                    "5. If the answer is not present or cannot be reliably determined from the "
                    "images, say so explicitly instead of guessing.\n"
                    "6. If the notes contain an incomplete explanation, acknowledge the limitation "
                    "rather than filling the gap with external knowledge.\n"
                    "7. Preserve the terminology, definitions, equations, and meaning used in "
                    "the original notes.\n"
                    "8. When useful, explain the answer step-by-step using the relevant information "
                    "from the notes.\n\n"

                    "Grounding rule:\n"
                    "Every factual claim in your answer must be supported by the provided images. "
                    "If you cannot find evidence for a claim in the images, do not make that claim.\n\n"
                    f"The question asked by the user is: {question}"
                ),
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()


qwen_service = QwenService()
