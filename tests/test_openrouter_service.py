import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from app.services.errors import ApiError
from app.services.openrouter_service import OpenRouterService


def response_payload(**overrides):
    payload = {
        "answer": "The notes define information retrieval.",
        "broad_subject": "Computer Science",
        "specific_topic": "Information Retrieval",
        "subject_confidence": 0.94,
        "evidence_found": True,
        "evidence": [
            {
                "page_index": 0,
                "bbox": [120, 250, 780, 420],
                "evidence_description": "Definition in the notes.",
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


class OpenRouterServiceTests(unittest.TestCase):
    def test_parses_grounded_answer_and_multiple_evidence_boxes(self):
        parsed = OpenRouterService._parse_response(
            "```json\n"
            + response_payload(
                evidence=[
                    {
                        "page_index": 0,
                        "bbox": [1, 2, 300, 400],
                        "evidence_description": "First page.",
                    },
                    {
                        "page_index": 1,
                        "bbox": [500, 600, 900, 999],
                        "evidence_description": "Second page.",
                    },
                ]
            )
            + "\n```",
            2,
        )
        self.assertEqual(parsed["broad_subject"], "Computer Science")
        self.assertEqual(len(parsed["evidence"]), 2)
        self.assertEqual(parsed["evidence"][1]["page_index"], 1)

    def test_accepts_unknown_subject_and_no_evidence(self):
        parsed = OpenRouterService._parse_response(
            response_payload(
                broad_subject="Unknown",
                specific_topic="Unknown",
                subject_confidence=0,
                evidence_found=False,
                evidence=[],
            ),
            1,
        )
        self.assertFalse(parsed["evidence_found"])
        self.assertEqual(parsed["evidence"], [])

    def test_rejects_invalid_json_missing_fields_and_invalid_evidence(self):
        invalid_responses = [
            "not json",
            json.dumps({"answer": "Incomplete"}),
            response_payload(evidence=[{"page_index": 0, "bbox": [0, 0, 1001, 2]}]),
            response_payload(evidence=[{"page_index": 2, "bbox": [0, 0, 10, 10], "evidence_description": "x"}]),
            response_payload(subject_confidence=1.1),
            response_payload(evidence_found=False),
        ]
        for invalid in invalid_responses:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ApiError):
                    OpenRouterService._parse_response(invalid, 2)

    @patch("app.services.openrouter_service.time.sleep")
    @patch("app.services.openrouter_service.httpx.Client")
    def test_retries_provider_failure(self, client_class, sleep):
        first = Mock(status_code=503, headers={}, is_error=True)
        second = Mock(status_code=200, headers={}, is_error=False)
        second.json.return_value = {
            "choices": [{"message": {"content": response_payload()}}]
        }
        client = client_class.return_value.__enter__.return_value
        client.post.side_effect = [first, second]

        test_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            openrouter_base_url="https://openrouter.example",
            openrouter_model="test-model",
            openrouter_max_tokens=100,
            openrouter_temperature=0,
        )
        with patch("app.services.openrouter_service.settings", test_settings):
            result = OpenRouterService().answer([Image.new("RGB", (2, 2))], "What is this?")

        self.assertEqual(result["answer"], "The notes define information retrieval.")
        self.assertEqual(client.post.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
