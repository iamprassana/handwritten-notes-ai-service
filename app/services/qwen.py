"""Backward-compatible import for clients using the old service module name."""

from app.services.gemini_service import GeminiService, gemini_service

QwenService = GeminiService
qwen_service = gemini_service

__all__ = ["GeminiService", "QwenService", "gemini_service", "qwen_service"]
