from pydantic import BaseModel, Field


class QuestionEmbeddingRequest(BaseModel):
    question: str = Field(..., min_length=1)


class EmbeddingResponse(BaseModel):
    embedding: list
    shape: list[int]


class AnswerResponse(BaseModel):
    answer: str
    broad_subject: str = "Unknown"
    specific_topic: str = "Unknown"
    subject_confidence: float = Field(default=0.0, ge=0, le=1)
    evidence_found: bool = False
    evidence: list[dict] = Field(default_factory=list)
