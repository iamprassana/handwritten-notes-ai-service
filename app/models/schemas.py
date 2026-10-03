from pydantic import BaseModel, Field


class QuestionEmbeddingRequest(BaseModel):
    question: str = Field(..., min_length=1)


class EmbeddingResponse(BaseModel):
    embedding: list
    shape: list[int]


class AnswerResponse(BaseModel):
    answer: str
