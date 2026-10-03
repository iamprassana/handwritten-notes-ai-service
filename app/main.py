from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.answer import router as answer_router
from app.routes.embedding import router as embedding_router
from app.services.embedding import embedding_service
from app.services.qwen import qwen_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load GPU models once when FastAPI starts.
    """

    print("Starting Handwritten Notes AI service...")

    embedding_service.load()
    qwen_service.load()

    print("AI models are ready.")

    yield

    print("Shutting down AI service...")


app = FastAPI(
    title="Handwritten Notes AI API",
    description=(
        "Stateless AI inference service for page detection, "
        "ColQwen2 embeddings and Qwen3-VL answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(
    embedding_router,
    prefix="/api",
)

app.include_router(
    answer_router,
    prefix="/api",
)


@app.get("/")
async def root():
    return {
        "service": "handwritten-notes-ai",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }
