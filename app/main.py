from contextlib import asynccontextmanager

import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.routes.auth import router as auth_router
from app.routes.answer import router as answer_router
from app.routes.embedding import router as embedding_router
from app.routes.handwritten import router as handwritten_router
from app.routes.documents import router as documents_router
from app.services.config import settings
from app.services.errors import ApiError
import app.services.vector_store as vector_store_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Handwritten Notes API; inference models load on demand.")
    if settings.qdrant_url and settings.qdrant_api_key:
        vector_store_service.vector_store = vector_store_service.QdrantVectorStore()
    try:
        yield
    finally:
        if vector_store_service.vector_store is not None:
            vector_store_service.vector_store.close()
            vector_store_service.vector_store = None
        logger.info("Shut down Handwritten Notes API.")


app = FastAPI(
    title="Handwritten Notes API",
    description=(
        "Authenticated API for handwritten notes, account access, "
        "Jina Embeddings v4 and Gemini Flash answers."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(
    embedding_router,
    prefix="/api",
)

app.include_router(
    answer_router,
    prefix="/api",
)
app.include_router(auth_router, prefix="/api")
app.include_router(handwritten_router, prefix="/api")
app.include_router(documents_router, prefix="/api")


@app.exception_handler(ApiError)
async def api_error_handler(_request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": str(exc)}},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "INTERNAL_ERROR" if exc.status_code >= 500 else "INVALID_REQUEST",
                "message": detail,
            }
        },
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if request.url.path in ("/api/auth/signup", "/api/auth/login"):
        missing = any(error["type"] == "missing" for error in errors)
        message = (
            "Email and password are required."
            if missing
            else "Provide a valid email and a password between 6 and 128 characters."
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_CREDENTIALS_BODY",
                    "message": message,
                }
            },
        )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "The request is invalid.",
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled request error at %s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected server error occurred.",
            }
        },
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
