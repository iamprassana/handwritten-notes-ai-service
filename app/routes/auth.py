from typing import Any

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from app.services.auth import auth_service, extract_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


class Credentials(BaseModel):
    email: str = Field(
        ...,
        min_length=3,
        max_length=254,
        pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$",
    )
    password: str = Field(..., min_length=6, max_length=128)


@router.post("/signup", status_code=201)
def sign_up(credentials: Credentials) -> dict[str, Any]:
    result = auth_service.sign_up(credentials.email, credentials.password)
    return {"data": result}


@router.post("/login")
def sign_in(credentials: Credentials) -> dict[str, Any]:
    result = auth_service.sign_in(credentials.email, credentials.password)
    return {"data": result}


@router.get("/me")
def get_current_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    user = auth_service.get_user(extract_access_token(authorization))
    return {
        "data": {
            "id": user["id"],
            "email": user.get("email"),
            "emailConfirmedAt": user.get("email_confirmed_at"),
            "createdAt": user.get("created_at"),
        }
    }


@router.post("/logout", status_code=204)
def sign_out(authorization: str | None = Header(default=None)) -> None:
    auth_service.sign_out(extract_access_token(authorization))
