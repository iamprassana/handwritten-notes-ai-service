import re
from typing import Any

import httpx

from app.services.config import settings
from app.services.errors import ApiError


class SupabaseAuthService:
    def __init__(self) -> None:
        self.base_url = f"{settings.supabase_url}/auth/v1"
        self.api_key = settings.supabase_publishable_key

    def sign_up(self, email: str, password: str) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/signup",
            json={"email": email, "password": password},
        )
        data = self._json(response)
        if response.status_code >= 400:
            message = str(data.get("msg") or data.get("message") or "").lower()
            if response.status_code == 409 or "already" in message or "registered" in message:
                raise ApiError(
                    409,
                    "An account with this email already exists.",
                    "EMAIL_EXISTS",
                )
            raise ApiError(
                400,
                str(data.get("msg") or data.get("message") or "Sign-up failed."),
                "SIGNUP_FAILED",
            )
        result = self._auth_result(data)
        user = result.get("user")
        if isinstance(user, dict) and user.get("identities") == []:
            raise ApiError(
                409,
                "An account with this email already exists.",
                "EMAIL_EXISTS",
            )
        return result

    def sign_in(self, email: str, password: str) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/token?grant_type=password",
            json={"email": email, "password": password},
        )
        data = self._json(response)
        if response.status_code >= 400:
            if response.status_code == 429:
                raise ApiError(429, "Too many sign-in attempts.", "RATE_LIMITED")
            raise ApiError(401, "Invalid email or password.", "INVALID_CREDENTIALS")
        return self._auth_result(data)

    def get_user(self, access_token: str) -> dict[str, Any]:
        response = self._request(
            "GET",
            "/user",
            access_token=access_token,
        )
        if response.status_code in (401, 403):
            raise ApiError(
                401,
                "Access token is invalid or expired.",
                "INVALID_TOKEN",
            )
        if response.status_code >= 400:
            raise ApiError(
                502,
                "Supabase Auth is unavailable.",
                "AUTH_SERVICE_UNAVAILABLE",
            )
        user = self._json(response)
        if not user.get("id"):
            raise ApiError(
                401,
                "Access token is invalid or expired.",
                "INVALID_TOKEN",
            )
        return user

    def sign_out(self, access_token: str) -> None:
        response = self._request(
            "POST",
            "/logout",
            access_token=access_token,
        )
        if response.status_code == 401:
            raise ApiError(401, "Session is invalid or expired.", "INVALID_TOKEN")
        if response.status_code >= 400:
            raise ApiError(502, "Could not end the Supabase session.", "LOGOUT_FAILED")

    def _request(
        self,
        method: str,
        path: str,
        *,
        access_token: str | None = None,
        json: dict[str, str] | None = None,
    ) -> httpx.Response:
        headers = {"apikey": self.api_key}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        try:
            with httpx.Client(timeout=10.0) as client:
                return client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=headers,
                    json=json,
                )
        except httpx.RequestError as exc:
            raise ApiError(
                502,
                "Supabase Auth is unavailable.",
                "AUTH_SERVICE_UNAVAILABLE",
            ) from exc

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _auth_result(data: dict[str, Any]) -> dict[str, Any]:
        if "user" in data and "session" in data:
            return {"user": data["user"], "session": data["session"]}

        session_keys = {
            "access_token",
            "token_type",
            "expires_in",
            "expires_at",
            "refresh_token",
            "provider_token",
            "provider_refresh_token",
        }
        user = data.get("user")
        if user is None:
            user = {key: value for key, value in data.items() if key not in session_keys}
        session = None
        if data.get("access_token"):
            session = {
                key: value for key, value in data.items() if key in session_keys
            }
            session["user"] = user
        return {"user": user, "session": session}


auth_service = SupabaseAuthService()


def extract_access_token(authorization: str | None) -> str:
    match = re.fullmatch(r"Bearer\s+(\S+)", authorization or "", re.IGNORECASE)
    if not match:
        raise ApiError(
            401,
            "A valid bearer access token is required.",
            "AUTH_REQUIRED",
        )
    return match.group(1)
