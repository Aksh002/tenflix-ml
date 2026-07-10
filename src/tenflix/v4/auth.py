from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .runtime_env import load_env_file
from .web_types import AppUser


@dataclass(frozen=True, slots=True)
class AuthClaims:
    subject: str
    email: str | None
    claims: dict[str, Any]


class AuthError(RuntimeError):
    pass


class SupabaseAuthenticator:
    """Small Supabase JWT verifier.

    Production deployments should set SUPABASE_JWT_SECRET.  Local development
    can set TENFLIX_DEV_AUTH_USER_ID to bypass JWTs for UI work against a local
    database; that bypass is intentionally opt-in.
    """

    def __init__(self, jwt_secret: str | None = None, dev_user_id: str | None = None):
        load_env_file()
        self.jwt_secret = jwt_secret or os.getenv("SUPABASE_JWT_SECRET")
        self.dev_user_id = dev_user_id or os.getenv("TENFLIX_DEV_AUTH_USER_ID")
        self.dev_email = os.getenv("TENFLIX_DEV_AUTH_EMAIL", "dev@tenflix.local")

    def claims_from_authorization(self, authorization: str | None) -> AuthClaims:
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            return self._verify_token(token)
        if self.dev_user_id:
            return AuthClaims(
                _require_uuid(self.dev_user_id, "TENFLIX_DEV_AUTH_USER_ID"),
                self.dev_email,
                {"dev": True},
            )
        raise AuthError("Missing bearer token")

    def _verify_token(self, token: str) -> AuthClaims:
        if not self.jwt_secret:
            raise AuthError("SUPABASE_JWT_SECRET is required to verify bearer tokens")
        try:
            import jwt
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install TenFlix with the 'web' extra to verify JWTs") from error
        try:
            claims = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
        except Exception as error:  # pragma: no cover - exercised by integration tests with PyJWT
            raise AuthError("Invalid bearer token") from error
        subject = claims.get("sub")
        if not subject:
            raise AuthError("Bearer token is missing sub claim")
        return AuthClaims(
            _require_uuid(str(subject), "Bearer token sub claim"), claims.get("email"), claims
        )


class UnsignedDevAuthenticator(SupabaseAuthenticator):
    """Test helper that accepts unsigned JSON-ish bearer tokens.

    Not used by production code; useful for local API tests where PyJWT is not
    installed.
    """

    def _verify_token(self, token: str) -> AuthClaims:
        try:
            payload = token.split(".")[1] if "." in token else token
            padded = payload + "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(padded).decode())
        except Exception as error:
            raise AuthError("Invalid dev bearer token") from error
        subject = claims.get("sub")
        if not subject:
            raise AuthError("Bearer token is missing sub claim")
        return AuthClaims(
            _require_uuid(str(subject), "Bearer token sub claim"), claims.get("email"), claims
        )


def _require_uuid(value: str, label: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as error:
        raise AuthError(f"{label} must be a UUID") from error


def as_app_user(app_user_id: int, claims: AuthClaims, provider_region: str = "IN") -> AppUser:
    return AppUser(
        app_user_id=app_user_id,
        auth_user_id=claims.subject,
        email=claims.email,
        provider_region=provider_region,
    )
