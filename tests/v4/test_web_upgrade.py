from __future__ import annotations

import base64
import builtins
import json

import pytest

from tenflix.v4.auth import AuthError, UnsignedDevAuthenticator
from tenflix.v4.enrichment import TMDbClient
from tenflix.v4.recommender import V4Recommender
from tenflix.v4.runtime_env import load_env_file
from tenflix.v4.web_service import create_product_fastapi_app
from tenflix.v4.web_repositories import PostgresConnectionFactory, _normalize_database_url
from tenflix.v4.web_repositories import external_actions
from tenflix.v4.web_types import imdb_url, stremio_url, tmdb_url


def test_external_watch_actions_use_available_ids_only():
    actions = external_actions(1, "tt0133093", 603, "movie")
    labels = {value.label for value in actions}
    assert "Open in Stremio" in labels
    assert "IMDb" in labels
    assert "TMDb" in labels
    assert stremio_url("tt0133093") == "stremio:///detail/movie/tt0133093"
    assert imdb_url("tt0133093") == "https://www.imdb.com/title/tt0133093/"
    assert tmdb_url(603) == "https://www.themoviedb.org/movie/603"


def test_missing_imdb_id_hides_stremio_action():
    actions = external_actions(1, None, 603, "movie")
    assert {value.action_type for value in actions} == {"tmdb"}


def test_unsigned_dev_authenticator_extracts_subject_and_email():
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "00000000-0000-0000-0000-000000000001", "email": "a@b.test"}).encode()
    ).decode().rstrip("=")
    claims = UnsignedDevAuthenticator(jwt_secret="unused").claims_from_authorization(
        f"Bearer header.{payload}.signature"
    )
    assert claims.subject.endswith("0001")
    assert claims.email == "a@b.test"


def test_unsigned_dev_authenticator_rejects_missing_subject():
    payload = base64.urlsafe_b64encode(json.dumps({"email": "a@b.test"}).encode()).decode().rstrip("=")
    with pytest.raises(AuthError):
        UnsignedDevAuthenticator(jwt_secret="unused").claims_from_authorization(
            f"Bearer header.{payload}.signature"
        )


def test_runtime_env_loader_reads_dotenv_without_overriding_existing(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
                [
                    "# comment",
                    "DATABASE_URL='postgresql://from-file'",
                    "export TENFLIX_TEST_ENV_LOADER=9999",
                ]
            ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://existing")
    load_env_file(path)
    assert __import__("os").environ["DATABASE_URL"] == "postgresql://existing"
    assert __import__("os").environ["TENFLIX_TEST_ENV_LOADER"] == "9999"


def test_runtime_env_loader_reads_dotenv_local_as_local_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TENFLIX_PROTECTED_ENV", "process-value")
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://from-env\nTENFLIX_PROTECTED_ENV=file-value\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "DATABASE_URL=postgresql://from-local\nTENFLIX_PROTECTED_ENV=local-value\n",
        encoding="utf-8",
    )
    load_env_file()
    assert __import__("os").environ["DATABASE_URL"] == "postgresql://from-local"
    assert __import__("os").environ["TENFLIX_PROTECTED_ENV"] == "process-value"


def test_product_api_allows_configured_cors_preflight(monkeypatch, v4_bundle):
    from fastapi.testclient import TestClient

    class Users:
        db = None

    monkeypatch.setenv("TENFLIX_CORS_ORIGINS", "http://localhost:3000")
    app = create_product_fastapi_app(
        V4Recommender(v4_bundle),
        Users(),
        UnsignedDevAuthenticator(jwt_secret="unused"),
    )
    response = TestClient(app).options(
        "/v1/catalog/rows",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_product_api_local_cors_defaults_cover_next_fallback_port(monkeypatch, v4_bundle):
    from fastapi.testclient import TestClient

    class Users:
        db = None

    monkeypatch.delenv("TENFLIX_CORS_ORIGINS", raising=False)
    app = create_product_fastapi_app(
        V4Recommender(v4_bundle),
        Users(),
        UnsignedDevAuthenticator(jwt_secret="unused"),
    )
    response = TestClient(app).options(
        "/v1/catalog/rows",
        headers={
            "Origin": "http://localhost:3001",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"


def test_authenticator_loads_dotenv_when_used_outside_cli(monkeypatch, tmp_path):
    monkeypatch.delenv("TENFLIX_DEV_AUTH_USER_ID", raising=False)
    monkeypatch.delenv("TENFLIX_DEV_AUTH_EMAIL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TENFLIX_DEV_AUTH_USER_ID=00000000-0000-0000-0000-000000000099\n"
        "TENFLIX_DEV_AUTH_EMAIL=direct@example.test\n",
        encoding="utf-8",
    )
    claims = UnsignedDevAuthenticator(jwt_secret="unused").claims_from_authorization(None)
    assert claims.subject == "00000000-0000-0000-0000-000000000099"
    assert claims.email == "direct@example.test"


def test_authenticator_rejects_non_uuid_dev_user_id(monkeypatch):
    monkeypatch.setenv("TENFLIX_DEV_AUTH_USER_ID", "local-dev-user")
    with pytest.raises(AuthError, match="TENFLIX_DEV_AUTH_USER_ID must be a UUID"):
        UnsignedDevAuthenticator(jwt_secret="unused").claims_from_authorization(None)


def test_tmdb_client_loads_dotenv_when_used_outside_cli(monkeypatch, tmp_path):
    monkeypatch.delenv("TMDB_API_TOKEN", raising=False)
    monkeypatch.delenv("TMDB_IMAGE_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TMDB_API_TOKEN=token-from-dotenv\n"
        "TMDB_IMAGE_BASE_URL=https://images.example.test/w500\n",
        encoding="utf-8",
    )
    client = TMDbClient()
    assert client.api_token == "token-from-dotenv"
    assert client.image_url("/poster.jpg") == "https://images.example.test/w500/poster.jpg"


def test_postgres_factory_fails_fast_when_psycopg_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError("simulated missing psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r'python -m pip install -e "\.\[web\]"'):
        PostgresConnectionFactory("postgresql://example.invalid/tenflix")


def test_database_url_normalization_removes_supabase_pgbouncer_parameter():
    assert _normalize_database_url(
        "postgresql://user:pass@example.supabase.com:6543/postgres?pgbouncer=true&sslmode=require"
    ) == "postgresql://user:pass@example.supabase.com:6543/postgres?sslmode=require"
