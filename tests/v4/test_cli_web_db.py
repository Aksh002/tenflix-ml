from __future__ import annotations

from tenflix.v4 import cli


def test_check_web_db_returns_success_when_schema_is_complete(monkeypatch, capsys):
    class FakeDb:
        def schema_status(self):
            return {"profiles": True, "catalog_items": True}

    monkeypatch.setattr("tenflix.v4.web_repositories.PostgresConnectionFactory", lambda: FakeDb())
    assert cli.main(["check-web-db"]) == 0
    assert '"profiles": true' in capsys.readouterr().out


def test_check_web_db_returns_failure_when_schema_is_incomplete(monkeypatch):
    class FakeDb:
        def schema_status(self):
            return {"profiles": True, "catalog_items": False}

    monkeypatch.setattr("tenflix.v4.web_repositories.PostgresConnectionFactory", lambda: FakeDb())
    assert cli.main(["check-web-db"]) == 2


def test_migrate_web_db_applies_migration_then_checks_schema(monkeypatch, tmp_path, capsys):
    migration = tmp_path / "migration.sql"
    migration.write_text("select 1;", encoding="utf-8")
    calls = []

    class FakeDb:
        def apply_sql_path(self, path):
            calls.append(path)
            return [path]

        def schema_status(self):
            return {"profiles": True}

    monkeypatch.setattr("tenflix.v4.web_repositories.PostgresConnectionFactory", lambda: FakeDb())
    assert cli.main(["migrate-web-db", "--migration", str(migration)]) == 0
    assert calls == [str(migration)]
    assert '"migration"' in capsys.readouterr().out
