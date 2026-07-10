from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> None:
    """Load a small dotenv-style file without adding a runtime dependency.

    PowerShell and cmd.exe do not automatically export dotenv values to child
    processes, but the V4 web CLI is configured through local env files.  The
    default call loads `.env` and then `.env.local`; real process variables win,
    while `.env.local` may override values loaded from `.env`.  Explicit paths
    keep the original single-file behavior.
    """

    if Path(path) == Path(".env") and not override:
        protected = set(os.environ)
        _load_single_env(Path(".env"), protected=protected, override=False)
        _load_single_env(Path(".env.local"), protected=protected, override=True)
        return

    _load_single_env(Path(path), protected=set(), override=override)


def _load_single_env(path: Path, *, protected: set[str], override: bool) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in protected or (not override and key in os.environ):
            continue
        os.environ[key] = _clean_value(value.strip())


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
