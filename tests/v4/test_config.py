from pathlib import Path

import pytest
import yaml

from tenflix.v4.config import DEFAULT_CONFIG, load_config


def test_packaged_defaults_match_repository_config():
    repository_config = yaml.safe_load(Path("configs/v4.yaml").read_text(encoding="utf-8"))
    assert DEFAULT_CONFIG == repository_config


def test_missing_config_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
