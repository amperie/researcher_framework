from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator


def repo_tmp_dir(*parts: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    path = root / ".tmp"
    for part in parts:
        if part:
            path /= part
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def temporary_directory(*, prefix: str, category: str = "runtime") -> Iterator[str]:
    base_dir = repo_tmp_dir(category)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=str(base_dir)) as tmpdir:
        yield tmpdir


def make_temp_dir(*, prefix: str, category: str = "runtime") -> Path:
    base_dir = repo_tmp_dir(category)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_dir)))
