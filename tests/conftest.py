import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from otiio import DerivedChannel

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parent.parent / "examples"
TESTDATA1 = DATA / "testdata1"
TESTDATA2 = DATA / "testdata2"


@pytest.fixture
def testdata1() -> Path:
    return TESTDATA1 / "testdata1.otii3"


@pytest.fixture
def testdata2() -> Path:
    return TESTDATA2 / "testdata2.otii3"


def _copy_project(src: Path, dst: Path) -> None:
    """Copy a project, symlinking sample blobs so large data isn't duplicated."""

    def copy(s: str, d: str) -> None:
        if s.endswith((".dat", ".region")):
            try:
                os.symlink(os.path.abspath(s), d)
                return
            except OSError:  # e.g. Windows without symlink privilege
                pass
        shutil.copy2(s, d)

    shutil.copytree(src, dst, copy_function=copy)


def saved_project_json(project_dir: Path) -> Path:
    data = project_dir / "data"
    version = (data / "meta" / "saved_version").read_text().strip()
    return data / "versions" / version / "data" / "project.json"


PatchFn = Callable[[dict[str, Any]], None]


@pytest.fixture
def patched(tmp_path: Path) -> Callable[[Path, PatchFn], Path]:
    """Copy a fixture project into tmp_path and edit its saved project.json."""

    def make(src: Path, edit: PatchFn) -> Path:
        dst = tmp_path / src.name
        _copy_project(src, dst)
        pj = saved_project_json(dst)
        project = json.loads(pj.read_text())
        edit(project)
        pj.write_text(json.dumps(project))
        return dst

    return make


def make_signal(
    values: Any,
    rate: float = 10.0,
    *,
    kind: str = "power",
    name: str = "mp",
    offset_us: float = 0.0,
    device: str = "Test",
) -> DerivedChannel:
    """An in-memory channel for unit tests."""
    arr = np.asarray(values, dtype=np.float64)
    return DerivedChannel(
        name=name,
        kind=kind,
        rate=rate,
        offset_us=offset_us,
        device=device,
        length=len(arr),
        reader=lambda i0, i1: arr[i0:i1],
    )
