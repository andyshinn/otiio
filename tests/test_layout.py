from pathlib import Path
from typing import Any

import pytest

import otiio
from otiio._layout import read_project, resolve_root

from .conftest import TESTDATA1, TESTDATA2


@pytest.mark.parametrize(
    "path",
    [TESTDATA2 / "testdata2.otii3", TESTDATA2, TESTDATA2 / "data", str(TESTDATA2)],
)
def test_resolve_root_accepts_file_folder_and_data_dir(path: Path | str) -> None:
    assert resolve_root(path) == TESTDATA2 / "data"


def test_not_a_project(tmp_path: Path) -> None:
    with pytest.raises(otiio.ProjectNotFoundError):
        otiio.open(tmp_path)
    with pytest.raises(FileNotFoundError):  # also a FileNotFoundError
        otiio.open(tmp_path / "missing.otii3")


def test_reads_saved_version_only() -> None:
    raw = read_project(TESTDATA1)
    assert raw.format_version == 17
    assert raw.version_id == "8afbc627-fb70-4cce-8375-7e56d28e692f"
    # The fixture keeps undo history and orphaned blobs; only 8 blobs are referenced.
    all_blobs = list((TESTDATA1 / "data" / "data" / "project").iterdir())
    referenced = {m["data"]["id"] for r in raw.project["recordings"] for m in r["measurements"]}
    assert len(all_blobs) > len(referenced) == 8
    assert len(list((TESTDATA1 / "data" / "versions").iterdir())) > 1


def test_attributes_are_decoded_text() -> None:
    raw = read_project(TESTDATA2)
    names = sorted(a["recording.name"] for a in raw.attributes.values() if "recording.name" in a)
    assert names == ["TEST3", "TEST4"]


def test_unknown_format_warns(patched: Any) -> None:
    project = patched(TESTDATA2, lambda p: None)
    (project / "data" / "meta" / "project_format").write_text("18")
    with pytest.warns(otiio.UnsupportedFormatWarning, match="project_format 18"):
        proj = otiio.open(project)
    assert proj.format_version == 18
    assert len(proj.recordings) == 2


def test_known_format_does_not_warn(testdata2: Path, recwarn: pytest.WarningsRecorder) -> None:
    otiio.open(testdata2)
    assert not [w for w in recwarn if issubclass(w.category, otiio.UnsupportedFormatWarning)]
