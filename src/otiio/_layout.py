"""Locate and read the raw files that make up an Otii 3 project.

A saved project is ``<name>.otii3`` (a small JSON stub) next to a ``data/`` folder::

    data/meta/{project_format,saved_version,current_version,saved_session_state}
    data/versions/<saved_version>/data/{project.json,attributes.db,sessions.json}
    data/data/project/<blob-id>/samples.dat

``versions/`` keeps undo history and ``data/project/`` keeps orphaned blobs, so only
objects referenced from the saved version are ever read.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import OtiioError, ProjectNotFoundError

SUPPORTED_FORMATS = frozenset({17})

JSON = dict[str, Any]


@dataclass(frozen=True)
class RawProject:
    root: Path  # the project's data/ directory
    format_version: int
    version_id: str
    project: JSON
    attributes: dict[str, dict[str, str]]
    sessions: list[JSON]
    session_state: list[JSON]

    def blob_dir(self, data_id: str) -> Path:
        return self.root / "data" / "project" / data_id


def resolve_root(path: str | Path) -> Path:
    """Find the ``data/`` directory from a ``.otii3`` file, project folder, or ``data/`` itself."""
    p = Path(path).expanduser()
    candidates = [p.parent / "data"] if p.is_file() else [p / "data", p]
    for c in candidates:
        if (c / "meta" / "saved_version").is_file():
            return c
    raise ProjectNotFoundError(f"{path}: not an Otii 3 project (no data/meta/saved_version)")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_attributes(db_path: Path) -> dict[str, dict[str, str]]:
    attrs: dict[str, dict[str, str]] = {}
    if not db_path.is_file():
        return attrs
    # Read-only URI so opening never creates or modifies anything in the project.
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        for object_id, name, data in con.execute("SELECT object_id, name, data FROM value"):
            attrs.setdefault(object_id, {})[name] = _decode(data)
    finally:
        con.close()
    return attrs


def read_project(path: str | Path) -> RawProject:
    root = resolve_root(path)
    meta = root / "meta"
    fmt_text = _read_text(meta / "project_format") if (meta / "project_format").is_file() else ""
    try:
        format_version = int(fmt_text)
    except ValueError:
        format_version = -1

    version_id = _read_text(meta / "saved_version")
    vdir = root / "versions" / version_id / "data"
    project = _read_json(vdir / "project.json")
    if project is None:
        raise OtiioError(f"{vdir / 'project.json'} is missing (saved_version {version_id})")

    session_state: list[JSON] = []
    state_file = meta / "saved_session_state"
    if state_file.is_file():
        sdir = root / "data" / "project" / _read_text(state_file)
        session_state = _read_json(sdir / "session_state_all.json") or _read_json(sdir / "session_state.json", [])

    return RawProject(
        root=root,
        format_version=format_version,
        version_id=version_id,
        project=project,
        attributes=read_attributes(vdir / "attributes.db"),
        sessions=_read_json(vdir / "sessions.json", []),
        session_state=session_state,
    )
