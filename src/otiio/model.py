"""Object model for an opened Otii 3 project."""

import datetime as dt
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, overload

import numpy as np

from ._layout import JSON, SUPPORTED_FORMATS, RawProject, read_project
from .channels import channel_info, rail_channel_name, unit_for
from .errors import AlignmentError, AmbiguousNameError, MissingDataError, UnsupportedFormatWarning
from .signal import FloatArray, Signal

if TYPE_CHECKING:
    import pandas as pd

    from .derived import DerivedChannel

SAMPLE_DTYPE = np.dtype("<f4")


@dataclass(frozen=True)
class Device:
    id: str
    type: str
    name: str
    serial: str | None
    settings: dict[str, Any]
    raw: JSON = field(repr=False)


@dataclass(frozen=True)
class Selection:
    """A named time range saved in the project, in microseconds on the project timeline."""

    id: str
    name: str
    color: str | None
    start_us: float
    end_us: float
    visible: bool = True
    note: str | None = None

    @property
    def start(self) -> float:
        return self.start_us / 1e6

    @property
    def end(self) -> float:
        return self.end_us / 1e6

    @property
    def duration(self) -> float:
        return (self.end_us - self.start_us) / 1e6


class Channel(Signal):
    """One stored measurement of one recording, read from its ``samples.dat``."""

    derived = False

    def __init__(
        self,
        *,
        id: str,
        name: str,
        kind: str,
        source_type: str,
        rate: float,
        offset_us: float,
        device: str,
        device_id: str | None,
        path: Path | None,
        raw: JSON,
    ) -> None:
        info = channel_info(name, kind)
        self.id = id
        self.name = name
        self.kind = kind
        self.unit = unit_for(kind)
        self.label_base = info.label
        self.rail = info.rail
        self.source_type = source_type
        self.rate = rate
        self.offset_us = offset_us
        self.device = device
        self.device_id = device_id
        self.path = path
        self.raw = raw
        self._n: int | None = None
        self._recording: Recording | None = None  # set by Recording

    def _matching_power(self) -> "Signal | None":
        if self._recording is None or self.kind != "current" or self.rail is None:
            return None
        if self.name != rail_channel_name(self.rail, "current"):
            return None  # e.g. mc1, which has no power channel of its own
        try:
            return self._recording.power(self.rail, self.device_id or self.device)
        except (KeyError, AlignmentError):
            return None

    def _check_readable(self) -> Path:
        if self.source_type != "samples":
            raise NotImplementedError(
                f"channel {self.name!r} has type {self.source_type!r}; only 'samples' channels can be read so far"
            )
        if self.path is None or not self.path.is_file():
            raise MissingDataError(f"samples for channel {self.name!r} not found: {self.path}")
        return self.path

    def _base_len(self) -> int:
        if self._n is None:
            self._n = self._check_readable().stat().st_size // SAMPLE_DTYPE.itemsize
        return self._n

    def _read_base(self, i0: int, i1: int) -> FloatArray:
        path = self._check_readable()
        if i1 <= i0 or self._base_len() == 0:
            return np.empty(0, dtype=SAMPLE_DTYPE)
        mm: FloatArray = np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")
        return mm[i0:i1]


class _Named(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str | None: ...


class NamedList[T: _Named](Sequence[T]):
    """A list that can also be indexed by name or id. Positions follow project.json order."""

    def __init__(self, items: list[T], what: str) -> None:
        self._items = items
        self._what = what

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> list[T]: ...
    @overload
    def __getitem__(self, key: str) -> T: ...
    def __getitem__(self, key: int | slice | str) -> T | list[T]:
        if not isinstance(key, str):
            return self._items[key]
        by_id = [x for x in self._items if x.id == key]
        if by_id:
            return by_id[0]
        by_name = [x for x in self._items if x.name == key]
        if len(by_name) == 1:
            return by_name[0]
        if not by_name:
            raise KeyError(f"no {self._what} named {key!r}; have {self.names()}")
        ids = [x.id for x in by_name]
        raise AmbiguousNameError(f"{len(ids)} {self._what}s are named {key!r}; use an id: {ids}")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return any(key in (x.id, x.name) for x in self._items)
        return key in self._items

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NamedList):
            return self._items == other._items
        if isinstance(other, list):
            return self._items == other
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def names(self) -> list[str | None]:
        return [x.name for x in self._items]

    def __repr__(self) -> str:
        return f"NamedList({self._items!r})"


class ChannelMap(Mapping[str, Channel]):
    """Channels of a recording keyed by short name (``"mc"``), independent of stored order.

    When two devices share a short name, plain lookup raises
    :class:`~otiio.errors.AmbiguousNameError`; use ``recording[name, device]``.
    """

    def __init__(self, channels: list[Channel]) -> None:
        self._all = channels
        self._by_name: dict[str, list[Channel]] = {}
        for ch in channels:
            self._by_name.setdefault(ch.name, []).append(ch)

    def __getitem__(self, name: str) -> Channel:
        matches = self._by_name.get(name)
        if not matches:
            raise KeyError(f"no channel {name!r}; have {sorted(self._by_name)}")
        if len(matches) > 1:
            devices = [c.device for c in matches]
            raise AmbiguousNameError(
                f"channel {name!r} exists on several devices {devices}; use recording[{name!r}, <device>]"
            )
        return matches[0]

    def get_on(self, name: str, device: str) -> Channel:
        for ch in self._by_name.get(name, []):
            if device in (ch.device, ch.device_id):
                return ch
        raise KeyError(f"no channel {name!r} on device {device!r}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def all(self) -> list[Channel]:
        """Every channel, including same-named channels on different devices."""
        return list(self._all)

    def __repr__(self) -> str:
        return f"ChannelMap({self._all!r})"


class Recording:
    def __init__(
        self,
        *,
        id: str,
        name: str | None,
        start_time_ms: int,
        offset_us: float,
        channels: list[Channel],
        attributes: dict[str, str],
        raw: JSON,
    ) -> None:
        self.id = id
        self.name = name
        self.start_time_ms = start_time_ms
        self.offset_us = offset_us
        self.channels = ChannelMap(channels)
        for ch in channels:
            ch._recording = self
        self.attributes = attributes
        self.raw = raw

    @property
    def start(self) -> dt.datetime:
        """Wall-clock start of the recording (UTC)."""
        return dt.datetime.fromtimestamp(self.start_time_ms / 1000, dt.UTC)

    @cached_property
    def duration(self) -> float:
        """Seconds from the timeline origin to the end of the longest channel."""
        ends = [
            ch.span_end
            for ch in self.channels.all()
            if ch.source_type == "samples" and ch.path is not None and ch.path.is_file()
        ]
        return max(ends, default=0.0)

    def __getitem__(self, key: str | tuple[str, str]) -> Channel:
        if isinstance(key, tuple):
            return self.channels.get_on(*key)
        return self.channels[key]

    def __contains__(self, name: object) -> bool:
        return name in self.channels

    def channel(self, name: str, device: str | None = None) -> Channel:
        return self.channels[name] if device is None else self.channels.get_on(name, device)

    # -- rail quantities: stored when available, otherwise derived ----------------------

    def _rail(self, rail: str, kind: str, device: str | None) -> Channel | None:
        name = rail_channel_name(rail, kind)
        try:
            return self.channel(name, device)
        except KeyError:
            return None

    def _need(self, rail: str, kind: str, device: str | None, want: str) -> Channel:
        ch = self._rail(rail, kind, device)
        if ch is None:
            raise KeyError(
                f"{want} on rail {rail!r} is not stored and cannot be derived: no {kind} channel in recording {self.name!r}"
            )
        return ch

    def current(self, rail: str = "Main", device: str | None = None) -> "Channel | DerivedChannel":
        """Stored current, or power / voltage when current was not recorded."""
        from .derived import divide

        stored = self._rail(rail, "current", device)
        if stored is not None:
            return stored
        p = self._need(rail, "power", device, "current")
        v = self._need(rail, "voltage", device, "current")
        return divide(p, v, name=rail_channel_name(rail, "current"), kind="current")

    def voltage(self, rail: str = "Main", device: str | None = None) -> "Channel | DerivedChannel":
        """Stored voltage, or power / current when voltage was not recorded."""
        from .derived import divide

        stored = self._rail(rail, "voltage", device)
        if stored is not None:
            return stored
        p = self._need(rail, "power", device, "voltage")
        i = self._need(rail, "current", device, "voltage")
        return divide(p, i, name=rail_channel_name(rail, "voltage"), kind="voltage")

    def power(self, rail: str = "Main", device: str | None = None) -> "Channel | DerivedChannel":
        """Stored power, or current x voltage when power was not recorded."""
        from .derived import multiply

        stored = self._rail(rail, "power", device)
        if stored is not None:
            return stored
        i = self._need(rail, "current", device, "power")
        v = self._need(rail, "voltage", device, "power")
        return multiply(i, v, name=rail_channel_name(rail, "power"), kind="power")

    def energy(self, rail: str = "Main", device: str | None = None) -> "DerivedChannel":
        """Cumulative energy in joules (trapezoidal integral of power).

        Unverified against Otii's own energy export.
        """
        from .derived import cumulative_energy

        return cumulative_energy(self.power(rail, device))

    # -- export -------------------------------------------------------------------------

    def to_parquet(self, path: str | Path) -> None:
        """Write all readable channels to one long-format Parquet file.

        Columns: ``channel``, ``device``, ``timestamp`` (s), ``value``. Each channel keeps
        its native sample rate.
        """
        from .export import recording_to_parquet

        recording_to_parquet(self, path)

    def to_dataframe(self) -> "pd.DataFrame":
        """All readable channels in long format.

        Columns: ``channel``, ``device``, ``timestamp`` (s), ``value``.
        """
        from ._pandas import recording_to_dataframe

        return recording_to_dataframe(self)

    def __repr__(self) -> str:
        return f"<Recording {self.name!r} start={self.start.isoformat()} channels={list(self.channels)}>"


class Project:
    """An opened Otii 3 project. Create with :func:`otiio.open`."""

    def __init__(self, raw: RawProject, path: Path) -> None:
        self.path = path
        self.root = raw.root
        self.format_version = raw.format_version
        self.version_id = raw.version_id
        self.raw = raw.project
        self.attributes = raw.attributes
        self.name: str = raw.project.get("name") or ""
        self.devices = NamedList(_build_devices(raw), "device")
        self.recordings = NamedList(_build_recordings(raw, self.devices), "recording")
        self.selections = NamedList(_build_selections(raw), "selection")

    def __repr__(self) -> str:
        return f"<Project {str(self.path)!r} format={self.format_version} recordings={self.recordings.names()}>"


def open(path: str | Path) -> Project:
    """Open an Otii 3 project from its ``.otii3`` file, project folder, or ``data/`` folder."""
    raw = read_project(path)
    if raw.format_version not in SUPPORTED_FORMATS:
        warnings.warn(
            f"project_format {raw.format_version} has not been verified with otiio "
            f"(known: {sorted(SUPPORTED_FORMATS)}); results may be wrong",
            UnsupportedFormatWarning,
            stacklevel=2,
        )
    return Project(raw, Path(path))


# -- builders -----------------------------------------------------------------------------


def _build_devices(raw: RawProject) -> list[Device]:
    serials: dict[str, str] = {}
    for entry in raw.project.get("assigned_devices", []):
        try:
            (serial, _name), device_id = entry
        except (TypeError, ValueError):
            continue
        serials[device_id] = serial
    devices = []
    for d in raw.project.get("devices", []):
        settings = d.get("settings") or {}
        devices.append(
            Device(
                id=d["id"],
                type=d.get("type", ""),
                name=settings.get("name") or d.get("type", ""),
                serial=serials.get(d["id"]),
                settings=settings,
                raw=d,
            )
        )
    return devices


def _build_recordings(raw: RawProject, devices: NamedList[Device]) -> list[Recording]:
    device_names = {d.id: d.name for d in devices}
    recordings = []
    for r in raw.project.get("recordings", []):
        channels = []
        for m in r.get("measurements", []):
            meas = m.get("measurement", {})
            src = meas.get("source", {})
            source_type = src.get("type", "")
            data_id = (m.get("data") or {}).get("id")
            path = raw.blob_dir(data_id) / "samples.dat" if data_id else None
            device_id = src.get("device")
            channels.append(
                Channel(
                    id=meas.get("id", ""),
                    name=src.get("name", ""),
                    kind=src.get("data_type", ""),
                    source_type=source_type,
                    rate=float(src.get("sample_rate") or 0) or 1.0,
                    offset_us=float(m.get("offset", 0)),
                    device=device_names.get(device_id, "") if device_id else "",
                    device_id=device_id,
                    path=path,
                    raw=m,
                )
            )
        attrs = raw.attributes.get(r["id"], {})
        recordings.append(
            Recording(
                id=r["id"],
                name=attrs.get("recording.name"),
                start_time_ms=int(r.get("start_time", 0)),
                offset_us=float(r.get("offset", 0)),
                channels=channels,
                attributes=attrs,
                raw=r,
            )
        )
    return recordings


def _build_selections(raw: RawProject) -> list[Selection]:
    ranges: dict[str, tuple[float, float]] = {}
    for state in raw.session_state:
        for group in state.get("group_state", []):
            data = group.get("data") or {}
            if "from" in data and "to" in data:
                ranges[group["id"]] = (float(data["from"]), float(data["to"]))

    selections = []
    for session in raw.sessions:
        for group in session.get("group_config", []):
            data = group.get("data") or {}
            if group.get("type") != "selection" or data.get("mode") != "range":
                continue  # skips the implicit whole-project selection
            client_id = data.get("client_id")
            span = ranges.get(group.get("id", ""))
            if not client_id or span is None:
                continue
            attrs = raw.attributes.get(client_id, {})
            selections.append(
                Selection(
                    id=client_id,
                    name=attrs.get("selection.name", client_id),
                    color=attrs.get("selection.color"),
                    start_us=span[0],
                    end_us=span[1],
                    visible=attrs.get("selection.visible", "true") != "false",
                    note=attrs.get("selection.note"),
                )
            )
    return selections
