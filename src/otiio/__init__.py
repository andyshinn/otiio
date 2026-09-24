"""Read Otii 3 project files. Unofficial; not affiliated with Qoitech."""

from importlib.metadata import PackageNotFoundError, version

from .derived import DerivedChannel
from .errors import (
    AlignmentError,
    AmbiguousNameError,
    MissingDataError,
    OtiioError,
    ProjectNotFoundError,
    UnsupportedFormatWarning,
)
from .model import Channel, ChannelMap, Device, NamedList, Project, Recording, Selection, open
from .signal import Block, Signal
from .stats import Stats

try:
    __version__ = version("otiio")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "AlignmentError",
    "AmbiguousNameError",
    "Block",
    "Channel",
    "ChannelMap",
    "DerivedChannel",
    "Device",
    "MissingDataError",
    "NamedList",
    "OtiioError",
    "Project",
    "ProjectNotFoundError",
    "Recording",
    "Selection",
    "Signal",
    "Stats",
    "UnsupportedFormatWarning",
    "__version__",
    "open",
]
