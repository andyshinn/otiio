"""Base class shared by stored and derived channels."""

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

    from .derived import DerivedChannel
    from .model import Selection
    from .stats import Stats

FloatArray = npt.NDArray[np.floating[Any]]

DEFAULT_CHUNK = 1_000_000


class Block(NamedTuple):
    """A contiguous run of samples: times in seconds and values in SI units."""

    t: npt.NDArray[np.float64]
    v: FloatArray


class Signal(ABC):
    """A uniformly sampled series of values, possibly a window of a longer one.

    Sample ``i`` of the window sits at ``start + i / rate`` seconds on the project timeline.
    As in Otii, each sample holds its value until the next one, so it covers
    ``[t, t + 1 / rate)``. A window (:meth:`slice`) covers an exact time range
    ``[span_start, span_end)`` and includes every sample overlapping it; :meth:`stats`
    weights the first and last samples by how much of them lies inside.
    """

    id: str
    name: str
    kind: str
    unit: str
    label_base: str
    rail: str | None
    device: str
    rate: float
    offset_us: float
    derived: bool

    _lo: int = 0
    _hi: int | None = None
    _t0: float | None = None  # exact window bounds in seconds; None = sample edges
    _t1: float | None = None

    # -- implemented by subclasses ------------------------------------------------------

    @abstractmethod
    def _base_len(self) -> int:
        """Number of samples in the full (unwindowed) series."""

    @abstractmethod
    def _read_base(self, i0: int, i1: int) -> FloatArray:
        """Samples ``[i0, i1)`` of the full series."""

    def _matching_power(self) -> "Signal | None":
        """Power on this current channel's rail, used for energy in :meth:`stats`."""
        return None

    # -- geometry -----------------------------------------------------------------------

    def __len__(self) -> int:
        hi = self._base_len() if self._hi is None else self._hi
        return max(hi - self._lo, 0)

    @property
    def label(self) -> str:
        """Otii-style name, e.g. ``"Main current - Ace"``."""
        return f"{self.label_base} - {self.device}" if self.device else self.label_base

    @property
    def start(self) -> float:
        """Time of the first sample, in seconds."""
        return self.offset_us / 1e6 + self._lo / self.rate

    @property
    def end(self) -> float:
        """Time of the last sample, in seconds (equal to ``start`` when empty)."""
        return self.start + max(len(self) - 1, 0) / self.rate

    @property
    def span_start(self) -> float:
        """Start of the time range this window covers, in seconds."""
        return self.start if self._t0 is None else self._t0

    @property
    def span_end(self) -> float:
        """End of the time range this window covers: the last sample's time plus ``1 / rate``
        for a whole channel, or the exact end of a slice."""
        return self.start + len(self) / self.rate if self._t1 is None else self._t1

    @property
    def duration(self) -> float:
        """Covered time, ``span_end - span_start`` seconds (``len / rate`` for a whole channel)."""
        return self.span_end - self.span_start

    def edge_weights(self) -> tuple[float, float]:
        """Fraction (0..1) of the first and last sample's interval that lies inside the window."""
        n = len(self)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            w = min(max((self.span_end - self.span_start) * self.rate, 0.0), 1.0)
            return w, w
        first = (self.start + 1 / self.rate - self.span_start) * self.rate
        last = (self.span_end - self.end) * self.rate
        return min(max(first, 0.0), 1.0), min(max(last, 0.0), 1.0)

    # -- data access --------------------------------------------------------------------

    @property
    def values(self) -> FloatArray:
        """All samples in the window.

        Zero-copy ``np.memmap`` for stored channels. Derived channels compute the full
        array, which can be large; prefer :meth:`chunks` for long recordings.
        """
        return self._read(0, len(self))

    def _read(self, i0: int, i1: int) -> FloatArray:
        return self._read_base(self._lo + i0, self._lo + i1)

    def times(self, i0: int = 0, i1: int | None = None) -> npt.NDArray[np.float64]:
        """Sample times in seconds for window indices ``[i0, i1)``."""
        i1 = len(self) if i1 is None else i1
        return self.offset_us / 1e6 + np.arange(self._lo + i0, self._lo + i1) / self.rate

    def chunks(self, n: int = DEFAULT_CHUNK) -> Iterator[Block]:
        """Iterate the window in blocks of at most ``n`` samples."""
        if n <= 0:
            raise ValueError("chunk size must be positive")
        total = len(self)
        for i0 in range(0, total, n):
            i1 = min(i0 + n, total)
            yield Block(self.times(i0, i1), self._read(i0, i1))

    def index_at(self, t: float) -> int:
        """Window index of the first sample at or after time ``t`` (seconds), clamped."""
        pos = (t - self.start) * self.rate
        # Round off float noise so a time exactly on a sample maps to that sample.
        i = math.ceil(round(pos, 6))
        return min(max(i, 0), len(self))

    def slice(self, t0: "float | Selection | None" = None, t1: float | None = None) -> Self:
        """The time range ``[t0, t1)`` (seconds, project timeline), as a view.

        Like Otii, the view includes every sample whose interval ``[t, t + 1 / rate)``
        overlaps the range, so a sample just before ``t0`` is included when ``t0`` falls
        between samples. :meth:`stats` counts those edge samples only for the part inside.
        The range is clipped to this signal's own span. ``t0`` may be a
        :class:`~otiio.Selection`; ``None`` means an open end.
        """
        from .model import Selection

        if isinstance(t0, Selection):
            if t1 is not None:
                raise TypeError("pass either a Selection or t0/t1, not both")
            sel = t0
            t0, t1 = sel.start, sel.end
        lo, hi = self.span_start, self.span_end
        t0 = lo if t0 is None else min(max(t0, lo), hi)
        t1 = hi if t1 is None else min(max(t1, t0), hi)
        # Round off float noise so a time exactly on a sample boundary maps onto it.
        i0 = min(max(math.floor(round((t0 - self.start) * self.rate, 6)), 0), len(self))
        i1 = min(max(math.ceil(round((t1 - self.start) * self.rate, 6)), i0), len(self))
        if t1 <= t0:
            i1 = i0
        view = self._window(i0, i1)
        view._t0, view._t1 = t0, t1
        return view

    def _window(self, i0: int, i1: int) -> Self:
        view = copy.copy(self)
        view._lo = self._lo + i0
        view._hi = self._lo + i1
        view._t0 = view._t1 = None
        return view

    # -- analysis -----------------------------------------------------------------------

    def stats(
        self,
        t0: "float | Selection | None" = None,
        t1: float | None = None,
        *,
        chunk: int = DEFAULT_CHUNK,
    ) -> "Stats":
        """Min/max/average/RMS over the window, plus energy and/or charge.

        Power channels get ``energy``. Current channels get ``charge``, and also ``energy``
        from the rail's power over the same time span when the recording has one, as
        Otii reports it.
        """
        from .stats import compute_stats

        target = self if t0 is None and t1 is None else self.slice(t0, t1)
        return compute_stats(target, chunk=chunk)

    def downsample(self, factor: int) -> "DerivedChannel":
        """Average non-overlapping blocks of ``factor`` samples, as Otii does.

        Like Otii, the final block is dropped, giving ``(len - 1) // factor`` samples.
        Works on whole samples: partial edge weights of a slice are ignored.
        """
        from .derived import downsample

        return downsample(self, factor)

    # -- export -------------------------------------------------------------------------

    def to_csv(self, path: str | Path, *, chunk: int = DEFAULT_CHUNK) -> Path:
        """Write CSV in Otii's export format, streaming in chunks; returns the file written.

        If ``path`` is a directory the file is named ``<label>.csv``, as Otii names it.
        """
        from .export import signal_to_csv

        return signal_to_csv(self, path, chunk=chunk)

    def to_parquet(self, path: str | Path, *, chunk: int = DEFAULT_CHUNK) -> None:
        """Write ``timestamp``/``value`` columns to Parquet (needs the ``pandas`` extra)."""
        from .export import signal_to_parquet

        signal_to_parquet(self, path, chunk=chunk)

    def to_dataframe(self) -> "pd.DataFrame":
        """A DataFrame indexed by ``timestamp`` (s) with one column named after the channel."""
        from ._pandas import signal_to_dataframe

        return signal_to_dataframe(self)

    def __repr__(self) -> str:
        kind = "derived " if self.derived else ""
        return (
            f"<{type(self).__name__} {self.name!r} {kind}{self.kind} ({self.unit}) "
            f"device={self.device!r} rate={self.rate:g} n={len(self)} "
            f"t={self.start:g}..{self.end:g}s>"
        )
