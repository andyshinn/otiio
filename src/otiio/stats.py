"""Streaming statistics over a channel."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .signal import Signal


@dataclass(frozen=True)
class Stats:
    """Summary of a channel window, computed the way the Otii app computes it.

    Each sample holds its value until the next one. ``average``/``rms`` are time-weighted
    over ``[start, end)``, with partially covered edge samples weighted by their overlap;
    ``min``/``max`` come from every sample that overlaps the window. ``energy`` (J) is set
    for power channels, and for a rail's current channel when its power is stored or can
    be derived. ``charge`` (C) is set for current channels. Both are the integral of the
    held values (a rectangle sum), which matches Otii's displayed E and C.
    """

    count: int
    start: float
    end: float
    min: float
    max: float
    average: float
    rms: float
    unit: str
    energy: float | None = None
    charge: float | None = None

    @property
    def mean(self) -> float:
        return self.average

    @property
    def duration(self) -> float:
        """Covered time in seconds, like the duration Otii shows for a selection."""
        return self.end - self.start


def compute_stats(sig: "Signal", *, chunk: int) -> Stats:
    """One chunked pass in float64."""
    n = len(sig)
    lo, hi = math.inf, -math.inf
    total = sq = 0.0
    first = last = 0.0
    for i, block in enumerate(sig.chunks(chunk)):
        v = np.asarray(block.v, dtype=np.float64)
        if i == 0:
            first = float(v[0])
        last = float(v[-1])
        lo = min(lo, float(v.min()))
        hi = max(hi, float(v.max()))
        total += float(v.sum())
        sq += float(np.dot(v, v))

    # Weight the edge samples by the fraction of their interval inside the window.
    w_first, w_last = sig.edge_weights()
    if n == 1:
        weight, total, sq = w_first, w_first * first, w_first * first * first
    else:
        weight = n - (1 - w_first) - (1 - w_last)
        total -= (1 - w_first) * first + (1 - w_last) * last
        sq -= (1 - w_first) * first * first + (1 - w_last) * last * last

    if n == 0 or weight <= 0:
        nan = math.nan
        return Stats(n, sig.span_start, sig.span_end, lo if n else nan, hi if n else nan, nan, nan, sig.unit)

    integral = total / sig.rate
    energy = integral if sig.kind == "power" else None
    if sig.kind == "current":
        power = sig._matching_power()
        if power is not None:
            # Same time span as the current window; power may be at a different rate.
            energy = compute_stats(power.slice(sig.span_start, sig.span_end), chunk=chunk).energy
    return Stats(
        count=n,
        start=sig.span_start,
        end=sig.span_end,
        min=lo,
        max=hi,
        average=total / weight,
        rms=math.sqrt(max(sq, 0.0) / weight),
        unit=sig.unit,
        energy=energy,
        charge=integral if sig.kind == "current" else None,
    )
