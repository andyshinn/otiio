"""Channels computed from stored ones: downsampling, products/quotients, energy."""

from collections.abc import Callable

import numpy as np

from .channels import channel_info, unit_for
from .errors import AlignmentError
from .signal import FloatArray, Signal

Reader = Callable[[int, int], FloatArray]


class DerivedChannel(Signal):
    """A channel whose samples are computed on demand, chunk by chunk."""

    derived = True

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        rate: float,
        offset_us: float,
        device: str,
        length: int,
        reader: Reader,
        label_base: str | None = None,
        rail: str | None = None,
        id: str = "",
    ) -> None:
        info = channel_info(name, kind)
        self.id = id
        self.name = name
        self.kind = kind
        self.unit = unit_for(kind)
        self.label_base = label_base if label_base is not None else info.label
        self.rail = rail if rail is not None else info.rail
        self.rate = rate
        self.offset_us = offset_us
        self.device = device
        self._length = length
        self._reader = reader

    def _base_len(self) -> int:
        return self._length

    def _read_base(self, i0: int, i1: int) -> FloatArray:
        i0, i1 = max(i0, 0), min(i1, self._length)
        if i1 <= i0:
            return np.empty(0, dtype=np.float64)
        return self._reader(i0, i1)


def downsample(sig: Signal, factor: int) -> DerivedChannel:
    """Block-mean ``factor`` samples into one, as Otii's own downsampling does.

    Verified to ~1e-8 against projects downsampled by 500 in Otii. Like Otii, this yields
    ``(n - 1) // factor`` samples: the final block is always dropped, even when it is
    complete. The first output sample sits at the window's start time.
    """
    if int(factor) != factor or factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    factor = int(factor)

    def read(i0: int, i1: int) -> FloatArray:
        block = np.asarray(sig._read(i0 * factor, i1 * factor), dtype=np.float64)
        return block.reshape(-1, factor).mean(axis=1)

    return DerivedChannel(
        name=sig.name,
        kind=sig.kind,
        rate=sig.rate / factor,
        offset_us=sig.start * 1e6,
        device=sig.device,
        length=max(len(sig) - 1, 0) // factor,
        reader=read,
        label_base=sig.label_base,
        rail=sig.rail,
        id=sig.id,
    )


def align(a: Signal, b: Signal) -> tuple[Signal, Signal]:
    """Bring two signals to a common (slower) rate so they can be combined per sample."""
    if not np.isclose(a.start, b.start, rtol=0, atol=1e-9):
        raise AlignmentError(
            f"{a.name!r} starts at {a.start}s but {b.name!r} at {b.start}s; slice them to a common start first"
        )
    rate = min(a.rate, b.rate)

    def to_rate(s: Signal) -> Signal:
        ratio = s.rate / rate
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise AlignmentError(f"cannot align {s.name!r} at {s.rate:g} sps to {rate:g} sps (non-integer ratio)")
        return s if round(ratio) == 1 else downsample(s, round(ratio))

    return to_rate(a), to_rate(b)


def _combine(
    a: Signal,
    b: Signal,
    op: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    name: str,
    kind: str,
) -> DerivedChannel:
    a, b = align(a, b)

    def read(i0: int, i1: int) -> FloatArray:
        x = np.asarray(a._read(i0, i1), dtype=np.float64)
        y = np.asarray(b._read(i0, i1), dtype=np.float64)
        return op(x, y)

    return DerivedChannel(
        name=name,
        kind=kind,
        rate=a.rate,
        offset_us=a.start * 1e6,
        device=a.device,
        length=min(len(a), len(b)),
        reader=read,
    )


def _safe_divide(x: FloatArray, y: FloatArray) -> FloatArray:
    out = np.full(x.shape, np.nan)
    np.divide(x, y, out=out, where=y != 0)
    return out


def multiply(a: Signal, b: Signal, *, name: str, kind: str) -> DerivedChannel:
    """``a * b`` per sample (e.g. current x voltage = power)."""
    return _combine(a, b, np.multiply, name=name, kind=kind)


def divide(a: Signal, b: Signal, *, name: str, kind: str) -> DerivedChannel:
    """``a / b`` per sample; division by zero gives NaN."""
    return _combine(a, b, _safe_divide, name=name, kind=kind)


def cumulative_energy(power: Signal) -> DerivedChannel:
    """Running integral of power (W) in joules: energy used before each sample's time.

    Like Otii's statistics, each sample holds its value until the next one, so
    ``E[k] = sum(p[:k]) / rate`` and ``E[0] == 0``. The total over the whole channel,
    ``E[-1] + p[-1] / rate``, equals ``power.stats().energy``. Sequential chunk reads only
    carry the prefix sum from the previous chunk.
    """
    n = len(power)
    cache = {"i": 0, "prefix": 0.0}  # prefix = sum(p[0:i])

    def prefix_to(i: int) -> float:
        if cache["i"] != i:
            start, total = (int(cache["i"]), cache["prefix"]) if cache["i"] < i else (0, 0.0)
            step = 1_000_000
            for j in range(start, i, step):
                total += float(np.sum(power._read(j, min(j + step, i)), dtype=np.float64))
            cache["i"], cache["prefix"] = i, total
        return cache["prefix"]

    def read(i0: int, i1: int) -> FloatArray:
        p = np.asarray(power._read(i0, i1), dtype=np.float64)
        s = prefix_to(i0) + np.cumsum(p)
        cache["i"], cache["prefix"] = i1, float(s[-1])
        return (s - p) / power.rate

    rail = power.rail
    name = f"{power.name[:-1]}e" if power.name.endswith("p") and rail else f"{power.name}_energy"
    return DerivedChannel(
        name=name,
        kind="energy",
        rate=power.rate,
        offset_us=power.start * 1e6,
        device=power.device,
        length=n,
        reader=read,
        label_base=f"{rail} energy" if rail else f"{power.label_base} energy",
        rail=rail,
    )
