from pathlib import Path

import numpy as np
import pytest

import otiio

from .conftest import make_signal


def test_times_follow_offset_and_rate() -> None:
    sig = make_signal(np.arange(5), rate=4.0, offset_us=2_500_000)
    np.testing.assert_allclose(sig.times(), [2.5, 2.75, 3.0, 3.25, 3.5])
    assert (sig.start, sig.end, sig.duration) == (2.5, 3.5, 1.25)


@pytest.mark.parametrize("n", [1, 3, 7, 10, 11, 1000])
def test_chunks_cover_everything_once(n: int) -> None:
    values = np.arange(10, dtype=np.float64)
    sig = make_signal(values, rate=2.0)
    blocks = list(sig.chunks(n))
    assert all(len(b.t) == len(b.v) <= n for b in blocks)
    np.testing.assert_array_equal(np.concatenate([b.v for b in blocks]), values)
    np.testing.assert_allclose(np.concatenate([b.t for b in blocks]), values / 2)


def test_chunks_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        list(make_signal([1.0]).chunks(0))


def test_slice_includes_overlapping_samples_and_clamps() -> None:
    sig = make_signal(np.arange(10), rate=10.0)  # sample k covers [k / 10, (k + 1) / 10)
    np.testing.assert_array_equal(sig.slice(0.2, 0.5).values, [2, 3, 4])
    # 0.25 falls inside sample 2's interval, so sample 2 is included (as in Otii).
    np.testing.assert_array_equal(sig.slice(0.25, 0.5).values, [2, 3, 4])
    np.testing.assert_array_equal(sig.slice(0.2, 0.51).values, [2, 3, 4, 5])
    np.testing.assert_array_equal(sig.slice(None, 0.2).values, [0, 1])
    np.testing.assert_array_equal(sig.slice(0.8).values, [8, 9])
    assert len(sig.slice(-5, 100)) == 10
    assert len(sig.slice(0.5, 0.2)) == 0
    assert len(sig.slice(5, 6)) == 0
    part = sig.slice(0.25, 0.51)
    assert (part.span_start, part.span_end) == (0.25, 0.51)
    assert part.duration == pytest.approx(0.26)
    assert part.edge_weights() == (pytest.approx(0.5), pytest.approx(0.1))
    whole = sig.slice(-5, 100)
    assert (whole.span_start, whole.span_end) == (0.0, 1.0)


def test_slice_of_slice_keeps_timeline() -> None:
    sig = make_signal(np.arange(10), rate=10.0, offset_us=1_000_000)
    inner = sig.slice(1.2, 1.8).slice(1.4, 1.6)
    assert sig.slice(1.2, 1.8).slice(1.0, 2.0).span_start == pytest.approx(1.2)  # clipped to parent
    np.testing.assert_array_equal(inner.values, [4, 5])
    np.testing.assert_allclose(inner.times(), [1.4, 1.5])
    assert inner.start == pytest.approx(1.4)


def test_slice_float_noise_lands_on_sample() -> None:
    sig = make_signal(np.arange(100_000), rate=50_000.0)
    # 0.1 * 50000 is 5000.000000000001 in floating point.
    assert sig.slice(0.1).values[0] == 5_000


def test_slice_selection_and_both_args_error(testdata2: Path) -> None:
    proj = otiio.open(testdata2)
    sel = proj.selections[0]
    with pytest.raises(TypeError):
        proj.recordings[0]["sn"].slice(sel, 1.0)  # type: ignore[arg-type]


def test_stored_slice_is_zero_copy(testdata2: Path) -> None:
    ch = otiio.open(testdata2).recordings[0]["sn"]
    part = ch.slice(0.1, 0.2)
    assert isinstance(part.values, np.memmap)
    assert len(part) == 500
    np.testing.assert_array_equal(part.values, ch.values[500:1000])
    np.testing.assert_allclose(part.times()[[0, -1]], [0.1, 0.1998])
