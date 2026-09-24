import math
from pathlib import Path

import numpy as np
import pytest

import otiio

from .conftest import make_signal


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 1000])
def test_stats_match_numpy(chunk: int) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(2.0, 0.5, 101)
    sig = make_signal(values, rate=50.0, kind="power")
    s = sig.stats(chunk=chunk)
    assert s.count == 101
    assert s.min == pytest.approx(values.min())
    assert s.max == pytest.approx(values.max())
    assert s.average == pytest.approx(values.mean())
    assert s.mean == s.average
    assert s.rms == pytest.approx(math.sqrt(np.mean(values**2)))
    # Each sample holds for 1 / rate, so energy is a rectangle sum over len / rate seconds.
    assert s.energy == pytest.approx(values.sum() / 50)
    assert s.duration == pytest.approx(101 / 50)
    assert s.charge is None
    assert s.unit == "W"


def test_charge_for_current_only() -> None:
    cur = make_signal([1.0, 1.0, 1.0], rate=1.0, kind="current", name="mc").stats()
    assert cur.charge == pytest.approx(3.0)  # 3 samples of 1 A, each held for 1 s
    assert cur.energy is None
    volt = make_signal([1.0, 2.0], kind="voltage", name="mv").stats()
    assert volt.energy is None and volt.charge is None


def test_stats_window() -> None:
    sig = make_signal(np.arange(10.0), rate=1.0)
    s = sig.stats(2, 5)
    assert (s.count, s.min, s.max, s.start, s.end, s.duration) == (3, 2.0, 4.0, 2.0, 5.0, 3.0)
    assert s.energy == pytest.approx(2 + 3 + 4)


def test_stats_weight_partial_edge_samples() -> None:
    """Otii integrates the held values over the exact window, edges included pro rata."""
    sig = make_signal([0.0, 10.0, 20.0, 30.0], rate=1.0)  # sample k covers [k, k + 1)
    s = sig.stats(0.5, 2.25)
    assert s.count == 3  # samples 0, 1 and 2 overlap the window
    assert (s.min, s.max) == (0.0, 20.0)
    assert s.energy == pytest.approx(0.5 * 0 + 1.0 * 10 + 0.25 * 20)
    assert s.average == pytest.approx(s.energy / 1.75)
    assert s.rms == pytest.approx(math.sqrt((1.0 * 100 + 0.25 * 400) / 1.75))
    one = sig.stats(1.2, 1.7)  # inside a single sample
    assert (one.count, one.average, one.energy) == (1, 10.0, pytest.approx(5.0))


def test_window_clipped_to_data() -> None:
    s = make_signal([1.0, 1.0], rate=1.0).stats(-5, 50)
    assert (s.start, s.end, s.energy) == (0.0, 2.0, 2.0)


def test_empty_stats() -> None:
    s = make_signal(np.arange(10.0), rate=1.0).stats(20, 30)
    assert s.count == 0
    assert math.isnan(s.average)


def test_single_sample_holds_for_one_interval() -> None:
    assert make_signal([5.0], rate=10.0).stats().energy == pytest.approx(0.5)


def test_fixture_stats_match_numpy(testdata1: Path) -> None:
    rec = otiio.open(testdata1).recordings["TEST1"]
    mp = rec["mp"]
    v = mp.values.astype(np.float64)
    s = mp.stats(chunk=10_000)
    assert s.average == pytest.approx(v.mean(), rel=1e-9)
    assert s.energy == pytest.approx(v.sum() / 50_000, rel=1e-9, abs=1e-18)
    assert rec["mc"].stats().charge == pytest.approx(rec["mc"].values.astype(np.float64).sum() / 50_000, rel=1e-9)


def test_current_stats_include_energy_from_power(testdata1: Path) -> None:
    rec = otiio.open(testdata1).recordings["TEST1"]
    assert rec["mc"].stats().energy == pytest.approx(rec["mp"].stats().energy, rel=1e-12)
    window = rec["mc"].stats(0.01, 0.02)
    assert window.energy == pytest.approx(rec["mp"].stats(0.01, 0.02).energy, rel=1e-12)
    assert window.charge is not None


def test_non_rail_current_has_no_energy() -> None:
    assert make_signal([1.0, 2.0], kind="current", name="mc").stats().energy is None
