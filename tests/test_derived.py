from pathlib import Path
from typing import Any

import numpy as np
import pytest

import otiio
from otiio.derived import cumulative_energy, divide, multiply

from .conftest import TESTDATA1, make_signal


def test_power_prefers_stored_channel(testdata1: Path) -> None:
    rec = otiio.open(testdata1).recordings["TEST2"]
    p = rec.power()
    assert p is rec["mp"]
    assert p.derived is False
    assert rec.current() is rec["mc"]
    assert rec.voltage() is rec["mv"]


def _drop(name: str) -> Any:
    def edit(project: dict[str, Any]) -> None:
        for rec in project["recordings"]:
            rec["measurements"] = [m for m in rec["measurements"] if m["measurement"]["source"]["name"] != name]

    return edit


def test_power_derived_when_not_stored(testdata1: Path, patched: Any) -> None:
    stored = otiio.open(testdata1).recordings["TEST2"]["mp"]
    rec = otiio.open(patched(TESTDATA1, _drop("mp"))).recordings["TEST2"]
    assert "mp" not in rec.channels
    p = rec.power()
    assert p.derived is True
    assert (p.name, p.kind, p.unit, p.label, p.rail) == (
        "mp",
        "power",
        "W",
        "Main power - Ace",
        "Main",
    )
    assert len(p) == len(stored)
    np.testing.assert_allclose(p.values, stored.values, rtol=1e-6, atol=1e-15)
    assert p.stats().average == pytest.approx(stored.stats().average, rel=1e-6)


def test_current_stats_energy_from_derived_power(testdata1: Path, patched: Any) -> None:
    stored = otiio.open(testdata1).recordings["TEST2"]["mc"].stats()
    rec = otiio.open(patched(TESTDATA1, _drop("mp"))).recordings["TEST2"]
    assert rec["mc"].stats().energy == pytest.approx(stored.energy, rel=1e-6)


def test_current_stats_without_power_or_voltage(patched: Any) -> None:
    def drop_both(project: dict[str, Any]) -> None:
        _drop("mp")(project)
        _drop("mv")(project)

    s = otiio.open(patched(TESTDATA1, drop_both)).recordings["TEST2"]["mc"].stats()
    assert s.energy is None and s.charge is not None


def test_current_derived_when_not_stored(patched: Any) -> None:
    rec = otiio.open(patched(TESTDATA1, _drop("mc"))).recordings["TEST2"]
    ref = otiio.open(TESTDATA1).recordings["TEST2"]["mc"].values.astype(np.float64)
    i = rec.current()
    assert i.derived and i.kind == "current"
    # mv is never 0 in this fixture, so I = P / V is finite everywhere.
    np.testing.assert_allclose(i.values, ref, rtol=1e-6, atol=1e-12)


def test_underivable_quantity(patched: Any) -> None:
    rec = otiio.open(patched(TESTDATA1, _drop("mp"))).recordings["TEST2"]
    with pytest.raises(KeyError, match="ADC"):
        rec.power("ADC")  # only av is stored on the ADC rail
    with pytest.raises(ValueError):
        rec.power("Nope")


def test_multiply_and_divide() -> None:
    a = make_signal([1.0, 2.0, 3.0], kind="current", name="mc")
    b = make_signal([2.0, 0.0, 4.0], kind="voltage", name="mv")
    np.testing.assert_array_equal(multiply(a, b, name="mp", kind="power").values, [2, 0, 12])
    q = divide(a, b, name="x", kind="current").values
    assert q[0] == 0.5 and np.isnan(q[1]) and q[2] == 0.75


def test_mixed_rate_combination_block_averages_faster_input() -> None:
    fast = make_signal(np.arange(10.0), rate=10.0, kind="voltage", name="mv")
    slow = make_signal([1.0, 2.0, 3.0], rate=2.0, kind="current", name="mc")
    p = multiply(slow, fast, name="mp", kind="power")
    assert p.rate == 2.0
    # mv is block-averaged by 5 to (10 - 1) // 5 = 1 sample (mean of 0..4 = 2.0).
    assert len(p) == 1
    np.testing.assert_allclose(p.values, [1.0 * 2.0])


def test_alignment_errors() -> None:
    a = make_signal(np.ones(10), rate=10.0)
    with pytest.raises(otiio.AlignmentError, match="non-integer"):
        multiply(a, make_signal(np.ones(10), rate=3.0), name="x", kind="power")
    with pytest.raises(otiio.AlignmentError, match="starts at"):
        multiply(a, make_signal(np.ones(10), rate=10.0, offset_us=5), name="x", kind="power")


def test_downsample_block_mean_drops_final_block() -> None:
    sig = make_signal(np.arange(12.0), rate=6.0, offset_us=1e6)
    ds = sig.downsample(3)
    # Otii yields (n - 1) // factor samples: the last block is dropped even when complete.
    np.testing.assert_allclose(ds.values, [1.0, 4.0, 7.0])
    assert (ds.rate, ds.start, ds.derived) == (2.0, 1.0, True)
    assert len(make_signal(np.arange(13.0)).downsample(3)) == 4
    with pytest.raises(ValueError):
        sig.downsample(0)


def test_downsample_sense_to_vbus_rate(testdata2: Path) -> None:
    rec = otiio.open(testdata2).recordings["TEST4"]
    ds = rec["sn"].downsample(5)
    assert ds.rate == rec["vb"].rate
    assert len(ds) == (6_710 - 1) // 5
    np.testing.assert_allclose(ds.values, rec["sn"].values[: len(ds) * 5].reshape(-1, 5).mean(axis=1), rtol=1e-6)


def test_cumulative_energy() -> None:
    values = np.array([1.0, 3.0, 2.0, 2.0, 5.0])
    power = make_signal(values, rate=2.0)
    e = cumulative_energy(power)
    # Energy used before each sample: each sample holds its value for 0.5 s.
    expected = [0.0, 0.5, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(e.values, expected)
    # Chunked, out-of-order and windowed reads agree with the full array.
    np.testing.assert_allclose(np.concatenate([b.v for b in e.chunks(2)]), expected)
    np.testing.assert_allclose(e.slice(1.0).values, expected[2:])
    np.testing.assert_allclose(e.slice(0.5, 1.5).values, expected[1:3])
    assert (e.kind, e.unit) == ("energy", "J")


def test_recording_energy(testdata1: Path) -> None:
    rec = otiio.open(testdata1).recordings["TEST1"]
    e = rec.energy()
    assert (e.name, e.label, e.rail) == ("me", "Main energy - Ace", "Main")
    mp = rec["mp"]
    total = float(e.values[-1]) + float(mp.values[-1]) / mp.rate
    assert total == pytest.approx(mp.stats().energy, rel=1e-9, abs=1e-18)
    assert e.values[0] == 0.0
