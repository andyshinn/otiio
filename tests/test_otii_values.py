"""Values read off the Otii 3 app for the same projects, at the precision it displays."""

from pathlib import Path

import pytest

import otiio

from .conftest import DATA

TESTWH = DATA / "testwh"


@pytest.fixture
def testwh() -> Path:
    return TESTWH / "testwh.otii3"


def test_testwh_matches_otii_app(testwh: Path) -> None:
    # Otii app, whole recording, Main power: 1.76 mWh, min 53.4 mW, avg 200 mW, max 1.39 W.
    rec = otiio.open(testwh).recordings["TESTWH"]
    assert sorted(rec.channels) == ["mc", "mp", "mv"]
    assert all(ch.rate == 1_000 and len(ch) == 31_633 for ch in rec.channels.values())
    s = rec["mp"].stats()
    assert s.energy is not None
    assert round(s.energy / 3.6, 2) == 1.76  # J -> mWh
    assert round(s.min * 1e3, 1) == 53.4
    assert round(s.average * 1e3) == 200
    assert round(s.max, 2) == 1.39


# -- battest1: Ace on a battery at 1 kHz, -1 A / -0.2 A steps, selections STEP and FLAT ---

BATTEST1 = DATA / "battest1"


PREFIX = {"n": 1e-9, "µ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3}


def shown(text: str) -> tuple[float, float]:
    """Parse a value as the Otii app displays it, e.g. ``"-636 mA"`` or ``"53 µV"``.

    Returns the value in SI units and half a unit of its last displayed digit, the most
    a correctly rounded value can differ from it.
    """
    number, unit = text.split()
    scale = PREFIX[unit[0]] if len(unit) > 1 and unit[0] in PREFIX else 1.0
    decimals = len(number.split(".")[1]) if "." in number else 0
    return float(number) * scale, 0.5 * 10**-decimals * scale


def assert_shown(value: float | None, text: str) -> None:
    assert value is not None
    expected, tolerance = shown(text)
    assert abs(value - expected) <= tolerance * (1 + 1e-9), f"{value!r} does not display as {text!r}"


# Window -> channel -> (min, avg, max) as displayed in the Otii app's Selections panel.
BATTEST1_OTII: dict[str, dict[str, tuple[str, str, str]]] = {
    "STEP": {
        "mc": ("-1.00 A", "-636 mA", "-200 mA"),
        "mv": ("3.60 V", "3.71 V", "3.84 V"),
        "mp": ("-3.60 W", "-2.31 W", "-767 mW"),
    },
    "FLAT": {
        "mc": ("-200 mA", "-200 mA", "-200 mA"),
        "mv": ("3.83 V", "3.83 V", "3.83 V"),
        "mp": ("-767 mW", "-766 mW", "-766 mW"),
    },
    "PROJECT": {
        "mc": ("-1.00 A", "-599 mA", "67.3 µA"),
        "mv": ("53 µV", "3.71 V", "3.90 V"),
        "mp": ("-3.61 W", "-2.17 W", "29.9 µW"),
    },
    "VIEW": {
        "mc": ("-200 mA", "-200 mA", "67.3 µA"),
        "mv": ("69 µV", "3.82 V", "3.83 V"),
        "mp": ("-767 mW", "-765 mW", "6 nW"),
    },
}
# Energy as displayed, in µWh/mWh; converted to J by the factor 3.6 (1 mWh = 3.6 J).
BATTEST1_ENERGY = {"STEP": "-2.45 µWh", "FLAT": "-213 µWh", "PROJECT": "-7.26 mWh", "VIEW": "-388 µWh"}
BATTEST1_VIEW = (10.200426, 12.082335)  # the saved viewport, which runs past the end of the data


def _battest1_stats(window: str, channel: str) -> otiio.Stats:
    proj = otiio.open(BATTEST1)
    ch = proj.recordings[0][channel]
    if window == "PROJECT":
        return ch.stats()
    if window == "VIEW":
        return ch.stats(*BATTEST1_VIEW)
    return ch.stats(proj.selections[window])


@pytest.mark.parametrize("window", list(BATTEST1_OTII))
@pytest.mark.parametrize("channel", ["mc", "mv", "mp"])
def test_battest1_min_avg_max_match_otii(window: str, channel: str) -> None:
    s = _battest1_stats(window, channel)
    lo, avg, hi = BATTEST1_OTII[window][channel]
    assert_shown(s.min, lo)
    assert_shown(s.average, avg)
    assert_shown(s.max, hi)


@pytest.mark.parametrize("window", list(BATTEST1_ENERGY))
def test_battest1_energy_matches_otii(window: str) -> None:
    # Otii shows the same energy on the current and power panels.
    for channel in ("mc", "mp"):
        energy = _battest1_stats(window, channel).energy
        assert energy is not None
        assert_shown(energy / 3600, BATTEST1_ENERGY[window].replace("Wh", "W"))  # J -> Wh


def test_battest1_charge_matches_otii() -> None:
    # Otii's statistics popup for FLAT: C -55.6 µAh.
    charge = _battest1_stats("FLAT", "mc").charge
    assert charge is not None
    assert_shown(charge / 3600, "-55.6 µA")  # C -> Ah


def test_battest1_windows() -> None:
    proj = otiio.open(BATTEST1)
    rec = proj.recordings[0]
    assert (rec["mc"].rate, len(rec["mc"])) == (1_000, 12_026)
    step, flat = proj.selections["STEP"], proj.selections["FLAT"]
    assert step.note == "STEP down from 1A to 200mA for testing"
    # STEP starts between samples; the sample at 6.020 s (-0.2 A) overlaps it and sets max.
    s = rec["mc"].stats(step)
    assert (s.count, s.duration) == (5, pytest.approx(0.003818))
    assert rec["mc"].stats(flat).count == 1_002  # partial samples at both ends
    # A window past the end of the data is clipped to the last sample's interval.
    assert rec["mp"].stats(*BATTEST1_VIEW).end == pytest.approx(12.026)
