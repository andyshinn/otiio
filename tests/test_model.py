import datetime as dt
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import otiio

from .conftest import TESTDATA1, TESTDATA2

# -- testdata1: Ace main + ADC channels at 50 kHz ------------------------------------------

MEANS_1 = {
    "TEST2": {
        "mc": -6.014825490114962e-08,
        "mp": -2.6501784605322384e-10,
        "mv": 0.004405629746750044,
        "av": -0.05129197927951573,
    },
    "TEST1": {
        "mc": -6.04444366524889e-08,
        "mp": -2.6679920019555646e-10,
        "mv": 0.004412560521608271,
        "av": -0.051296785547362896,
    },
}


def test_testdata1_recordings(testdata1: Path) -> None:
    proj = otiio.open(testdata1)
    assert proj.format_version == 17
    assert proj.recordings.names() == ["TEST2", "TEST1"]  # project.json order
    assert [r.start_time_ms for r in proj.recordings] == [1790212546942, 1790212553582]
    assert proj.recordings[0].start == dt.datetime(2026, 9, 24, 1, 15, 46, 942000, dt.UTC)
    assert [d.type for d in proj.devices] == ["Uart", "Ace"]
    assert proj.devices["Ace"].settings["mainvolt"] == 4.0


@pytest.mark.parametrize("rec_name", ["TEST2", "TEST1"])
def test_testdata1_channels(testdata1: Path, rec_name: str) -> None:
    rec = otiio.open(testdata1).recordings[rec_name]
    assert sorted(rec.channels) == ["av", "mc", "mp", "mv"]
    assert rec.duration == pytest.approx(2.26032)
    for name, mean in MEANS_1[rec_name].items():
        ch = rec[name]
        assert len(ch) == 113_016
        assert ch.rate == 50_000
        assert ch.offset_us == 0
        assert ch.device == "Ace"
        assert ch.values.dtype == np.float32
        assert isinstance(ch.values, np.memmap)
        assert float(np.mean(ch.values, dtype=np.float64)) == pytest.approx(mean, rel=1e-6)


def test_testdata1_labels_and_rails(testdata1: Path) -> None:
    rec = otiio.open(testdata1).recordings["TEST2"]
    got = {n: (rec[n].label, rec[n].rail, rec[n].kind, rec[n].unit) for n in rec.channels}
    assert got == {
        "mc": ("Main current - Ace", "Main", "current", "A"),
        "mv": ("Main voltage - Ace", "Main", "voltage", "V"),
        "mp": ("Main power - Ace", "Main", "power", "W"),
        "av": ("ADC voltage - Ace", "ADC", "voltage", "V"),
    }


def test_testdata1_has_no_named_selections(testdata1: Path) -> None:
    # Only the implicit whole-project selection and a viewport exist.
    assert otiio.open(testdata1).selections == []


def test_stored_power_matches_current_times_voltage(testdata1: Path) -> None:
    rec = otiio.open(testdata1).recordings["TEST2"]
    product = rec["mc"].values.astype(np.float64) * rec["mv"].values
    np.testing.assert_allclose(rec["mp"].values, product, rtol=1e-7, atol=1e-15)


# -- testdata2: sense/VBUS channels at mixed rates, one named selection --------------------

LENGTHS_2 = {"TEST4": (6_710, 1_342, 1.342), "TEST3": (7_360, 1_472, 1.472)}


@pytest.mark.parametrize("rec_name", ["TEST4", "TEST3"])
def test_testdata2_mixed_rates(testdata2: Path, rec_name: str) -> None:
    rec = otiio.open(testdata2).recordings[rec_name]
    n_fast, n_vb, duration = LENGTHS_2[rec_name]
    assert sorted(rec.channels) == ["sn", "sp", "vb"]
    assert (rec["sn"].rate, len(rec["sn"])) == (5_000, n_fast)
    assert (rec["sp"].rate, len(rec["sp"])) == (5_000, n_fast)
    assert (rec["vb"].rate, len(rec["vb"])) == (1_000, n_vb)
    assert rec.duration == pytest.approx(duration)
    assert rec["sn"].duration == pytest.approx(duration)
    assert rec["vb"].end == pytest.approx((n_vb - 1) / 1000)


def test_testdata2_labels(testdata2: Path) -> None:
    rec = otiio.open(testdata2).recordings["TEST4"]
    assert {n: (rec[n].label, rec[n].rail) for n in rec.channels} == {
        "sn": ("Sense- voltage - Ace", None),
        "sp": ("Sense+ voltage - Ace", None),
        "vb": ("VBUS - Ace", None),
    }
    assert rec["vb"].stats().average == pytest.approx(5.004561968247688, rel=1e-6)
    assert rec["sn"].stats().average == pytest.approx(0.0055862441084357384, rel=1e-6)


def test_testdata2_selection(testdata2: Path) -> None:
    proj = otiio.open(testdata2)
    assert len(proj.selections) == 1
    sel = proj.selections["TESTSEL"]
    assert sel.id == "099afdd9-749f-4554-b4b9-effb80c4a1fa"
    assert sel.color == "#e85f5f"
    assert (sel.start_us, sel.end_us) == (0, 1_295_796)
    assert sel.duration == pytest.approx(1.295796)
    rec = proj.recordings["TEST4"]
    # Samples with start <= t < end.
    assert len(rec["sn"].slice(sel)) == 6_479
    assert len(rec["vb"].slice(sel)) == 1_296
    assert rec["vb"].stats(sel).count == 1_296


# -- lookups ------------------------------------------------------------------------------


def test_recording_lookup_by_index_name_and_id(testdata2: Path) -> None:
    proj = otiio.open(testdata2)
    rec = proj.recordings[0]
    assert proj.recordings["TEST4"] is rec
    assert proj.recordings[rec.id] is rec
    assert proj.recordings[-1].name == "TEST3"
    assert [r.name for r in proj.recordings[0:2]] == ["TEST4", "TEST3"]
    assert "TEST4" in proj.recordings
    assert rec.id in proj.recordings
    with pytest.raises(KeyError, match="no recording named 'nope'"):
        proj.recordings["nope"]


def test_ambiguous_recording_name(testdata2: Path) -> None:
    proj = otiio.open(testdata2)
    for rec in proj.recordings:
        rec.name = "SAME"
    with pytest.raises(otiio.AmbiguousNameError, match="use an id"):
        proj.recordings["SAME"]
    with pytest.raises(KeyError):  # AmbiguousNameError is a KeyError
        proj.recordings["SAME"]


def test_channel_lookup_with_device(testdata2: Path) -> None:
    rec = otiio.open(testdata2).recordings["TEST4"]
    assert rec["vb", "Ace"] is rec["vb"]
    assert rec.channel("vb", device="Ace") is rec["vb"]
    assert rec.channel("vb", device=rec["vb"].device_id or "") is rec["vb"]
    assert "vb" in rec
    with pytest.raises(KeyError):
        rec["vb", "Arc"]
    with pytest.raises(KeyError, match="no channel 'mc'"):
        rec["mc"]


def test_same_channel_name_on_two_devices(patched: Any) -> None:
    def add_second_device(project: dict[str, Any]) -> None:
        ace = next(d for d in project["devices"] if d["type"] == "Ace")
        twin = {**ace, "id": "twin-device", "settings": {**ace["settings"], "name": "Ace2"}}
        project["devices"].append(twin)
        rec = project["recordings"][0]
        vb = next(m for m in rec["measurements"] if m["measurement"]["source"]["name"] == "vb")
        copy = {
            **vb,
            "measurement": {
                "id": "twin-vb",
                "source": {**vb["measurement"]["source"], "device": "twin-device"},
            },
        }
        rec["measurements"].append(copy)

    proj = otiio.open(patched(TESTDATA2, add_second_device))
    rec = proj.recordings[0]
    with pytest.raises(otiio.AmbiguousNameError, match="several devices"):
        rec["vb"]
    assert rec["vb", "Ace2"].id == "twin-vb"
    assert rec["vb", "Ace"].device == "Ace"
    assert len(rec.channels.all()) == 4


def test_channel_order_does_not_matter(patched: Any) -> None:
    """Shuffling stored lists must not change which channel is which."""
    ref = otiio.open(TESTDATA1)

    def shuffle(project: dict[str, Any]) -> None:
        rng = random.Random(1)
        for rec in project["recordings"]:
            rng.shuffle(rec["measurements"])
        for dev in project["devices"]:
            rng.shuffle(dev["channels"]["outputs"])
        rng.shuffle(project["measurements"])
        rng.shuffle(project["devices"])

    proj = otiio.open(patched(TESTDATA1, shuffle))
    for rec in ref.recordings:
        other = proj.recordings[rec.id]
        assert sorted(other.channels) == sorted(rec.channels)
        for name, ch in rec.channels.items():
            o = other[name]
            assert (o.id, o.label, o.rail, o.kind, o.rate) == (
                ch.id,
                ch.label,
                ch.rail,
                ch.kind,
                ch.rate,
            )
            np.testing.assert_array_equal(o.values, ch.values)


def test_missing_blob_raises_on_read(patched: Any) -> None:
    def break_blob(project: dict[str, Any]) -> None:
        project["recordings"][0]["measurements"][0]["data"]["id"] = "does-not-exist"

    rec = otiio.open(patched(TESTDATA2, break_blob)).recordings[0]
    broken = rec.channels.all()[0]
    with pytest.raises(otiio.MissingDataError):
        broken.values  # noqa: B018
    # The rest of the recording is still usable.
    assert rec.duration > 0


def test_non_sample_channel_not_implemented(patched: Any) -> None:
    def make_digital(project: dict[str, Any]) -> None:
        project["recordings"][0]["measurements"][0]["measurement"]["source"]["type"] = "digital"

    rec = otiio.open(patched(TESTDATA2, make_digital)).recordings[0]
    digital = next(c for c in rec.channels.all() if c.source_type == "digital")
    with pytest.raises(NotImplementedError, match="digital"):
        digital.values  # noqa: B018


def test_dangling_session_references_are_ignored(testdata1: Path) -> None:
    # sessions.json in testdata1 refers to recordings that no longer exist.
    proj = otiio.open(testdata1)
    assert len(proj.recordings) == 2


def test_reprs(testdata2: Path) -> None:
    proj = otiio.open(testdata2)
    assert "TEST4" in repr(proj)
    assert "TEST4" in repr(proj.recordings[0])
    assert "'vb'" in repr(proj.recordings[0]["vb"])


def test_gpi_labels() -> None:
    assert otiio.channels.channel_info("i1", "digital").label == "GPI1"
    assert otiio.channels.channel_info("i2", "digital").label == "GPI2"
