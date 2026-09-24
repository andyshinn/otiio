"""Checks against the large projects in examples/ (not committed; skipped when absent)."""

import numpy as np
import pytest

import otiio

from .conftest import EXAMPLES

MPPT = EXAMPLES / "MPPT_TEST"
MPPT_DS = EXAMPLES / "MPPT_TEST_DOWNSAMPLED"

pytestmark = [
    pytest.mark.examples,
    pytest.mark.skipif(not MPPT.exists(), reason="examples/MPPT_TEST not present"),
]

# name: (samples/channel, offset µs, t_end s, mean mc A, mean mv V)
REFERENCE = {
    "WAVESHARE": (44_984_315, 2_490_160, 902.17644, 0.063428, 3.92200),
    "WAVESHARE NO I2C": (801_550, 0, 16.03098, 0.0648329, 3.90219),
    "5V DIRECT": (827_000, 0, 16.53998, 0.0229418, 5.00000),
    "DFROBOT": (820_800, 0, 16.41598, 0.0362592, 3.92726),
    "DFROBOT NO I2C": (664_850, 0, 13.29698, 0.0347335, 3.92708),
}


@pytest.fixture(scope="module")
def mppt() -> otiio.Project:
    return otiio.open(MPPT / "MPPT_TEST.otii3")


@pytest.mark.parametrize("name", list(REFERENCE))
def test_reference_table(mppt: otiio.Project, name: str) -> None:
    n, offset, t_end, mean_mc, mean_mv = REFERENCE[name]
    rec = mppt.recordings[name]
    for ch in ("mc", "mv", "mp"):
        assert len(rec[ch]) == n
        assert rec[ch].rate == 50_000
        assert rec[ch].offset_us == offset
    assert rec["mc"].start == pytest.approx(offset / 1e6)
    assert rec["mc"].end == pytest.approx(t_end)
    assert rec["mc"].stats().average == pytest.approx(mean_mc, rel=1e-5)
    assert rec["mv"].stats().average == pytest.approx(mean_mv, rel=1e-5)


def test_selections(mppt: otiio.Project) -> None:
    sel = mppt.selections["Sel 1"]
    assert (sel.start_us, sel.end_us, sel.color) == (2_482_248, 902_168_554, "#e85f5f")
    assert "Sel 2" in mppt.selections


def test_virtual_device_without_data_is_tolerated(mppt: otiio.Project) -> None:
    assert "VirtualDevice" in [d.type for d in mppt.devices]
    assert all(ch.device == "Ace" for rec in mppt.recordings for ch in rec.channels.values())


@pytest.mark.skipif(not MPPT_DS.exists(), reason="examples/MPPT_TEST_DOWNSAMPLED not present")
def test_downsample_matches_otii(mppt: otiio.Project) -> None:
    ds_proj = otiio.open(MPPT_DS)
    for rec in mppt.recordings:
        otii = ds_proj.recordings[rec.id]["mc"]
        ours = rec["mc"].downsample(500)
        assert otii.rate == ours.rate == 100
        assert len(ours) == len(otii)
        assert ours.start == pytest.approx(otii.start)
        np.testing.assert_allclose(ours.values, otii.values, atol=5e-8)


@pytest.mark.skipif(not MPPT_DS.exists(), reason="examples/MPPT_TEST_DOWNSAMPLED not present")
def test_mixed_rate_power() -> None:
    rec = otiio.open(MPPT_DS).recordings["DFROBOT NO I2C"]
    assert (rec["mc"].rate, rec["mv"].rate) == (100, 50_000)
    p = otiio.derived.multiply(rec["mc"], rec["mv"], name="mp", kind="power")
    ref = rec["mp"].downsample(500)
    assert p.rate == 100 and len(p) == len(ref)
    # mean(I) * mean(V) differs slightly from mean(I * V) within each block.
    np.testing.assert_allclose(p.values, ref.values, rtol=5e-3)


@pytest.mark.skipif(not MPPT_DS.exists(), reason="examples/MPPT_TEST_DOWNSAMPLED not present")
def test_downsampled_current_energy_uses_full_rate_power() -> None:
    rec = otiio.open(MPPT_DS).recordings["DFROBOT NO I2C"]
    mc = rec["mc"]
    assert mc.rate == 100 and rec["mp"].rate == 50_000
    # mc covers whole 500-sample blocks of mp, so the trailing partial block is left out.
    expected = rec["mp"].slice(mc.start, mc.start + mc.duration).stats().energy
    assert mc.stats().energy == pytest.approx(expected, rel=1e-12)
