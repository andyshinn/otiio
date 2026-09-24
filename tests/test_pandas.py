from pathlib import Path

import numpy as np
import pytest

import otiio

pd = pytest.importorskip("pandas")


def test_channel_dataframe(testdata2: Path) -> None:
    ch = otiio.open(testdata2).recordings["TEST4"]["vb"]
    df = ch.to_dataframe()
    assert list(df.columns) == ["vb"]
    assert df.index.name == "timestamp"
    assert len(df) == 1_342
    np.testing.assert_array_equal(df["vb"].to_numpy(), ch.values)
    assert df.attrs["unit"] == "V"
    assert df.attrs["label"] == "VBUS - Ace"


def test_recording_dataframe(testdata2: Path) -> None:
    rec = otiio.open(testdata2).recordings["TEST4"]
    df = rec.to_dataframe()
    assert list(df.columns) == ["channel", "device", "timestamp", "value"]
    assert isinstance(df["channel"].dtype, pd.CategoricalDtype)
    means = df.groupby("channel", observed=True)["value"].mean()
    assert means["vb"] == pytest.approx(5.004561968247688, rel=1e-6)
    assert df.attrs["recording"] == "TEST4"
