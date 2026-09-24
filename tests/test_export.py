from pathlib import Path

import numpy as np
import pytest

import otiio

from .conftest import DATA, make_signal

CSVTEST1 = DATA / "csvtest1"


@pytest.mark.parametrize("name", ["sn", "sp", "vb"])
def test_csv_is_byte_identical_to_otii_export(testdata2: Path, tmp_path: Path, name: str) -> None:
    """tests/data/csvtest1 holds Otii's own CSV export of testdata2 recording TEST3."""
    ch = otiio.open(testdata2).recordings["TEST3"][name]
    out = ch.to_csv(tmp_path, chunk=500)  # a directory: named like Otii names it
    otii = CSVTEST1 / f"{ch.label}.csv"
    assert out.name == otii.name
    assert out.read_bytes() == otii.read_bytes()


def test_csv_roundtrip(testdata2: Path, tmp_path: Path) -> None:
    ch = otiio.open(testdata2).recordings["TEST4"]["vb"]
    out = ch.to_csv(tmp_path / "vb.csv")
    assert out == tmp_path / "vb.csv"
    lines = out.read_text().splitlines()
    assert lines[0] == '"Timestamp","Value"'
    assert len(lines) == 1 + len(ch)
    data = np.loadtxt(out, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 0], ch.times(), atol=1e-6)
    # %.9g round-trips float32 exactly.
    np.testing.assert_array_equal(data[:, 1].astype(np.float32), ch.values)


def test_csv_of_slice_and_derived(tmp_path: Path) -> None:
    sig = make_signal([1.0, 2.0, 3.0], rate=1.0, name="mp", device="Ace")
    out = sig.slice(1).to_csv(tmp_path)
    assert out.name == "Main power - Ace.csv"
    assert out.read_text().splitlines() == ['"Timestamp","Value"', "1.000000,2", "2.000000,3"]


def test_parquet_roundtrip(testdata2: Path, tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    ch = otiio.open(testdata2).recordings["TEST4"]["sn"]
    out = tmp_path / "sn.parquet"
    ch.to_parquet(out, chunk=1000)
    table = pq.read_table(out)
    assert table.column_names == ["timestamp", "value"]
    np.testing.assert_array_equal(table["value"].to_numpy(), ch.values)
    np.testing.assert_allclose(table["timestamp"].to_numpy(), ch.times())
    meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert meta["otiio.label"] == "Sense- voltage - Ace"
    assert meta["otiio.unit"] == "V"
    assert float(meta["otiio.rate"]) == 5000


def test_recording_parquet_long_format(testdata2: Path, tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    rec = otiio.open(testdata2).recordings["TEST3"]
    out = tmp_path / "rec.parquet"
    rec.to_parquet(out)
    df = pq.read_table(out).to_pandas()
    assert list(df.columns) == ["channel", "device", "timestamp", "value"]
    counts = df["channel"].astype(str).value_counts().to_dict()
    assert counts == {"sn": 7_360, "sp": 7_360, "vb": 1_472}
    vb = df[df["channel"] == "vb"]
    np.testing.assert_array_equal(vb["value"].to_numpy(), rec["vb"].values)
