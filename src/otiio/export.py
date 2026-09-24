"""CSV and Parquet export."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .signal import DEFAULT_CHUNK, Signal

if TYPE_CHECKING:
    from .model import Recording


# Otii's export names the channel in the file name ("VBUS - Ace.csv"), not the header.
CSV_HEADER = '"Timestamp","Value"'


def signal_to_csv(sig: Signal, path: str | Path, *, chunk: int = DEFAULT_CHUNK) -> Path:
    """Write CSV in Otii's export format; byte-identical to Otii for stored channels.

    If ``path`` is a directory, the file is named like Otii's: ``<label>.csv``.
    """
    out = Path(path)
    if out.is_dir():
        out = out / f"{sig.label}.csv"
    # %.6f seconds and %.9g values (enough to round-trip float32), as Otii writes them.
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write(CSV_HEADER + "\n")
        for block in sig.chunks(chunk):
            np.savetxt(f, np.column_stack((block.t, block.v)), fmt=("%.6f", "%.9g"), delimiter=",")
    return out


def _pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("Parquet export needs pyarrow: pip install 'otiio[pandas]'") from e
    return pa, pq


def _metadata(sig: Signal) -> dict[str, str]:
    return {
        "otiio.name": sig.name,
        "otiio.label": sig.label,
        "otiio.kind": sig.kind,
        "otiio.unit": sig.unit,
        "otiio.device": sig.device,
        "otiio.rate": repr(sig.rate),
        "otiio.derived": str(sig.derived).lower(),
    }


def signal_to_parquet(sig: Signal, path: str | Path, *, chunk: int = DEFAULT_CHUNK) -> None:
    pa, pq = _pyarrow()
    value_type = pa.float64() if sig.derived else pa.float32()
    schema = pa.schema([("timestamp", pa.float64()), ("value", value_type)], metadata=_metadata(sig))
    with pq.ParquetWriter(str(path), schema) as writer:
        for block in sig.chunks(chunk):
            writer.write_table(pa.table({"timestamp": block.t, "value": block.v}, schema=schema))


def recording_to_parquet(rec: "Recording", path: str | Path, *, chunk: int = DEFAULT_CHUNK) -> None:
    pa, pq = _pyarrow()
    schema = pa.schema(
        [
            ("channel", pa.dictionary(pa.int16(), pa.string())),
            ("device", pa.dictionary(pa.int16(), pa.string())),
            ("timestamp", pa.float64()),
            ("value", pa.float32()),
        ],
        metadata={
            "otiio.recording": rec.name or "",
            "otiio.recording_id": rec.id,
            "otiio.start": rec.start.isoformat(),
        },
    )
    with pq.ParquetWriter(str(path), schema) as writer:
        for ch in readable_channels(rec):
            for block in ch.chunks(chunk):
                n = len(block.t)
                table = pa.table(
                    {
                        "channel": pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, dtype=np.int16)), [ch.name]),
                        "device": pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, dtype=np.int16)), [ch.device]),
                        "timestamp": block.t,
                        "value": block.v,
                    },
                    schema=schema,
                )
                writer.write_table(table)


def readable_channels(rec: "Recording") -> list[Signal]:
    return [ch for ch in rec.channels.all() if ch.source_type == "samples" and ch.path is not None and ch.path.is_file()]
