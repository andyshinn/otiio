"""pandas helpers (``pip install 'otiio[pandas]'``)."""

from typing import TYPE_CHECKING

import numpy as np

from .export import readable_channels

if TYPE_CHECKING:
    import pandas as pd

    from .model import Recording
    from .signal import Signal


def _pandas() -> "type[pd.DataFrame]":
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("DataFrame export needs pandas: pip install 'otiio[pandas]'") from e
    return pd.DataFrame


def signal_to_dataframe(sig: "Signal") -> "pd.DataFrame":
    DataFrame = _pandas()
    df = DataFrame({sig.name: np.asarray(sig.values)}, index=sig.times())
    df.index.name = "timestamp"
    df.attrs.update(
        name=sig.name,
        label=sig.label,
        kind=sig.kind,
        unit=sig.unit,
        device=sig.device,
        rate=sig.rate,
        derived=sig.derived,
    )
    return df


def recording_to_dataframe(rec: "Recording") -> "pd.DataFrame":
    import pandas as pd

    _pandas()
    frames = [
        pd.DataFrame(
            {
                "channel": pd.Categorical([ch.name] * len(ch)),
                "device": pd.Categorical([ch.device] * len(ch)),
                "timestamp": ch.times(),
                "value": np.asarray(ch.values),
            }
        )
        for ch in readable_channels(rec)
    ]
    if not frames:
        return pd.DataFrame(columns=["channel", "device", "timestamp", "value"])
    df = pd.concat(frames, ignore_index=True)
    for col in ("channel", "device"):
        df[col] = df[col].astype("category")
    df.attrs.update(recording=rec.name, recording_id=rec.id, start=rec.start)
    return df
