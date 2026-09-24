# otiio

Parse and read [Otii 3](https://www.qoitech.com/) project files (`.otii3`) in Python. Helpful for generating charts or other Otii project reporting in Python.

- Zero-copy sample access through `numpy.memmap`, so multi-GB recordings open instantly
- Recordings, channels, devices and saved selections, looked up by name
- Streaming statistics (min, max, average, RMS, energy, charge) over any time window
- Power, current or voltage derived from the other two when one wasn't recorded, plus cumulative energy
- CSV export identical to Otii's own; Parquet and pandas with the `pandas` extra

## Install

```sh
pip install otiio            # numpy only
pip install 'otiio[pandas]'  # adds pandas + pyarrow for DataFrames and Parquet
```

Requires Python 3.12+.

## Usage

```python
import otiio

proj = otiio.open("MPPT_TEST/MPPT_TEST.otii3")  # or the project folder

for rec in proj.recordings:
    print(rec.name, rec.start, f"{rec.duration:.2f}s", list(rec.channels))

rec = proj.recordings["WAVESHARE"]  # by name, id, or position
mc = rec["mc"]  # by short channel name
mc.label, mc.unit, mc.rate  # 'Main current - Ace', 'A', 50000.0
mc.values  # np.memmap float32, in amperes
mc.times()  # float64 seconds on the project timeline

for t, v in mc.chunks(1_000_000):  # stream long recordings
    ...

window = mc.slice(10.0, 20.0)  # the time range [10, 20), no copy
sel = proj.selections["Sel 1"]  # a selection saved in Otii
print(rec["mp"].stats(sel))  # Stats(min, max, average, rms, energy=... J)
rec["mc"].stats(sel).energy  # current stats carry energy from power too, as in Otii

p = rec.power()  # stored mp, or mc * mv when power wasn't recorded
e = rec.energy()  # cumulative joules, derived from power
slow = mc.downsample(500)  # block mean, same as Otii's downsampling

mc.to_csv("out/")  # out/Main current - Ace.csv, same format as Otii's export
mc.to_parquet("mc.parquet")  # needs otiio[pandas]
df = rec.to_dataframe()  # long format: channel, device, timestamp, value
```

### Channels

A channel is identified by its short name (`mc`, `mv`, `mp`, `ac`, `av`, `sn`, `sp`, `vb`, …)
and device, never by numeric id or stored order. Rails come from the name: `m*` is **Main**,
`a*` is **ADC**. If two devices in one recording share a short name, use
`rec["mc", "Ace"]`.

Each channel has its own sample rate and time offset, so rates can differ within one
recording. Combining channels at different rates (e.g. `rec.power()` from a downsampled
current) block-averages the faster one first.

### Downsampling

`downsample(factor)` averages non-overlapping blocks of `factor` samples, the same way
Otii's own downsampling does. It matches projects downsampled in Otii to about 1e-8.
Like Otii, it returns `(n - 1) // factor` samples, so the final block is always dropped,
even when it's complete. The result is computed lazily, chunk by chunk, so it works on
recordings too large to fit in memory:

```python
slow = rec["mc"].downsample(500)  # 50 kHz -> 100 sps
slow.rate, slow.start  # 100.0, same start time as the source
for t, v in slow.chunks():  # streams; never loads the full 50 kHz channel
    ...
```

For an array that fits in memory, a numpy `reshape` gives the same values:

```python
import numpy as np

v = rec["mc"].values
factor = 500
n = (len(v) - 1) // factor  # Otii drops the final block
blocks = v[: n * factor].reshape(n, factor)  # one row per block
means = blocks.mean(axis=1, dtype=np.float64)  # same values as mc.downsample(500).values
```

### Statistics match the Otii app

`stats()` computes min, max, average, energy and charge the way the Otii app does. The
results match its Selections panel at the precision it displays (checked against
`tests/data/battest1`):

- Each sample holds its value until the next one, so it covers `[t, t + 1/rate)`.
- A window (a selection, or `slice(t0, t1)`) covers exactly `[t0, t1)`. It includes every
  sample whose interval overlaps that range, so min and max can come from a sample that
  starts just before `t0`. Average, energy and charge weight the first and last samples
  by how much of them falls inside.
- Energy and charge are the integral of those held values, a rectangle sum rather than a
  trapezoid. The average is the time-weighted mean, so energy = average power × duration.
- Windows are clipped to the data. A whole channel covers `len / rate` seconds.

### What's not supported yet

- Digital/GPIO, UART log, and other non-sample channels are listed, but reading their
  values raises `NotImplementedError`.
- Writing or modifying projects.

## Development

```sh
uv sync --all-extras
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest
```

`tests/data/` holds small real projects. Tests marked `examples` also run against the
larger projects in `examples/` when that folder exists (it's not committed).

## License

MIT
