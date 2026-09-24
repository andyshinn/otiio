"""Human-readable names for Otii channels.

Channels are identified by their short name string (``source.name`` in project.json),
never by the numeric channel id, group id, or position in a list: those depend on the
device and on how the project was saved.
"""

from typing import NamedTuple


class ChannelInfo(NamedTuple):
    label: str
    rail: str | None = None


# Short name -> label/rail. Labels follow Otii's naming, e.g. "Main current - Ace" in its
# CSV export. Sources: otii-tcp-client-python channel list and observed Ace outputs.
CHANNELS: dict[str, ChannelInfo] = {
    "mc": ChannelInfo("Main current", "Main"),
    "mv": ChannelInfo("Main voltage", "Main"),
    "mp": ChannelInfo("Main power", "Main"),
    "me": ChannelInfo("Main energy", "Main"),
    "ac": ChannelInfo("ADC current", "ADC"),
    "av": ChannelInfo("ADC voltage", "ADC"),
    "ap": ChannelInfo("ADC power", "ADC"),
    "ae": ChannelInfo("ADC energy", "ADC"),
    "mc1": ChannelInfo("Main current 1", "Main"),
    "mc2": ChannelInfo("Main current 2", "Main"),
    "mc3": ChannelInfo("Main current 3", "Main"),
    "sn": ChannelInfo("Sense- voltage"),
    "sp": ChannelInfo("Sense+ voltage"),
    "vb": ChannelInfo("VBUS"),
    "vj": ChannelInfo("DC jack voltage"),
    "vpwr": ChannelInfo("Power supply voltage"),
    "tp": ChannelInfo("Temperature"),
    "tm": ChannelInfo("Temperature"),
    "rx": ChannelInfo("UART logs"),
    "i1": ChannelInfo("GPI1"),
    "i2": ChannelInfo("GPI2"),
}

UNITS: dict[str, str] = {
    "current": "A",
    "voltage": "V",
    "power": "W",
    "energy": "J",
    "charge": "C",
    "temperature": "°C",
}

# Rail -> short-name prefix for its current/voltage/power/energy channels.
RAIL_PREFIX: dict[str, str] = {"Main": "m", "ADC": "a"}
KIND_SUFFIX: dict[str, str] = {"current": "c", "voltage": "v", "power": "p", "energy": "e"}


def channel_info(name: str, kind: str) -> ChannelInfo:
    """Label and rail for a channel short name, falling back to ``"<name> <kind>"``."""
    return CHANNELS.get(name, ChannelInfo(f"{name} {kind}"))


def unit_for(kind: str) -> str:
    return UNITS.get(kind, "")


def rail_channel_name(rail: str, kind: str) -> str:
    """Short name of a rail's channel, e.g. ("Main", "power") -> "mp"."""
    try:
        return RAIL_PREFIX[rail] + KIND_SUFFIX[kind]
    except KeyError:
        raise ValueError(f"no {kind!r} channel on rail {rail!r}") from None
