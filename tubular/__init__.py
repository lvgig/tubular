from contextlib import suppress
from importlib.metadata import version

from tubular import (
    base,
    capping,
    dates,
    imputers,
    mapping,
    misc,
    nominal,
    numeric,
    strings,
)

with suppress(ModuleNotFoundError):
    __version__ = version("tubular")
