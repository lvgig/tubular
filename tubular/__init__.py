from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

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

with suppress(PackageNotFoundError):
    __version__ = version("tubular")
