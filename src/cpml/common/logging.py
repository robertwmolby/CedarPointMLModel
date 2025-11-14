from __future__ import annotations
import logging
import os
import sys
from typing import Optional

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a namespaced logger (e.g., 'cpml.predictive.preprocess').
    Does NOT configure handlers—safe for library code.
    """
    logger = logging.getLogger("cpml" + ("" if not name else f".{name}"))
    # Library modules should have a NullHandler by default.
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger

def configure_logging(
    level: str | int = None,
    stream: object = sys.stdout,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """
    App/notebook entrypoint: attach a single StreamHandler to 'cpml' root.
    Safe to call once; subsequent calls won’t duplicate handlers.
    """
    root = logging.getLogger("cpml")
    if root.handlers:
        return

    lvl = (
        level
        if level is not None
        else os.getenv("CPML_LOG_LEVEL", "INFO")
    )
    root.setLevel(lvl)

    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(h)
