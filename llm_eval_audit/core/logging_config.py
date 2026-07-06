"""Consistent logging setup shared by every component in the framework."""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Configures the root logging handler exactly once per process so that
    importing this module from many places doesn't create duplicate handlers
    (and therefore duplicate log lines).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
