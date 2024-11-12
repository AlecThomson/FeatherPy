"""FeatherPy logging module"""

from __future__ import annotations

import logging

logging.captureWarnings(True)

# Following guide from gwerbin/multiprocessing_logging.py
# https://gist.github.com/gwerbin/e9ab7a88fef03771ab0bf3a11cf921bc


def setup_logger() -> logging.Logger:
    """Setup a logger

    Args:
        filename (Optional[str], optional): Output log file. Defaults to None.

    Returns:
        logging.Logger: The logger
    """
    logger = logging.getLogger("cutout_fits")
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        fmt="[%(threadName)s] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


logger = setup_logger()
