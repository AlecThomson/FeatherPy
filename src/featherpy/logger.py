"""FeatherPy logging module"""

from __future__ import annotations

import logging

logging.captureWarnings(True)


logger = logging.getLogger("featherpy")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="[%(threadName)s] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
