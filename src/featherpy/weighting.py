"""Feather weighting"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# Approximation of the standard normal distribution at 1 sigma
# to use in sigmoid function (sigma -> k)
ONE_STD = float(-1 * np.log((1 - stats.norm.cdf(1)) * 2))


def sigmoid(x: NDArray[np.float64], x0: float = 0, k: float = 1) -> NDArray[np.float64]:
    """Sigmoid function

    Args:
        x (NDArray[np.float64]): x values
        x0 (float, optional): x offset. Defaults to 0.
        k (float, optional): growth rate. Defaults to 1.

    Returns:
        NDArray[np.float64]: sigmoid values
    """
    return np.array(1 / (1 + np.exp(-k * (x - x0))))
