"""Utility functions for the RSI divergence detector."""

import numpy as np
from numba import njit
from scipy.signal import find_peaks

@njit
def wilder_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using Wilder's smoothing.

    Args:
        prices: Array of prices.
        period: The lookback period for RSI calculation.

    Returns:
        An array containing the RSI values.
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]

    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def find_extrema(series: np.ndarray, prominence: int = 1, width: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Find local minima and maxima in a series.

    Args:
        series: The data series.
        prominence: The prominence of the peaks.
        width: The width of the peaks.

    Returns:
        A tuple containing arrays of indices for minima and maxima.
    """
    maxima_indices, _ = find_peaks(series, prominence=prominence, width=width)
    minima_indices, _ = find_peaks(-series, prominence=prominence, width=width)

    return minima_indices, maxima_indices

