"""Utility functions for the RSI divergence detector."""

import numpy as np

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



