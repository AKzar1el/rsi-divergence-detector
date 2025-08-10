"""Utility functions for the RSI divergence detector."""
from __future__ import annotations
import numpy as np

def wilder_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using Wilder's smoothing (recursive EMA with alpha=1/period).

    Returns
    -------
    np.ndarray
        RSI values (0..100). Initial warmup up to `period` will be 0.0.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < period + 1:
        raise ValueError("prices must be 1D and length >= period+1")

    deltas = np.diff(prices)
    seed = deltas[:period]  # fixed off-by-one
    up = np.clip(seed, 0, None).sum() / period
    down = -np.clip(seed, None, 0).sum() / period
    down = max(down, 1e-12)  # guard zero

    rs = up / down
    rsi = np.zeros_like(prices, dtype=float)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0.0)
        downval = max(-delta, 0.0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        down = max(down, 1e-12)
        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi
