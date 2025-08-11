# utils.py
from __future__ import annotations
import numpy as np

def wilder_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder RSI using SMA seed over first `period` deltas, then recursive smoothing.

    Returns
    -------
    np.ndarray
        RSI values (0..100). First `period` entries are filled with the seed RSI.
        When both avg gain and avg loss are zero (flat), RSI is 50.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < period + 1:
        raise ValueError("prices must be 1D and length >= period+1")

    deltas = np.diff(prices)                           # length N-1
    gains  = np.clip(deltas,  0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    # --- Wilder seed over first `period` deltas
    up   = gains[:period].mean()
    down = losses[:period].mean()

    rsi = np.empty_like(prices, dtype=float)

    # Seed RSI for the warmup range
    if up == 0.0 and down == 0.0:
        seed_rsi = 50.0
    else:
        rs = up / (down if down > 0.0 else 1e-12)
        seed_rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = seed_rsi

    # --- Recursive Wilder smoothing
    for i in range(period, len(prices)):
        g = gains[i - 1]
        l = losses[i - 1]
        up   = (up * (period - 1) + g) / period
        down = (down * (period - 1) + l) / period

        if up == 0.0 and down == 0.0:
            rsi[i] = 50.0
        else:
            rs = up / (down if down > 0.0 else 1e-12)
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi
