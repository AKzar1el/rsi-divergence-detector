import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from scipy.signal import find_peaks

# ... keep calculate_rsi and _find_pivots as-is ...

def _nearest_monotonic(right: np.ndarray, target: int, max_lag: int, last_taken: Optional[int]) -> Optional[int]:
    """
    Pick the nearest index in `right` to `target` within ±max_lag bars,
    but enforce monotonically increasing time by requiring right[k] > last_taken.
    Returns the POSITION k in `right` or None.
    """
    k = int(np.searchsorted(right, target))
    best = None
    for cand in (k-1, k, k+1):
        if 0 <= cand < len(right):
            t = right[cand]
            if last_taken is not None and t <= last_taken:
                continue
            if abs(t - target) <= max_lag:
                dist = abs(t - target)
                if best is None or dist < best[0]:
                    best = (dist, cand)
    return None if best is None else best[1]


def find_divergences(
    prices: pd.Series,
    rsi: pd.Series,
    *,
    rsi_period: int = 14,
    price_prominence: Optional[float] = None,
    rsi_prominence: Optional[float] = None,
    price_width: Optional[int] = None,
    rsi_width: Optional[int] = None,
    distance: Optional[int] = None,
    max_lag: int = 3,
    include_hidden: bool = True,
    allow_equal: bool = True,   # NEW: treat ties as valid
) -> pd.DataFrame:
    """
    Detect regular and (optionally) hidden divergences by pairing *consecutive*
    price pivots with *nearest-in-time* RSI pivots (monotonic).
    """
    prices = pd.Series(prices, dtype="float64")
    rsi = pd.Series(rsi, dtype="float64").reindex(prices.index)

    # --- Adaptive defaults (unchanged) ---
    if price_prominence is None:
        vol = prices.pct_change().rolling(rsi_period).std().iloc[-1]
        vol = float(vol) if np.isfinite(vol) and vol > 0 else 0.005
        price_prominence = 0.5 * vol * float(prices.iloc[-1])
    if rsi_prominence is None:
        rsi_prominence = 5.0
    if price_width is None:
        price_width = 2
    if rsi_width is None:
        rsi_width = 2
    if distance is None:
        distance = max(1, rsi_period // 2)

    # --- Find pivots ---
    p_min, p_max = _find_pivots(prices, price_prominence, price_width, distance)
    r_min, r_max = _find_pivots(rsi,     rsi_prominence,   rsi_width, distance)

    out: List[Divergence] = []

    def cmp(a, b, op: str) -> bool:
        if allow_equal:
            return (a <= b) if op == "<" else (a >= b)
        return (a < b) if op == "<" else (a > b)

    # ---------- Regular Bullish: price LL & RSI HL ----------
    last_r = None
    for i in range(len(p_min) - 1):
        p1, p2 = p_min[i], p_min[i + 1]
        k1 = _nearest_monotonic(r_min, p1, max_lag, last_r)
        if k1 is None:
            continue
        k2 = _nearest_monotonic(r_min, p2, max_lag, k1)
        if k2 is None:
            continue
        r1, r2 = r_min[k1], r_min[k2]
        last_r = k2

        if cmp(prices.iat[p2], prices.iat[p1], "<") and cmp(rsi.iat[r2], rsi.iat[r1], ">"):
            out.append(Divergence(
                "regular_bullish",
                prices.index[p1], float(prices.iat[p1]),
                prices.index[p2], float(prices.iat[p2]),
                rsi.index[r1], float(rsi.iat[r1]),
                rsi.index[r2], float(rsi.iat[r2]),
            ))

    # ---------- Regular Bearish: price HH & RSI LH ----------
    last_r = None
    for i in range(len(p_max) - 1):
        p1, p2 = p_max[i], p_max[i + 1]
        k1 = _nearest_monotonic(r_max, p1, max_lag, last_r)
        if k1 is None:
            continue
        k2 = _nearest_monotonic(r_max, p2, max_lag, k1)
        if k2 is None:
            continue
        r1, r2 = r_max[k1], r_max[k2]
        last_r = k2

        if cmp(prices.iat[p2], prices.iat[p1], ">") and cmp(rsi.iat[r2], rsi.iat[r1], "<"):
            out.append(Divergence(
                "regular_bearish",
                prices.index[p1], float(prices.iat[p1]),
                prices.index[p2], float(prices.iat[p2]),
                rsi.index[r1], float(rsi.iat[r1]),
                rsi.index[r2], float(rsi.iat[r2]),
            ))

    if include_hidden:
        # ---------- Hidden Bullish: price HL & RSI LL ----------
        last_r = None
        for i in range(len(p_min) - 1):
            p1, p2 = p_min[i], p_min[i + 1]
            k1 = _nearest_monotonic(r_min, p1, max_lag, last_r)
            if k1 is None:
                continue
            k2 = _nearest_monotonic(r_min, p2, max_lag, k1)
            if k2 is None:
                continue
            r1, r2 = r_min[k1], r_min[k2]
            last_r = k2

            if cmp(prices.iat[p2], prices.iat[p1], ">") and cmp(rsi.iat[r2], rsi.iat[r1], "<"):
                out.append(Divergence(
                    "hidden_bullish",
                    prices.index[p1], float(prices.iat[p1]),
                    prices.index[p2], float(prices.iat[p2]),
                    rsi.index[r1], float(rsi.iat[r1]),
                    rsi.index[r2], float(rsi.iat[r2]),
                ))

        # ---------- Hidden Bearish: price LH & RSI HH ----------
        last_r = None
        for i in range(len(p_max) - 1):
            p1, p2 = p_max[i], p_max[i + 1]
            k1 = _nearest_monotonic(r_max, p1, max_lag, last_r)
            if k1 is None:
                continue
            k2 = _nearest_monotonic(r_max, p2, max_lag, k1)
            if k2 is None:
                continue
            r1, r2 = r_max[k1], r_max[k2]
            last_r = k2

            if cmp(prices.iat[p2], prices.iat[p1], "<") and cmp(rsi.iat[r2], rsi.iat[r1], ">"):
                out.append(Divergence(
                    "hidden_bearish",
                    prices.index[p1], float(prices.iat[p1]),
                    prices.index[p2], float(prices.iat[p2]),
                    rsi.index[r1], float(rsi.iat[r1]),
                    rsi.index[r2], float(rsi.iat[r2]),
                ))

    return pd.DataFrame([asdict(d) for d in out])
