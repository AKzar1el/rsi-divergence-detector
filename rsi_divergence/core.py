# core.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_SCHEMA = ["kind","p1_idx","p1_price","p2_idx","p2_price","r1_idx","r1_val","r2_idx","r2_val"]
__all__ = ["calculate_rsi", "find_divergences", "Divergence", "DivergenceParams"]


@dataclass(frozen=True)
class Divergence:
    """Single divergence detection result."""
    kind: str  # "regular_bullish" | "regular_bearish" | "hidden_bullish" | "hidden_bearish"
    p1_idx: pd.Timestamp
    p1_price: float
    p2_idx: pd.Timestamp
    p2_price: float
    r1_idx: pd.Timestamp
    r1_val: float
    r2_idx: pd.Timestamp
    r2_val: float


@dataclass(frozen=True)
class DivergenceParams:
    """Detector settings (no globals)."""
    rsi_period: int = 14
    price_prominence: Optional[float] = None
    rsi_prominence: Optional[float] = None
    price_width: Optional[int] = None
    rsi_width: Optional[int] = None
    distance: Optional[int] = None
    max_lag: int = 3
    include_hidden: bool = True
    allow_equal: bool = True


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using the *exact* Wilder method (SMA seed + recursive smoothing).

    Preconditions
    -------------
    - `prices` is a non-empty numeric Series with monotonic-increasing DatetimeIndex.
    - `period >= 2`.

    Postconditions
    --------------
    - Returns a Series aligned to `prices.index`.
    - Values are in [0, 100]. First `period` positions are the seeded RSI.
    """
    if not isinstance(prices, pd.Series) or prices.empty:
        raise ValueError("`prices` must be a non-empty pandas Series.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("`prices.index` must be monotonic increasing.")
    if period < 2:
        raise ValueError("`period` must be >= 2.")

    from .utils import wilder_rsi  # safe local import

    arr = wilder_rsi(prices.to_numpy(dtype=float), period=period)
    return pd.Series(arr, index=prices.index, name="rsi")


def _find_pivots(
    series: pd.Series,
    prominence: float,
    width: int,
    distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (minima_idx, maxima_idx) using SciPy `find_peaks` on y and -y."""
    # Prominence/width/distance are the canonical gates for robust peak selection. :contentReference[oaicite:0]{index=0}
    y = series.values.astype("float64", copy=False)
    maxima, _ = find_peaks(y,   prominence=prominence, width=width, distance=distance)
    minima, _ = find_peaks(-y,  prominence=prominence, width=width, distance=distance)
    return minima, maxima


def _nearest_monotonic(
    right_positions: np.ndarray,
    target_pos: int,
    max_lag: int,
    last_taken_abs: Optional[int],
) -> Optional[int]:
    """Return the **slot index** k into `right_positions` nearest to `target_pos`
    within ±max_lag, strictly after `last_taken_abs` (an absolute series index)."""
    k_guess = int(np.searchsorted(right_positions, target_pos))
    best: Optional[Tuple[int, int]] = None  # (dist, slot_k)
    for cand in (k_guess - 1, k_guess, k_guess + 1):
        if 0 <= cand < len(right_positions):
            abs_idx = right_positions[cand]        # absolute pivot position in the series
            if last_taken_abs is not None and abs_idx <= last_taken_abs:
                continue
            dist = abs(abs_idx - target_pos)
            if dist <= max_lag and (best is None or dist < best[0]):
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
    allow_equal: bool = True,
) -> pd.DataFrame:
    """Detect **regular** and (optionally) **hidden** RSI divergences."""
    if not isinstance(prices, pd.Series) or not isinstance(rsi, pd.Series):
        raise ValueError("`prices` and `rsi` must be pandas Series.")
    if prices.empty:
        raise ValueError("`prices` must be non-empty.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("`prices.index` must be monotonic increasing.")
    rsi = rsi.reindex(prices.index)
    if len(rsi) != len(prices):
        raise ValueError("`rsi` could not be aligned to `prices.index`.")

    prices = prices.astype("float64")
    rsi = rsi.astype("float64")

    # Adaptive defaults
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

    # Find pivots
    p_min, p_max = _find_pivots(prices, price_prominence, price_width, distance)
    r_min, r_max = _find_pivots(rsi,     rsi_prominence,   rsi_width,   distance)

    out: List[Divergence] = []

    def _cmp(a: float, b: float, op: str) -> bool:
        if allow_equal:
            return (a <= b) if op == "<" else (a >= b)
        return (a < b) if op == "<" else (a > b)

    # ---------- Regular Bullish: price LL & RSI HL ----------
    last_r_abs: Optional[int] = None
    for i in range(len(p_min) - 1):
        p1, p2 = p_min[i], p_min[i + 1]
        k1 = _nearest_monotonic(r_min, p1, max_lag, last_r_abs)
        if k1 is None:
            continue
        r1_abs = r_min[k1]
        k2 = _nearest_monotonic(r_min, p2, max_lag, r1_abs)
        if k2 is None:
            continue
        r2_abs = r_min[k2]
        last_r_abs = r2_abs  # <-- store absolute pivot index, not slot

        if _cmp(prices.iat[p2], prices.iat[p1], "<") and _cmp(rsi.iat[r2_abs], rsi.iat[r1_abs], ">"):
            out.append(Divergence(
                "regular_bullish",
                prices.index[p1], float(prices.iat[p1]),
                prices.index[p2], float(prices.iat[p2]),
                rsi.index[r1_abs], float(rsi.iat[r1_abs]),
                rsi.index[r2_abs], float(rsi.iat[r2_abs]),
            ))

    # ---------- Regular Bearish: price HH & RSI LH ----------
    last_r_abs = None
    for i in range(len(p_max) - 1):
        p1, p2 = p_max[i], p_max[i + 1]
        k1 = _nearest_monotonic(r_max, p1, max_lag, last_r_abs)
        if k1 is None:
            continue
        r1_abs = r_max[k1]
        k2 = _nearest_monotonic(r_max, p2, max_lag, r1_abs)
        if k2 is None:
            continue
        r2_abs = r_max[k2]
        last_r_abs = r2_abs

        if _cmp(prices.iat[p2], prices.iat[p1], ">") and _cmp(rsi.iat[r2_abs], rsi.iat[r1_abs], "<"):
            out.append(Divergence(
                "regular_bearish",
                prices.index[p1], float(prices.iat[p1]),
                prices.index[p2], float(prices.iat[p2]),
                rsi.index[r1_abs], float(rsi.iat[r1_abs]),
                rsi.index[r2_abs], float(rsi.iat[r2_abs]),
            ))

    if include_hidden:
        # ---------- Hidden Bullish: price HL & RSI LL ----------
        last_r_abs = None
        for i in range(len(p_min) - 1):
            p1, p2 = p_min[i], p_min[i + 1]
            k1 = _nearest_monotonic(r_min, p1, max_lag, last_r_abs)
            if k1 is None:
                continue
            r1_abs = r_min[k1]
            k2 = _nearest_monotonic(r_min, p2, max_lag, r1_abs)
            if k2 is None:
                continue
            r2_abs = r_min[k2]
            last_r_abs = r2_abs

            if _cmp(prices.iat[p2], prices.iat[p1], ">") and _cmp(rsi.iat[r2_abs], rsi.iat[r1_abs], "<"):
                out.append(Divergence(
                    "hidden_bullish",
                    prices.index[p1], float(prices.iat[p1]),
                    prices.index[p2], float(prices.iat[p2]),
                    rsi.index[r1_abs], float(rsi.iat[r1_abs]),
                    rsi.index[r2_abs], float(rsi.iat[r2_abs]),
                ))

        # ---------- Hidden Bearish: price LH & RSI HH ----------
        last_r_abs = None
        for i in range(len(p_max) - 1):
            p1, p2 = p_max[i], p_max[i + 1]
            k1 = _nearest_monotonic(r_max, p1, max_lag, last_r_abs)
            if k1 is None:
                continue
            r1_abs = r_max[k1]
            k2 = _nearest_monotonic(r_max, p2, max_lag, r1_abs)
            if k2 is None:
                continue
            r2_abs = r_max[k2]
            last_r_abs = r2_abs

            if _cmp(prices.iat[p2], prices.iat[p1], "<") and _cmp(rsi.iat[r2_abs], rsi.iat[r1_abs], ">"):
                out.append(Divergence(
                    "hidden_bearish",
                    prices.index[p1], float(prices.iat[p1]),
                    prices.index[p2], float(prices.iat[p2]),
                    rsi.index[r1_abs], float(rsi.iat[r1_abs]),
                    rsi.index[r2_abs], float(rsi.iat[r2_abs]),
                ))

    rows = [asdict(d) for d in out]
    return pd.DataFrame(rows, columns=_SCHEMA)
