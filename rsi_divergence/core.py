from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks  # robust peak finder (prominence/width/distance)

# --- RSI (Wilder-like via EWM) ---
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI using exponential smoothing with alpha=1/period and adjust=False.
    Leaves initial NaNs for warmup; caller can choose how to handle them.
    """
    prices = pd.Series(prices, dtype="float64")
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi
    # Pandas EWM behavior & Wilder smoothing α documented in pandas/RSI refs.  :contentReference[oaicite:1]{index=1}


def _find_pivots(
    series: pd.Series,
    prominence: float,
    width: int,
    distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (minima_idx, maxima_idx) using SciPy's find_peaks with robust controls."""
    y = series.values.astype("float64", copy=False)
    maxima, _ = find_peaks(y, prominence=prominence, width=width, distance=distance)
    minima, _ = find_peaks(-y, prominence=prominence, width=width, distance=distance)
    return minima, maxima
    # Prominence/width/distance are the core quality gates for peaks. :contentReference[oaicite:2]{index=2}


def _pair_nearest(left: np.ndarray, right: np.ndarray, max_lag: int) -> List[Tuple[int, int]]:
    """Greedy nearest-neighbor pairing within +/- max_lag bars."""
    pairs: List[Tuple[int, int]] = []
    j = 0
    for i, li in enumerate(left):
        while j < len(right) and right[j] < li - max_lag:
            j += 1
        candidates = []
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(right):
                r = right[k]
                if abs(r - li) <= max_lag:
                    candidates.append((abs(r - li), k))
        if candidates:
            _, kbest = min(candidates, key=lambda t: t[0])
            pairs.append((i, kbest))
    return pairs


@dataclass(frozen=True)
class Divergence:
    kind: str  # "regular_bullish" | "regular_bearish" | "hidden_bullish" | "hidden_bearish"
    p1_idx: pd.Timestamp
    p1_price: float
    p2_idx: pd.Timestamp
    p2_price: float
    r1_idx: pd.Timestamp
    r1_val: float
    r2_idx: pd.Timestamp
    r2_val: float


def find_divergences(
    prices: pd.Series,
    rsi: pd.Series,
    *,
    rsi_period: int = 14,               # NEW: to compute adaptive defaults
    price_prominence: Optional[float] = None,
    rsi_prominence: Optional[float] = None,
    price_width: Optional[int] = None,
    rsi_width: Optional[int] = None,
    distance: Optional[int] = None,     # NEW: pass to find_peaks
    max_lag: int = 3,
    include_hidden: bool = True,
) -> pd.DataFrame:
    """
    Detect regular and (optionally) hidden divergences with windowed pivot pairing.

    Returns a DataFrame with:
    ['kind','p1_idx','p1_price','p2_idx','p2_price','r1_idx','r1_val','r2_idx','r2_val']
    """
    prices = pd.Series(prices, dtype="float64")
    rsi = pd.Series(rsi, dtype="float64").reindex(prices.index)

    # -------- Adaptive defaults --------
    # Price prominence ~ half a volatility unit (scale-invariant, then back to price scale)
    if price_prominence is None:
        vol = prices.pct_change().rolling(rsi_period).std().iloc[-1]
        # guard against NaN or zero in tiny samples
        vol = float(vol) if np.isfinite(vol) and vol > 0 else 0.005  # ~0.5% fallback
        price_prominence = 0.5 * vol * float(prices.iloc[-1])

    if rsi_prominence is None:
        rsi_prominence = 5.0  # RSI points

    if price_width is None:
        price_width = 2

    if rsi_width is None:
        rsi_width = 2

    if distance is None:
        distance = max(1, rsi_period // 2)

    # -------- Pivot detection --------
    p_min, p_max = _find_pivots(prices, price_prominence, price_width, distance)
    r_min, r_max = _find_pivots(rsi,     rsi_prominence,   rsi_width, distance)

    out: List[Divergence] = []

    # Regular Bullish: price LL, RSI HL
    mb = _pair_nearest(p_min, r_min, max_lag)
    for (i_li, i_ri), (j_li, j_ri) in zip(mb[:-1], mb[1:]):
        p1, p2 = p_min[i_li], p_min[j_li]
        r1, r2 = r_min[i_ri], r_min[j_ri]
        if prices.iat[p2] < prices.iat[p1] and rsi.iat[r2] > rsi.iat[r1]:
            out.append(Divergence(
                "regular_bullish",
                prices.index[p1], float(prices.iat[p1]),
                prices.index[p2], float(prices.iat[p2]),
                rsi.index[r1], float(rsi.iat[r1]),
                rsi.index[r2], float(rsi.iat[r2]),
            ))

    # Regular Bearish: price HH, RSI LH
    mb = _pair_nearest(p_max, r_max, max_lag)
    for (i_li, i_ri), (j_li, j_ri) in zip(mb[:-1], mb[1:]):
        p1, p2 = p_max[i_li], p_max[j_li]
        r1, r2 = r_max[i_ri], r_max[j_ri]
        if prices.iat[p2] > prices.iat[p1] and rsi.iat[r2] < rsi.iat[r1]:
            out.append(Divergence(
                "regular_bearish",
                prices.index[p1], float(prices.iat[p1]),
                prices.index[p2], float(prices.iat[p2]),
                rsi.index[r1], float(rsi.iat[r1]),
                rsi.index[r2], float(rsi.iat[r2]),
            ))

    if include_hidden:
        # Hidden Bullish: price HL, RSI LL
        mb = _pair_nearest(p_min, r_min, max_lag)
        for (i_li, i_ri), (j_li, j_ri) in zip(mb[:-1], mb[1:]):
            p1, p2 = p_min[i_li], p_min[j_li]
            r1, r2 = r_min[i_ri], r_min[j_ri]
            if prices.iat[p2] > prices.iat[p1] and rsi.iat[r2] < rsi.iat[r1]:
                out.append(Divergence(
                    "hidden_bullish",
                    prices.index[p1], float(prices.iat[p1]),
                    prices.index[p2], float(prices.iat[p2]),
                    rsi.index[r1], float(rsi.iat[r1]),
                    rsi.index[r2], float(rsi.iat[r2]),
                ))

        # Hidden Bearish: price LH, RSI HH
        mb = _pair_nearest(p_max, r_max, max_lag)
        for (i_li, i_ri), (j_li, j_ri) in zip(mb[:-1], mb[1:]):
            p1, p2 = p_max[i_li], p_max[j_li]
            r1, r2 = r_max[i_ri], r_max[j_ri]
            if prices.iat[p2] < prices.iat[p1] and rsi.iat[r2] > rsi.iat[r1]:
                out.append(Divergence(
                    "hidden_bearish",
                    prices.index[p1], float(prices.iat[p1]),
                    prices.index[p2], float(prices.iat[p2]),
                    rsi.index[r1], float(rsi.iat[r1]),
                    rsi.index[r2], float(rsi.iat[r2]),
                ))

    return pd.DataFrame([asdict(d) for d in out])
