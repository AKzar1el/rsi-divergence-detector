# tests/test_divergences_unit.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rsi_divergence import calculate_rsi, find_divergences


def _series(vals, name: str = "close") -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="min")
    return pd.Series(vals, index=idx, name=name)


def test_regular_bullish_min_example():
    vals = [
        98.53182173213305, 97.16401343642414, 95.42104131059811, 94.21368201574103,
        93.10955859561426, 93.77699231876348, 94.9141119322064,  94.6878330303028,
        94.75808271842814, 96.98923572986341, 96.14652798341596, 95.96149659609323,
        95.79401610280505, 95.38235318148409, 94.81973382373931, 94.48029570493402,
        95.07924006254096, 95.58704406082344, 96.0869363250924,  96.91210652631644,
        97.33705359236093, 97.62700853456847, 98.1535268901027,  97.8724200884738,
        97.60965032034908,
    ]
    price = _series(vals, "close")
    rsi = calculate_rsi(price, period=5)

    divs = find_divergences(
        price, rsi,
        rsi_period=5,
        price_prominence=0.1,
        rsi_prominence=0.8,
        price_width=1,
        rsi_width=1,
        distance=2,
        max_lag=3,
        include_hidden=False,
    )
    kinds = set(divs["kind"]) if not divs.empty else set()
    assert "regular_bullish" in kinds


def test_hidden_bearish_max_example():
    # PRICE: first top higher than second (LH)
    # Make the first leg reach ~105.0 (i goes to 10), then ensure H2 < 105.0
    price_vals = (
        [100 + 0.5 * i for i in range(11)]  # up to 105.0 at idx ~10  (H1)
        + [104.1, 103.2, 102.9]             # pullback
        + [103.9, 104.6, 104.8]             # H2 = 104.8 (< 105.0)
        + [104.1, 103.7, 103.0, 102.6, 102.2]
    )
    price = _series(price_vals, "close")

    # RSI: second top higher than first (HH), roughly aligned with price tops
    rsi_vals = (
        [50, 52, 54, 55, 56, 57, 58, 59, 60, 59, 58]  # first RSI top ~60 near idx ~8–10
        + [48, 46, 45]                                 # valley
        + [55, 62, 72]                                 # second RSI top ~72 (HH)
        + [66, 60, 55, 52, 50]
    )
    rsi = _series(rsi_vals, "rsi")

    # Gates — modest for small synthetic data
    price_prom = 0.25
    rsi_prom   = 1.0
    width      = 1
    distance   = 2

    # Debug: check we truly have two tops in both series under the same gates
    pmax, _ = find_peaks(price.values, prominence=price_prom, width=width, distance=distance)
    rmax, _ = find_peaks(rsi.values,   prominence=rsi_prom,   width=width, distance=distance)
    print("pmax:", pmax, "rmax:", rmax, "first Δ:", (abs(pmax[0]-rmax[0]) if (len(pmax) and len(rmax)) else "n/a"))

    # Make pairing window wide enough for observed deltas
    first_delta  = abs(pmax[0] - rmax[0]) if (len(pmax) >= 1 and len(rmax) >= 1) else 3
    second_delta = abs(pmax[1] - rmax[1]) if (len(pmax) >= 2 and len(rmax) >= 2) else first_delta
    max_lag = max(3, int(max(first_delta, second_delta) + 1))

    divs = find_divergences(
        prices=price,
        rsi=rsi,
        rsi_period=14,  # irrelevant when supplying RSI
        price_prominence=price_prom,
        rsi_prominence=rsi_prom,
        price_width=width,
        rsi_width=width,
        distance=distance,
        max_lag=max_lag,
        include_hidden=True,
    )

    if divs.empty:
        # Make failures actionable
        if len(pmax) >= 2 and len(rmax) >= 2:
            p1, p2 = pmax[0], pmax[1]
            r1, r2 = rmax[0], rmax[1]
            print(
                f"price[p1]={price.iat[p1]:.2f}, price[p2]={price.iat[p2]:.2f} (want LH: p2<p1)\n"
                f"rsi[r1]={rsi.iat[r1]:.2f},   rsi[r2]={rsi.iat[r2]:.2f} (want HH: r2>r1)\n"
                f"Δs: first={first_delta}, second={second_delta}, max_lag={max_lag}"
            )
        else:
            print(f"Not enough peaks. len(pmax)={len(pmax)}, len(rmax)={len(rmax)}")
            print("Loosen prominence/width/distance or widen max_lag.")

    assert not divs.empty, "No divergences; check printed peaks/Δ and price tops ordering."
    assert "hidden_bearish" in set(divs["kind"])
