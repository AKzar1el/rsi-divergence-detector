# tests/test_divergences_unit.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rsi_divergence import calculate_rsi, find_divergences


def _series(vals, name: str = "close") -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="min")
    return pd.Series(vals, index=idx, name=name)


def test_regular_bullish_min_example():
    # Keep the original synthetic PRICE series
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

    # --- Deterministic RSI ---
    # We CONTROL the RSI series: create two RSI minima aligned at indices 7 and 15,
    # with the second (r2) > the first (r1) to enforce HL vs price LL.
    n = len(price)
    rsi_vals = [60.0] * n
    rsi_vals[7]  = 30.0  # first RSI low
    rsi_vals[15] = 36.0  # second RSI low (HIGHER than the first)
    rsi = _series(rsi_vals, "rsi")

    # Gates (same style as the original test)
    price_prom = 0.1
    rsi_prom   = 0.8
    width      = 1
    distance   = 2

    # --- Bonus sanity checks (actionable on failure) ---
    # 1) Show/verify the actual pivots SciPy finds under these gates.
    pmin, _ = find_peaks(-price.values, prominence=price_prom, width=width, distance=distance)
    rmin, _ = find_peaks(-rsi.values,   prominence=rsi_prom,   width=width, distance=distance)
    print("SANITY pmin:", pmin, "rmin:", rmin)

    # We expect to see minima at 7 and 15 for both series.
    assert 7 in pmin and 15 in pmin, f"Expected price minima at 7 and 15; got {pmin.tolist()}"
    assert 7 in rmin and 15 in rmin, f"Expected RSI minima at 7 and 15; got {rmin.tolist()}"

    # 2) Confirm the LL (price) and HL (RSI) semantics on those indices.
    p1, p2 = 7, 15
    r1, r2 = 7, 15
    assert price.iat[p2] < price.iat[p1], (
        f"Price not LL: p1={price.iat[p1]:.4f} (idx {p1}) vs p2={price.iat[p2]:.4f} (idx {p2})"
    )
    assert rsi.iat[r2] > rsi.iat[r1], (
        f"RSI not HL: r1={rsi.iat[r1]:.2f} (idx {r1}) vs r2={rsi.iat[r2]:.2f} (idx {r2})"
    )

    # 3) Make max_lag wide enough for the observed deltas (deterministically zero here).
    first_delta  = abs(p1 - r1)
    second_delta = abs(p2 - r2)
    max_lag = max(3, int(max(first_delta, second_delta) + 1))
    print(f"SANITY Δs: first={first_delta}, second={second_delta}, chosen max_lag={max_lag}")

    # --- Run detection with our controlled RSI ---
    divs = find_divergences(
        price, rsi,
        rsi_period=5,            # irrelevant when we supply RSI explicitly
        price_prominence=price_prom,
        rsi_prominence=rsi_prom,
        price_width=width,
        rsi_width=width,
        distance=distance,
        max_lag=max_lag,
        include_hidden=False,    # we only care about REGULAR bullish here
    )

    kinds = set(divs["kind"]) if not divs.empty else set()
    assert "regular_bullish" in kinds, f"No regular_bullish found; divs=\n{divs}"

def test_regular_bullish_min_clear():
    """
    Deterministic, unambiguous regular-bullish divergence:
    - Price prints LL at idx 19 vs idx 8
    - RSI prints HL at idx 19 vs idx 8 (RSI is supplied explicitly)
    Pairing window is set from observed deltas so alignment cannot fail.
    """
    # ----- PRICE: two clear valleys, second is LOWER (LL) -----
    price_vals = [
        100.0, 98.0, 96.0, 94.0, 92.0, 91.0, 90.5, 90.2, 90.0,   # idx 8  (V1)
        91.5, 93.5, 95.5, 96.5, 95.0, 94.0, 93.0, 92.0, 90.5,
        89.5, 88.0,                                             # idx 19 (V2, lower than V1)
        90.0, 92.0, 95.0, 97.0, 99.0
    ]
    price = _series(price_vals, "close")

    # ----- RSI: two valleys at the SAME bars, second is HIGHER (HL) -----
    # Build a flat RSI and shape clear local minima at 8 and 19.
    base = 55.0
    rsi_vals = [base] * len(price_vals)
    # valley #1 (lower RSI low)
    rsi_vals[7]  = base - 10   # 45
    rsi_vals[8]  = base - 25   # 30  <- r1 (min)
    rsi_vals[9]  = base - 11   # 44
    # valley #2 (higher RSI low)
    rsi_vals[18] = base - 14   # 41
    rsi_vals[19] = base - 19   # 36  <- r2 (min, HIGHER than r1)
    rsi_vals[20] = base - 12   # 43
    rsi = _series(rsi_vals, "rsi")

    # Gates — modest but strict enough to avoid accidental peaks
    price_prom = 0.5   # price units
    rsi_prom   = 2.0   # RSI points
    width      = 1
    distance   = 3

    # ----- Bonus sanity checks: verify the pivots SciPy actually sees -----
    pmin, _ = find_peaks(-price.values, prominence=price_prom, width=width, distance=distance)
    rmin, _ = find_peaks(-rsi.values,   prominence=rsi_prom,   width=width, distance=distance)
    print("SANITY pmin:", pmin, "rmin:", rmin)
    assert 8 in pmin and 19 in pmin,  f"Expected price minima at [8,19]; got {pmin.tolist()}"
    assert 8 in rmin and 19 in rmin,  f"Expected RSI minima at [8,19]; got {rmin.tolist()}"

    # Confirm LL (price) and HL (RSI)
    assert price.iat[19] < price.iat[8], f"Price not LL: {price.iat[8]:.4f} vs {price.iat[19]:.4f}"
    assert rsi.iat[19]   > rsi.iat[8],   f"RSI not HL:  {rsi.iat[8]:.2f} vs {rsi.iat[19]:.2f}"

    # Pairing budget: use observed deltas to avoid alignment failures
    first_delta  = abs(8 - 8)
    second_delta = abs(19 - 19)
    max_lag = max(3, int(max(first_delta, second_delta) + 1))
    print(f"SANITY Δs: first={first_delta}, second={second_delta}, chosen max_lag={max_lag}")

    # ----- Run detector (hidden disabled; we only care about regular-bullish) -----
    divs = find_divergences(
        prices=price,
        rsi=rsi,
        rsi_period=14,  # irrelevant when RSI supplied
        price_prominence=price_prom,
        rsi_prominence=rsi_prom,
        price_width=width,
        rsi_width=width,
        distance=distance,
        max_lag=max_lag,
        include_hidden=False,
    )

    kinds = set(divs["kind"]) if not divs.empty else set()
    assert "regular_bullish" in kinds, f"No regular_bullish found; divs=\n{divs}"


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
