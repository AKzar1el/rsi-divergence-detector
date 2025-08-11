from collections import Counter
import numpy as np
import pandas as pd

from rsi_divergence import calculate_rsi, find_divergences

def synth_close():
    rng = np.random.default_rng(7)
    n = 300
    steps = rng.normal(0, 0.35, size=n)
    steps[30:50]  += -0.9
    steps[50:70]  += +0.7
    steps[120:140]+= +0.8
    steps[140:165]+= -0.7
    price = 100 + steps.cumsum()
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    return pd.Series(price, index=idx, name="close")

def test_scenario_detects_some_divergences():
    close = synth_close()
    rsi = calculate_rsi(close, period=14)
    divs = find_divergences(
        close, rsi,
        rsi_period=14,
        # Let adaptive defaults compute prominence/width/distance
        max_lag=8, include_hidden=True,
    )
    # Invariants: correct columns; kinds limited; at least a few rows
    cols = {'kind','p1_idx','p1_price','p2_idx','p2_price','r1_idx','r1_val','r2_idx','r2_val'}
    assert set(divs.columns) == cols
    assert all(k in {"regular_bullish","regular_bearish","hidden_bullish","hidden_bearish"} for k in divs["kind"])
    assert len(divs) >= 1  # not brittle on exact count
