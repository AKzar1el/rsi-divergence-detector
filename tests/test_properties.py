import numpy as np
import pandas as pd
import hypothesis.strategies as st
from hypothesis import given, settings

from rsi_divergence import calculate_rsi, find_divergences

@st.composite
def price_series(draw):
    n = draw(st.integers(min_value=40, max_value=250))
    base = draw(st.floats(min_value=50.0, max_value=150.0))
    # random walk with bounded step sizes
    steps = draw(st.lists(st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
                          min_size=n, max_size=n))
    arr = np.cumsum(np.array(steps, dtype=float)) + float(base)
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    return pd.Series(arr, index=idx, name="close")

@given(s=price_series())
@settings(deadline=None, max_examples=30)
def test_detector_never_crashes_and_has_valid_schema(s):
    rsi = calculate_rsi(s, period=14)
    divs = find_divergences(s, rsi, rsi_period=14, max_lag=8, include_hidden=True)
    assert set(divs.columns) == {
        "kind","p1_idx","p1_price","p2_idx","p2_price","r1_idx","r1_val","r2_idx","r2_val"
    }
    assert all(k in {"regular_bullish","regular_bearish","hidden_bullish","hidden_bearish"} for k in divs["kind"])
