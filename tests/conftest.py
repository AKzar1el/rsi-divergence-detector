import numpy as np
import pandas as pd
import pytest

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)

@pytest.fixture
def close_series(rng):
    # Small synthetic “price”: random walk with a couple of shaped segments
    n = 180
    steps = rng.normal(0, 0.4, size=n)
    # Add a few shaped moves to coax pivots
    steps[20:35] += -1.0
    steps[35:50] += +0.7
    steps[80:95] += +0.9
    steps[95:110] += -0.8
    close = 100 + steps.cumsum()
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    return pd.Series(close, index=idx, name="close")

@pytest.fixture
def ohlc(close_series, rng):
    o = close_series.shift(1).fillna(close_series.iloc[0])
    body = (close_series - o).abs()
    wiggle = (0.25 * body + 0.05).clip(lower=0.02)
    h = np.maximum(o, close_series) + (wiggle + rng.uniform(0.0, 0.05, len(close_series)))
    l = np.minimum(o, close_series) - (wiggle + rng.uniform(0.0, 0.05, len(close_series)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close_series}, index=close_series.index)
