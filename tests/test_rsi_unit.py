import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from rsi_divergence import calculate_rsi, wilder_rsi

def test_calculate_rsi_contract(close_series):
    rsi = calculate_rsi(close_series, period=14)
    # index alignment
    assert rsi.index.equals(close_series.index)
    # range check (ignore initial NaNs)
    vals = rsi.dropna().values
    assert np.nanmin(vals) >= 0.0 - 1e-9
    assert np.nanmax(vals) <= 100.0 + 1e-9

def test_calculate_rsi_errors():
    s = pd.Series([], dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        calculate_rsi(s, period=14)
    with pytest.raises(ValueError):
        calculate_rsi(pd.Series([1,2,3], index=[3,2,1]), period=14)  # non-monotonic index
    with pytest.raises(ValueError):
        calculate_rsi(pd.Series([1,2,3], index=pd.date_range("2024-01-01", periods=3)), period=1)

def test_flat_series_gives_neutralish_rsi():
    idx = pd.date_range("2024-01-01", periods=60, freq="T")
    flat = pd.Series(100.0, index=idx)
    rsi = calculate_rsi(flat, period=14)
    # After warmup, RSI should hover ~50 for flat price
    tail = rsi.iloc[20:].dropna()
    assert np.allclose(tail, 50.0, atol=1.0)  # allow tolerance due to smoothing

def test_wilder_vs_calculate_rsi_consistency(close_series):
    period = 14
    rsi_series = calculate_rsi(close_series, period=period)
    rsi_np = wilder_rsi(close_series.to_numpy(dtype=float), period=period)
    # Compare after warmup; methods differ slightly, so use tolerances
    a = rsi_series.values[period:]
    b = rsi_np[period:]
    mask = ~np.isnan(a)
    assert np.allclose(a[mask], b[mask], rtol=1e-2, atol=1e-1)
