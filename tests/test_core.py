"""Tests for the core logic."""

import pytest
import pandas as pd
import numpy as np
from rsi_divergence.core import find_divergences

def test_regular_bullish_divergence():
    """Test that a regular bullish divergence is detected."""
    # Create synthetic data with a regular bullish divergence
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
    price = np.sin(np.linspace(0, 10, 100)) * 5 + 50
    price[20:40] = price[20:40] - 5  # First trough
    price[70:90] = price[70:90] - 10 # Lower second trough

    ohlc = pd.DataFrame({'close': price}, index=dates)
    
    # Manually create a higher trough in RSI
    ohlc['rsi'] = 50 + np.sin(np.linspace(0, 10, 100)) * 10
    ohlc.loc[ohlc.index[20:40], 'rsi'] = ohlc['rsi'][20:40] - 10 # First trough
    ohlc.loc[ohlc.index[70:90], 'rsi'] = ohlc['rsi'][70:90] - 5  # Higher second trough


    divergences = find_divergences(ohlc, rsi_period=14, price_prominence=1, rsi_prominence=1)

    assert not divergences.empty
    assert 'Regular Bullish' in divergences['type'].values
