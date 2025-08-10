"""Tests for the core logic."""

import numpy as np
import pandas as pd
import pytest

import os
import matplotlib.pyplot as plt
from rsi_divergence.core import calculate_rsi, find_divergences, find_extrema


def test_no_divergence():
    """Test that no divergence is found in a simple trendless series."""
    # Create a simple, trendless price series
    # Create a perfectly linear, upward-trending price series.
    # This should not have any divergences.
    prices = pd.Series(
        np.linspace(50, 100, 100),
        index=pd.to_datetime(pd.date_range("2023-01-01", periods=100))
    )
    rsi = calculate_rsi(prices)

    # Find divergences
    # Increase order to avoid picking up noise in the smooth sine wave
    divergences = find_divergences(prices, rsi, order=10)

    # Expect no divergences
    assert len(divergences) == 0


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

    # The function expects price and rsi series, not a dataframe.
    # The function signature was also incorrect in the test.
    divergences = find_divergences(ohlc['close'], ohlc['rsi'])

    # Check that at least one divergence was found
    assert len(divergences) > 0

    # Check that a regular bullish divergence is in the results
    assert any(d[1] == 'regular_bullish' for d in divergences)


def test_multiple_bullish_divergences():
    """Tests for multiple regular bullish divergences using synthetic data."""
    # 1. Generate Synthetic Data with three bullish divergences
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=300))
    # Trough 1
    p1 = np.linspace(100, 80, 50)
    p2 = np.linspace(80, 90, 20)
    # Trough 2 (first divergence)
    p3 = np.linspace(90, 70, 50)
    p4 = np.linspace(70, 85, 20)
    # Trough 3 (second divergence)
    p5 = np.linspace(85, 60, 50)
    p6 = np.linspace(60, 80, 20)
    # Trough 4 (third divergence)
    p7 = np.linspace(80, 50, 50)
    p8 = np.linspace(50, 75, 40)
    price_data = np.concatenate([p1, p2, p3, p4, p5, p6, p7, p8])
    price = pd.Series(price_data, index=dates)

    # Calculate RSI from the price data
    rsi = calculate_rsi(price)

    # 2. Find Divergences
    divergences = find_divergences(price, rsi, order=5)

    # 3. Assert at least three bullish divergences were found
    bullish_divergences = [d for d in divergences if d[1] == 'regular_bullish']
    assert len(bullish_divergences) >= 3


def test_regular_bearish_divergence():
    """Tests for a regular bearish divergence using synthetic data."""
    # 1. Generate Synthetic Data for a bearish divergence
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
    # Create a price series that rallies to a new high with less momentum
    p1 = np.linspace(50, 70, 40)
    p2 = np.linspace(70, 60, 10)
    p3 = np.linspace(60, 80, 40)  # Higher high
    p4 = np.linspace(80, 65, 10)  # Pullback
    price_data = np.concatenate([p1, p2, p3, p4])
    price = pd.Series(price_data, index=dates)

    # Calculate RSI
    rsi = calculate_rsi(price)

    # 2. Find Divergences
    divergences = find_divergences(price, rsi, order=5)

    # 3. Assert a bearish divergence was found
    bearish_divergences = [d for d in divergences if d[1] == 'regular_bearish']
    assert len(bearish_divergences) > 0
