"""Core logic for RSI divergence detection."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use Exponential Moving Average for RSI calculation
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Fill initial NaNs with a neutral RSI value of 50
    return rsi.fillna(50)


def find_extrema(data: pd.Series, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Finds local minima and maxima in a series."""
    minima = argrelextrema(data.values, np.less_equal, order=order)[0]
    maxima = argrelextrema(data.values, np.greater_equal, order=order)[0]
    return minima, maxima


def find_divergences(
    prices: pd.Series, rsi: pd.Series, order: int = 5
) -> List[Tuple[list, str, list]]:
    """Finds RSI divergences and returns pairs of points for each."""
    divergences = []
    price_minima_idx, price_maxima_idx = find_extrema(prices, order)
    rsi_minima_idx, rsi_maxima_idx = find_extrema(rsi, order)

    # Regular Bullish: lower low in price, higher low in RSI
    min_pairs = np.intersect1d(price_minima_idx, rsi_minima_idx)
    for i in range(len(min_pairs) - 1):
        p_low1_idx, p_low2_idx = min_pairs[i], min_pairs[i+1]
        if prices.iloc[p_low2_idx] < prices.iloc[p_low1_idx] and rsi.iloc[p_low2_idx] > rsi.iloc[p_low1_idx]:
            price_point1 = (prices.index[p_low1_idx], prices.iloc[p_low1_idx])
            price_point2 = (prices.index[p_low2_idx], prices.iloc[p_low2_idx])
            rsi_point1 = (rsi.index[p_low1_idx], rsi.iloc[p_low1_idx])
            rsi_point2 = (rsi.index[p_low2_idx], rsi.iloc[p_low2_idx])
            divergences.append(
                ([price_point1, price_point2], "regular_bullish", [rsi_point1, rsi_point2])
            )

    # Regular Bearish: higher high in price, lower high in RSI
    max_pairs = np.intersect1d(price_maxima_idx, rsi_maxima_idx)
    for i in range(len(max_pairs) - 1):
        p_high1_idx, p_high2_idx = max_pairs[i], max_pairs[i+1]
        if prices.iloc[p_high2_idx] > prices.iloc[p_high1_idx] and rsi.iloc[p_high2_idx] < rsi.iloc[p_high1_idx]:
            price_point1 = (prices.index[p_high1_idx], prices.iloc[p_high1_idx])
            price_point2 = (prices.index[p_high2_idx], prices.iloc[p_high2_idx])
            rsi_point1 = (rsi.index[p_high1_idx], rsi.iloc[p_high1_idx])
            rsi_point2 = (rsi.index[p_high2_idx], rsi.iloc[p_high2_idx])
            divergences.append(
                ([price_point1, price_point2], "regular_bearish", [rsi_point1, rsi_point2])
            )

    return divergences
