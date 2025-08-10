"""Core logic for RSI divergence detection."""

import pandas as pd
import numpy as np
from .utils import wilder_rsi, find_extrema

def find_divergences(
    ohlc: pd.DataFrame,
    rsi_period: int = 14,
    price_prominence: int = 1,
    rsi_prominence: int = 1,
    price_width: int = 1,
    rsi_width: int = 1,
) -> pd.DataFrame:
    """
    Finds regular and hidden RSI divergences.

    Args:
        ohlc: DataFrame with a 'close' column and a DatetimeIndex.
        rsi_period: Lookback period for RSI.
        price_prominence: Prominence for price peak detection.
        rsi_prominence: Prominence for RSI peak detection.
        price_width: Width for price peak detection.
        rsi_width: Width for RSI peak detection.

    Returns:
        A DataFrame containing detected divergences.
    """
    close_prices = ohlc['close'].to_numpy()
    rsi_values = wilder_rsi(close_prices, period=rsi_period)
    ohlc['rsi'] = rsi_values

    price_minima, price_maxima = find_extrema(close_prices, prominence=price_prominence, width=price_width)
    rsi_minima, rsi_maxima = find_extrema(rsi_values, prominence=rsi_prominence, width=rsi_width)

    divergences = []

    # Align extrema - this is a simplification. A more robust solution would search for the nearest corresponding extremum.
    aligned_maxima = np.intersect1d(price_maxima, rsi_maxima)
    aligned_minima = np.intersect1d(price_minima, rsi_minima)

    # Regular Bullish: Price Lower Lows, RSI Higher Lows
    for i in range(len(aligned_minima) - 1):
        start_idx, end_idx = aligned_minima[i], aligned_minima[i+1]
        if close_prices[end_idx] < close_prices[start_idx] and rsi_values[end_idx] > rsi_values[start_idx]:
            divergences.append({
                'type': 'Regular Bullish',
                'start_date': ohlc.index[start_idx],
                'end_date': ohlc.index[end_idx],
            })

    # Hidden Bullish: Price Higher Lows, RSI Lower Lows
    for i in range(len(aligned_minima) - 1):
        start_idx, end_idx = aligned_minima[i], aligned_minima[i+1]
        if close_prices[end_idx] > close_prices[start_idx] and rsi_values[end_idx] < rsi_values[start_idx]:
            divergences.append({
                'type': 'Hidden Bullish',
                'start_date': ohlc.index[start_idx],
                'end_date': ohlc.index[end_idx],
            })

    # Regular Bearish: Price Higher Highs, RSI Lower Highs
    for i in range(len(aligned_maxima) - 1):
        start_idx, end_idx = aligned_maxima[i], aligned_maxima[i+1]
        if close_prices[end_idx] > close_prices[start_idx] and rsi_values[end_idx] < rsi_values[start_idx]:
            divergences.append({
                'type': 'Regular Bearish',
                'start_date': ohlc.index[start_idx],
                'end_date': ohlc.index[end_idx],
            })

    # Hidden Bearish: Price Lower Highs, RSI Higher Highs
    for i in range(len(aligned_maxima) - 1):
        start_idx, end_idx = aligned_maxima[i], aligned_maxima[i+1]
        if close_prices[end_idx] < close_prices[start_idx] and rsi_values[end_idx] > rsi_values[start_idx]:
            divergences.append({
                'type': 'Hidden Bearish',
                'start_date': ohlc.index[start_idx],
                'end_date': ohlc.index[end_idx],
            })

    return pd.DataFrame(divergences)
