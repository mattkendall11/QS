import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('btc_orderbook_full_20250421_154425_final.csv')
midprice = df['midprice']
time = df['timestamp']


def trends(prices, threshold=0.002):
    """
    Calculate price movements for the next 5 time steps for each price in the array.

    For each price p at time t:
    - 0 (Down): if future price p' < p * (1 - threshold)
    - 1 (Stable): if p * (1 - threshold) ≤ p' ≤ p * (1 + threshold)
    - 2 (Up): if future price p' > p * (1 + threshold)

    Args:
        prices (numpy.ndarray): Array of prices
        threshold (float): Threshold for price movement classification (default: 0.002)

    Returns:
        numpy.ndarray: Array of shape (len(prices)-5, 5) with price movement classifications
    """
    if len(prices) <= 5:
        return np.array([])

    # Number of valid data points (excluding the last 5 that don't have enough future data)
    n = len(prices) - 5
    prices = np.array(prices.values)
    # Initialize result array
    result = np.zeros((n, 5), dtype=np.int8)

    # Calculate upper and lower bounds for each price
    upper_bounds = prices[:n] * (1 + threshold)
    lower_bounds = prices[:n] * (1 - threshold)

    # Calculate movements for each horizon (1 to 5)
    for k in range(1, 6):
        # Get future prices at horizon k
        future_prices = prices[k:k + n]

        # Classify movements
        # Down (0): future price < lower bound
        down_mask = future_prices < lower_bounds
        result[:, k - 1][down_mask] = 0

        # Up (2): future price > upper bound
        up_mask = future_prices > upper_bounds
        result[:, k - 1][up_mask] = 2

        # Stable (1): lower bound ≤ future price ≤ upper bound
        # This is the default (zeros were initialized), so we only need to set where it's 1
        stable_mask = ~(down_mask | up_mask)
        result[:, k - 1][stable_mask] = 1

    return result

ments = trends(midprice)
print(ments)
