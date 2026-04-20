import pytest
import pandas as pd
from src.market import BootstrapMarket

def test_bootstrap_block_integrity():
    """
    Verify that the bootstrap sampler retrieves contiguous years.
    Using the sample data (1990-2020).
    """
    # Use block_size=5 to test block jumps
    market = BootstrapMarket("data/historical_returns.csv", block_size=5, seed=10)
    market.start_new_path()
    
    # 1. First block
    results = [market.get_annual_returns() for _ in range(5)]
    
    # Check that they were sequential
    # (Note: index 0 = 1990, index 1 = 1991, etc.)
    # The actual starting index depends on the seed.
    
    # If we find 1990, the next must be 1991, then 1992, etc.
    # We can verify this by checking if 'Domestic' returns match the CSV values.
    df = pd.read_csv("data/historical_returns.csv")
    
    # Locate the starting row in the CSV based on the first domestic return
    first_ret = results[0]['Domestic']
    start_idx = df[df['Domestic'] == first_ret].index[0]
    
    for i in range(5):
        expected_idx = (start_idx + i) % len(df)
        assert results[i]['Domestic'] == pytest.approx(df.iloc[expected_idx]['Domestic'])

def test_bootstrap_wrapping():
    """Verify that if we hit the end of the CSV, it wraps to the beginning."""
    market = BootstrapMarket("data/historical_returns.csv", block_size=100)
    market.current_index = 30 # Last row (2020)
    market.remaining_in_block = 10
    
    r1 = market.get_annual_returns() # 2020
    r2 = market.get_annual_returns() # 1990 (wrapped)
    
    df = pd.read_csv("data/historical_returns.csv")
    assert r1['Domestic'] == pytest.approx(df.iloc[30]['Domestic'])
    assert r2['Domestic'] == pytest.approx(df.iloc[0]['Domestic'])
