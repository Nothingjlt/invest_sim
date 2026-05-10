import pytest
import pandas as pd
from src.market import BootstrapMarket

def test_bootstrap_block_integrity():
    """
    Verify that the bootstrap sampler retrieves contiguous years.
    Using the global historical data.
    """
    csv_path = "data/global_historical_returns.csv"
    # Use block_size=5 to test block jumps
    market = BootstrapMarket(csv_path, block_size=5, seed=10)
    market.start_new_path()
    
    # 1. First block
    results = [market.get_annual_returns() for _ in range(5)]
    
    # Check that they were sequential
    df = pd.read_csv(csv_path)
    
    # Locate the starting row in the CSV based on the first USA return
    # Use approx comparison to find the row
    first_ret = results[0]['USA']
    start_idx = df[df['USA'] == first_ret].index[0]
    
    for i in range(5):
        expected_idx = (start_idx + i) % len(df)
        assert results[i]['USA'] == pytest.approx(df.iloc[expected_idx]['USA'])

def test_bootstrap_wrapping():
    """Verify that if we hit the end of the CSV, it wraps to the beginning."""
    csv_path = "data/global_historical_returns.csv"
    df = pd.read_csv(csv_path)
    last_idx = len(df) - 1
    
    market = BootstrapMarket(csv_path, block_size=100)
    market.current_index = last_idx # Last row
    market.remaining_in_block = 10
    
    r1 = market.get_annual_returns() # Last row
    r2 = market.get_annual_returns() # First row (wrapped)
    
    assert r1['USA'] == pytest.approx(df.iloc[last_idx]['USA'])
    assert r2['USA'] == pytest.approx(df.iloc[0]['USA'])
