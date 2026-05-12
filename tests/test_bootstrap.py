import pytest
import pandas as pd
from src.config import MarketConfig, SimulationConfig
from src.market import BootstrapMarket
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy


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
    first_ret = results[0]["USA"]
    start_idx = df[df["USA"] == first_ret].index[0]

    for i in range(5):
        expected_idx = (start_idx + i) % len(df)
        assert results[i]["USA"] == pytest.approx(df.iloc[expected_idx]["USA"])


def test_bootstrap_wrapping():
    """Verify that if we hit the end of the CSV, it wraps to the beginning."""
    csv_path = "data/global_historical_returns.csv"
    df = pd.read_csv(csv_path)
    last_idx = len(df) - 1

    market = BootstrapMarket(csv_path, block_size=100)
    market.current_index = last_idx  # Last row
    market.remaining_in_block = 10

    r1 = market.get_annual_returns()  # Last row
    r2 = market.get_annual_returns()  # First row (wrapped)

    assert r1["USA"] == pytest.approx(df.iloc[last_idx]["USA"])
    assert r2["USA"] == pytest.approx(df.iloc[0]["USA"])


class CountingBootstrapMarket(BootstrapMarket):
    def __init__(self, csv_path, block_size=1, seed=None):
        super().__init__(csv_path, block_size, seed)
        self.start_calls = 0

    def start_new_path(self):
        self.start_calls += 1
        super().start_new_path()


def test_bootstrap_market_starts_new_path_every_trial():
    csv_path = "data/global_historical_returns.csv"
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=26,
        markets=[MarketConfig("USA", 0.0, 0.0, 1.0)],
    )
    sim = Simulator(config)
    market = CountingBootstrapMarket(csv_path, block_size=1, seed=42)
    strategy = FixedAllocationStrategy({"USA": 1.0})
    sim.run_stochastic(strategy, num_trials=5, market_engine=market)

    assert market.start_calls == 5
