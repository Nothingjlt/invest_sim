import statistics
import pytest
from src.config import MarketConfig
from src.market import SyntheticMarket


def test_market_return_convergence():
    """
    Verify that the mean of generated returns matches the expected return.
    (Law of Large Numbers)
    """
    expected_mean = 0.07
    volatility = 0.15
    market_config = MarketConfig(
        name="Test", expected_return=expected_mean, volatility=volatility, weight=1.0
    )
    market = SyntheticMarket([market_config], seed=42)

    num_samples = 100000
    samples = [market.get_annual_returns()["Test"] for _ in range(num_samples)]

    actual_mean = statistics.mean(samples)
    actual_std = statistics.stdev(samples)

    # Verify mean is within ~0.001 of expected (given large sample)
    assert actual_mean == pytest.approx(expected_mean, abs=1e-3)
    # Verify standard deviation matches input volatility
    assert actual_std == pytest.approx(volatility, abs=1e-3)
