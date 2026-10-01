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


def test_synthetic_returns_never_fall_below_total_loss():
    market_config = MarketConfig(
        name="High Volatility", expected_return=0.05, volatility=1.0, weight=1.0
    )
    market = SyntheticMarket([market_config], seed=42)

    samples = [market.get_annual_returns()["High Volatility"] for _ in range(10000)]

    assert min(samples) >= -1.0


def test_zero_volatility_returns_expected_return_exactly():
    expected_return = 0.07
    market_config = MarketConfig(
        name="Deterministic", expected_return=expected_return, volatility=0.0,
        weight=1.0,
    )
    market = SyntheticMarket([market_config], seed=42)

    samples = [market.get_annual_returns()["Deterministic"] for _ in range(10)]

    assert samples == [expected_return] * 10


@pytest.mark.parametrize(
    ("expected_return", "volatility"),
    [
        (-1.0, 0.0),
        (-1.01, 0.1),
        (float("nan"), 0.1),
        (float("inf"), 0.1),
        (0.05, -0.1),
        (0.05, float("nan")),
        (0.05, float("inf")),
    ],
)
def test_synthetic_market_rejects_invalid_moments(expected_return, volatility):
    market_config = MarketConfig(
        name="Invalid", expected_return=expected_return,
        volatility=volatility, weight=1.0,
    )

    with pytest.raises(ValueError):
        SyntheticMarket([market_config])


def test_synthetic_market_seed_reproduces_sample_sequence():
    market_config = MarketConfig(
        name="Test", expected_return=0.07, volatility=0.15, weight=1.0
    )
    first = SyntheticMarket([market_config], seed=123)
    first_samples = [first.get_annual_returns() for _ in range(10)]

    second = SyntheticMarket([market_config], seed=123)
    second_samples = [second.get_annual_returns() for _ in range(10)]

    assert first_samples == second_samples
