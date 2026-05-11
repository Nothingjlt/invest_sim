import pytest
from src.config import SimulationConfig, MarketConfig


def test_default_config_is_valid():
    """Verify the default configuration passes validation."""
    config = SimulationConfig()
    config.validate()


def test_invalid_ages_raises_error():
    """Verify that inconsistent ages raise a ValueError."""
    # Starting age after retirement
    with pytest.raises(ValueError, match="Starting age must be before retirement age."):
        SimulationConfig(starting_age=65, retirement_age=60).validate()

    # Retirement age after end age
    with pytest.raises(ValueError, match="Retirement age must be before end age."):
        SimulationConfig(retirement_age=101, end_age=100).validate()


def test_invalid_market_weights_raises_error():
    """Verify that market weights not summing to 1.0 raise a ValueError."""
    markets = [
        MarketConfig(name="Stock", expected_return=0.07, volatility=0.15, weight=0.5),
        MarketConfig(name="Bond", expected_return=0.03, volatility=0.05, weight=0.4),
    ]
    with pytest.raises(ValueError, match="Total market weights must sum to 1.0."):
        SimulationConfig(markets=markets).validate()
