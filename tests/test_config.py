import pytest
from src.config import SimulationConfig, MarketConfig


def test_default_config_is_valid():
    """Verify the default configuration passes validation."""
    config = SimulationConfig()
    config.validate()
    assert all(market.weight is None for market in config.markets)


def test_world_factory_is_a_valid_unallocated_investment_universe():
    markets = SimulationConfig.get_world_market_configs()
    assert [market.name for market in markets] == [
        "USA", "GBR", "JPN", "DEU", "FRA", "CAN", "AUS",
        "CHN", "IND", "BRA", "ZAF", "ARE", "Bonds", "Bills",
    ]
    assert all(market.weight is None for market in markets)
    SimulationConfig(markets=markets).validate()
    assert all(market.weight is None for market in markets)


def test_paper_bootstrap_factory_is_valid_without_an_allocation():
    config = SimulationConfig.get_paper_bootstrap_config()
    config.validate()
    assert all(market.weight is None for market in config.markets)


def test_descriptor_only_configuration_needs_no_weight_placeholders():
    markets = [MarketConfig("Stock", 0.07, 0.15), MarketConfig("Bond", 0.03, 0.05)]
    SimulationConfig(markets=markets).validate()
    assert all(market.weight is None for market in markets)


def test_explicit_none_weights_are_descriptors():
    SimulationConfig(markets=[MarketConfig("Stock", 0.07, 0.15, weight=None)]).validate()


def test_legacy_positional_and_keyword_weights_remain_valid_metadata():
    markets = [
        MarketConfig("Stock", 0.07, 0.15, 0.6),
        MarketConfig("Bond", 0.03, 0.05, weight=0.4),
    ]
    SimulationConfig(markets=markets).validate()
    assert [market.weight for market in markets] == [0.6, 0.4]


@pytest.mark.parametrize("explicit_weight", [0.0, 1.0])
def test_mixed_descriptors_and_legacy_weights_require_an_explicit_choice(explicit_weight):
    markets = [MarketConfig("Stock", 0.07, 0.15, explicit_weight), MarketConfig("Bond", 0.03, 0.05)]
    with pytest.raises(ValueError, match="omitted for all descriptors.*supplied for every asset"):
        SimulationConfig(markets=markets).validate()


def test_explicit_all_zero_weights_are_not_a_valid_portfolio():
    markets = [MarketConfig("Stock", 0.07, 0.15, 0.0), MarketConfig("Bond", 0.03, 0.05, 0.0)]
    with pytest.raises(ValueError, match="Total market weights must sum to 1.0"):
        SimulationConfig(markets=markets).validate()


@pytest.mark.parametrize("field,value", [
    ("name", ""), ("expected_return", -1), ("expected_return", float("nan")),
    ("volatility", -0.1), ("volatility", float("inf")),
])
def test_descriptors_still_validate_asset_names_and_return_assumptions(field, value):
    market = MarketConfig("Stock", 0.07, 0.15)
    setattr(market, field, value)
    with pytest.raises(ValueError):
        SimulationConfig(markets=[market]).validate()


def test_descriptor_only_configuration_rejects_duplicate_assets():
    with pytest.raises(ValueError, match="Duplicate asset names"):
        SimulationConfig(markets=[MarketConfig("Stock", 0, 0), MarketConfig("Stock", 0, 0)]).validate()


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


def test_decumulation_granularity_valid_values():
    """Both 'annual' and 'monthly' must pass validation."""
    for granularity in ("annual", "monthly"):
        config = SimulationConfig(decumulation_granularity=granularity)
        config.validate()  # should not raise


def test_decumulation_granularity_invalid_value_raises_error():
    """Any value other than 'annual' or 'monthly' must raise ValueError."""
    with pytest.raises(ValueError, match="decumulation_granularity must be"):
        SimulationConfig(decumulation_granularity="weekly").validate()

    with pytest.raises(ValueError, match="decumulation_granularity must be"):
        SimulationConfig(decumulation_granularity="").validate()


@pytest.mark.parametrize("field,value", [
    ("initial_salary", -1), ("social_security_benefit", -1),
    ("withdrawal_cap", -1), ("withdrawal_floor", -1),
    ("savings_rate", -0.1), ("savings_rate", 1.1),
    ("withdrawal_rate", -0.1), ("withdrawal_rate", 1.1),
    ("salary_growth_rate", -1.1), ("withdrawal_strategy", "unknown"),
    ("starting_age", 0), ("starting_age", 25.5), ("end_age", True),
    ("enable_mortality", "false"), ("withdrawal_inflation_adjusted", 1),
    ("withdrawal_cap_inflation_adjusted", None), ("withdrawal_floor_inflation_adjusted", "yes"),
])
def test_config_rejects_invalid_numeric_rates_and_modes(field, value):
    with pytest.raises(ValueError):
        SimulationConfig(**{field: value}).validate()


@pytest.mark.parametrize("field", [
    "initial_salary", "salary_growth_rate", "savings_rate", "withdrawal_rate",
    "social_security_benefit", "withdrawal_cap", "withdrawal_floor",
    "starting_age", "retirement_age", "end_age",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_config_rejects_nonfinite_and_nonnumeric_fields(field, value):
    with pytest.raises(ValueError):
        SimulationConfig(**{field: value}).validate()


@pytest.mark.parametrize("field,value", [
    ("expected_return", -1), ("expected_return", float("nan")),
    ("volatility", -0.1), ("volatility", float("inf")),
    ("weight", float("nan")), ("weight", -0.1), ("name", ""),
])
def test_config_validates_individual_market_fields(field, value):
    market = MarketConfig("A", 0.0, 0.0, 1.0)
    setattr(market, field, value)
    with pytest.raises(ValueError):
        SimulationConfig(markets=[market]).validate()


def test_config_rejects_signed_weights_even_when_sum_is_one():
    with pytest.raises(ValueError):
        SimulationConfig(markets=[
            MarketConfig("A", 0, 0, -0.1), MarketConfig("B", 0, 0, 1.1),
        ]).validate()


def test_config_rejects_duplicate_labels_before_collapsing_weights():
    with pytest.raises(ValueError, match="Duplicate asset names"):
        SimulationConfig(markets=[
            MarketConfig("A", 0, 0, 0), MarketConfig("A", 0, 0, 1),
        ]).validate()


def test_config_accepts_rounding_error_zero_income_and_negative_growth():
    SimulationConfig(
        initial_salary=0, social_security_benefit=0, salary_growth_rate=-0.1,
        savings_rate=1, withdrawal_rate=0, withdrawal_strategy="fixed_real",
        withdrawal_floor=0, withdrawal_cap=0,
        markets=[MarketConfig("A", -0.1, 0, 1 + 5e-7)],
    ).validate()


@pytest.mark.parametrize("field,value", [
    ("markets", None), ("markets", []), ("markets", ["A"]),
    ("withdrawal_strategy", []), ("decumulation_granularity", []),
    ("markets", [MarketConfig([], 0, 0, 1)]),
])
def test_malformed_config_containers_raise_clear_validation_error(field, value):
    with pytest.raises(ValueError):
        SimulationConfig(**{field: value}).validate()
