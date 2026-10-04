from dataclasses import replace

import pytest

from src.assets import DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS
from src.config import MarketConfig, SimulationConfig
from src.investor import Investor
from src.market import Market
from src.simulator import Simulator
from src.strategy import (
    FixedAllocationStrategy,
    PaperOptimalStrategy,
    PaperTDFStrategy,
    Strategy,
)


class ConstantMarket(Market):
    def __init__(self, returns):
        self.returns = returns

    def get_annual_returns(self):
        return dict(self.returns)


def short_config(**changes):
    values = dict(
        starting_age=25,
        retirement_age=28,
        end_age=29,
        initial_salary=1000,
        salary_growth_rate=0,
        savings_rate=1,
        withdrawal_rate=0,
        markets=[
            replace(m, expected_return=0.5, volatility=0)
            for m in SimulationConfig.get_paper_market_configs()
        ],
    )
    values.update(changes)
    return SimulationConfig(**values)


@pytest.mark.parametrize("strategy_type", [PaperOptimalStrategy, PaperTDFStrategy])
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_default_paper_strategies_receive_equity_returns(strategy_type, granularity):
    result = Simulator(short_config(decumulation_granularity=granularity)).run_stochastic(
        strategy_type(), num_trials=1, track_paths=True
    )
    assert result.paths[0] == pytest.approx([0, 1000, 2500, 4750, 7125])


@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_unknown_allocation_fails_before_first_contribution(granularity):
    with pytest.raises(ValueError, match="FixedAllocationStrategy at age 25:.*USA"):
        Simulator(short_config(decumulation_granularity=granularity)).run_stochastic(
            FixedAllocationStrategy({"USA": 1}), num_trials=1
        )


@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_assets_introduced_at_retirement_are_validated(granularity):
    class MissingAtRetirementStrategy(Strategy):
        def _get_allocation(self, age):
            if age < 26:
                return {DOMESTIC_STOCK: 1.0}
            return {"Missing Retirement Asset": 1.0}

    config = short_config(
        retirement_age=26,
        end_age=27,
        decumulation_granularity=granularity,
        markets=[
            MarketConfig(DOMESTIC_STOCK, 0, 0, 0.34),
            MarketConfig(INTERNATIONAL_STOCK, 0, 0, 0.66),
        ],
    )
    message = "MissingAtRetirementStrategy at age 26:.*Missing Retirement Asset"
    with pytest.raises(ValueError, match=message):
        Simulator(config).run_stochastic(MissingAtRetirementStrategy(), num_trials=1)


def test_zero_weight_names_do_not_require_return_series():
    config = short_config(markets=[MarketConfig(DOMESTIC_STOCK, 0.5, 0, 1)])
    result = Simulator(config).run_stochastic(
        FixedAllocationStrategy({DOMESTIC_STOCK: 1, "Unused": 0}), num_trials=1
    )
    assert result == pytest.approx([7125])


def test_missing_held_asset_is_rejected_before_any_returns_are_applied():
    investor = Investor(25, 0, {"A": 100, "B": 100})
    with pytest.raises(ValueError, match="Investor at age 25:.*B"):
        investor.apply_returns({"A": 0.1})
    assert investor.holdings == {"A": 100, "B": 100}


def test_zero_holdings_and_unused_metadata_do_not_block_returns():
    investor = Investor(25, 0, {"A": 100, "B": 0})
    investor.apply_returns({"A": 0.1, "Inflation": 0.05, "Unused": 0.2})
    assert investor.holdings == pytest.approx({"A": 110, "B": 0})


@pytest.mark.parametrize("strategy_type", [PaperOptimalStrategy, PaperTDFStrategy])
def test_explicit_country_strategy_still_works_with_country_markets(strategy_type):
    config = short_config(markets=[
        MarketConfig("USA", 0.5, 0, 0.33),
        MarketConfig("GBR", 0.5, 0, 0.335),
        MarketConfig("JPN", 0.5, 0, 0.335),
        MarketConfig(BONDS, 0.5, 0, 0),
        MarketConfig(BILLS, 0.5, 0, 0),
    ])
    strategy = strategy_type(dom_label="USA", intl_assets={"GBR": 0.5, "JPN": 0.5})
    result = Simulator(config).run_stochastic(strategy, num_trials=1)
    assert result == pytest.approx([7125])


def test_validation_uses_the_supplied_engine_return_names():
    config = short_config(markets=[MarketConfig("Configured Asset", 0, 0, 1)])
    result = Simulator(config).run_stochastic(
        FixedAllocationStrategy({"USA": 1}),
        num_trials=1,
        market_engine=ConstantMarket({"USA": 0.5, "Inflation": 0.05}),
    )
    assert result == pytest.approx([7125])


@pytest.mark.parametrize("strategy_type", [PaperOptimalStrategy, PaperTDFStrategy])
def test_empty_international_allocations_are_not_replaced_with_defaults(strategy_type):
    with pytest.raises(ValueError, match="International asset weights must sum to 1.0"):
        strategy_type(intl_assets={})
