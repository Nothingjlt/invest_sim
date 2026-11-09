"""Strategies remain authoritative for descriptor and legacy configurations."""

import pytest

from src.config import SimulationConfig
from src.simulator import Simulator
from src.strategy import (
    BalancedStrategy, FixedAllocationStrategy, GlidePathStrategy,
    PaperOptimalStrategy, PaperTDFStrategy, Strategy, WorldEquityStrategy,
)


class CustomSleeveStrategy(Strategy):
    def _get_allocation(self, age):
        equity = 0.6 if age <= 25 else 0.8
        return {"My Equity": equity, "Bills": 1 - equity}

    def resolve_allocation_for_market(self, allocation, available_assets):
        return super().resolve_allocation_for_market(
            {"USA": allocation["My Equity"], "Bills": allocation["Bills"]},
            available_assets,
        )


RETURNS = {
    "USA": 0.20, "GBR": 0.10, "JPN": -0.20, "FRA": 0.30, "DEU": 0.50,
    "Domestic Stock": 0.20, "International Stock": 0.13,
    "Bonds": 0.04, "Bills": -0.02,
}

CASES = [
    pytest.param("world", lambda: PaperOptimalStrategy(retire_age=26), {
        "USA": 0.34, "GBR": 0.198, "JPN": 0.198, "FRA": 0.132, "DEU": 0.132,
    }, id="paper-optimal-country-sleeves"),
    pytest.param("world", lambda: PaperTDFStrategy(retire_age=26), {
        "USA": 0.10, "GBR": 0.021, "JPN": 0.021, "FRA": 0.014, "DEU": 0.014,
        "Bonds": 0.73, "Bills": 0.10,
    }, id="paper-tdf-country-sleeves"),
    pytest.param("world", lambda: WorldEquityStrategy({
        "Developed": {"USA": 0.2, "GBR": 0.5, "JPN": 0.3},
    }), {"USA": 0.2, "GBR": 0.5, "JPN": 0.3}, id="world-equity"),
    pytest.param("world", lambda: BalancedStrategy(domestic_label="USA"), {
        "USA": 0.6, "Bonds": 0.4,
    }, id="balanced-world"),
    pytest.param("world", lambda: FixedAllocationStrategy({"GBR": 0.25, "JPN": 0.75}), {
        "GBR": 0.25, "JPN": 0.75,
    }, id="fixed-allocation"),
    pytest.param("world", lambda: GlidePathStrategy(
        25, 26, end_equity=0.3, equity_assets={"USA": 0.25, "GBR": 0.75},
    ), {"USA": 0.075, "GBR": 0.225, "Bonds": 0.7}, id="glide-path"),
    pytest.param("world", CustomSleeveStrategy, {"USA": 0.8, "Bills": 0.2}, id="custom-sleeve"),
    pytest.param("paper", lambda: PaperOptimalStrategy(retire_age=26), {
        "Domestic Stock": 0.34, "International Stock": 0.66,
    }, id="paper-optimal-aggregate-sleeves"),
    pytest.param("paper", lambda: PaperTDFStrategy(retire_age=26), {
        "Domestic Stock": 0.10, "International Stock": 0.07,
        "Bonds": 0.73, "Bills": 0.10,
    }, id="paper-tdf-aggregate-sleeves"),
]


@pytest.mark.parametrize("universe,strategy_factory,expected_weights", CASES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_strategy_owns_allocation_with_descriptors_or_conflicting_legacy_weights(
    universe, strategy_factory, expected_weights, granularity,
):
    factory = (
        SimulationConfig.get_world_market_configs if universe == "world"
        else SimulationConfig.get_paper_market_configs
    )
    strategy = strategy_factory()
    original_allocations = [strategy.get_allocation(age) for age in (25, 26)]
    resolved = strategy.resolve_allocation_for_market(
        original_allocations[1], [market.name for market in factory()],
    )
    assert resolved == pytest.approx(expected_weights)

    if granularity == "annual":
        growth = 1 + sum(weight * RETURNS[asset] for asset, weight in expected_weights.items())
    else:
        # Monthly rebalancing compounds the weighted constituent gross returns.
        growth = sum(
            weight * (1 + RETURNS[asset]) ** (1 / 12)
            for asset, weight in expected_weights.items()
        ) ** 12
    expected_path = [0.0, 1000.0, 1000.0 * growth]

    for legacy_weights in (False, True):
        markets = factory()
        assert all(market.weight is None for market in markets)
        for market in markets:
            market.expected_return = RETURNS.get(market.name, 0.0)
            market.volatility = 0.0
            if legacy_weights:
                # A valid legacy allocation deliberately disagrees with every
                # strategy above. It must never become the simulation policy.
                market.weight = 1.0 if market.name == "Bills" else 0.0
        original_weights = [market.weight for market in markets]
        config = SimulationConfig(
            starting_age=25, retirement_age=26, end_age=27,
            initial_salary=1000.0, salary_growth_rate=0.0,
            savings_rate=1.0, withdrawal_rate=0.0,
            decumulation_granularity=granularity, markets=markets,
        )
        config.validate()
        result = Simulator(config).run_stochastic(strategy, num_trials=1, track_paths=True)
        assert result.paths[0] == pytest.approx(expected_path)
        assert result.terminal_wealths == pytest.approx([expected_path[-1]])
        assert [market.weight for market in markets] == original_weights
        assert [strategy.get_allocation(age) for age in (25, 26)] == original_allocations
