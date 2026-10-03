import pytest

from src.config import MarketConfig, SimulationConfig
from src.investor import Investor
from src.market import Market
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy


class AllocationSwitchStrategy(FixedAllocationStrategy):
    def __init__(self, switch_age):
        super().__init__({"A": 1.0})
        self.switch_age = switch_age

    def _get_allocation(self, age):
        return {"B": 1.0} if age >= self.switch_age else {"A": 1.0}


class TwoAssetMarket(Market):
    def get_annual_returns(self):
        return {"A": 0.0, "B": 1.0}


def timing_config(**overrides):
    values = dict(
        starting_age=25,
        retirement_age=26,
        end_age=27,
        initial_salary=1000.0,
        salary_growth_rate=0.0,
        savings_rate=1.0,
        withdrawal_rate=0.0,
        markets=[
            MarketConfig("A", 0.0, 0.0, 1.0),
            MarketConfig("B", 1.0, 0.0, 0.0),
        ],
    )
    values.update(overrides)
    return SimulationConfig(**values)


@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_retirement_allocation_receives_the_first_period_return(granularity):
    config = timing_config(decumulation_granularity=granularity)
    result = Simulator(config).run_stochastic(
        AllocationSwitchStrategy(config.retirement_age),
        num_trials=1,
        market_engine=TwoAssetMarket(),
        track_paths=True,
    )

    assert result.paths[0] == pytest.approx([0.0, 1000.0, 2000.0])
    assert result.terminal_wealths == pytest.approx([2000.0])


def test_allocation_change_applies_during_accumulation():
    config = timing_config(retirement_age=27, end_age=28)
    result = Simulator(config).run_stochastic(
        AllocationSwitchStrategy(26),
        num_trials=1,
        market_engine=TwoAssetMarket(),
        track_paths=True,
    )

    assert result.paths[0] == pytest.approx([0.0, 1000.0, 3000.0, 6000.0])


def test_monthly_retirement_switch_funds_withdrawals_before_returns(monkeypatch):
    config = timing_config(
        decumulation_granularity="monthly",
        withdrawal_strategy="fixed_real",
        withdrawal_rate=0.12,
    )
    balances_before_returns = []
    apply_returns = Investor.apply_returns

    def record_returns(investor, returns):
        balances_before_returns.append(dict(investor.holdings))
        apply_returns(investor, returns)

    monkeypatch.setattr(Investor, "apply_returns", record_returns)
    result = Simulator(config).run_stochastic(
        AllocationSwitchStrategy(config.retirement_age),
        num_trials=1,
        market_engine=TwoAssetMarket(),
        track_paths=True,
    )

    monthly_growth = 2.0 ** (1.0 / 12.0)
    expected_wealth = (
        2000.0 - 10.0 * monthly_growth * (2.0 - 1.0) / (monthly_growth - 1.0)
    )
    assert result.terminal_wealths == pytest.approx([expected_wealth])
    assert result.withdrawal_paths[0] == pytest.approx([0.0, 0.0, 120.0])
    assert all(
        balance >= 0.0
        for snapshot in balances_before_returns
        for balance in snapshot.values()
    )
