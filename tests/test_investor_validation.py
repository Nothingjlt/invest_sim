import pytest

from src.investor import Investor
from src.simulator import Simulator
from src.config import SimulationConfig
from src.strategy import BalancedStrategy, Strategy


@pytest.mark.parametrize("allocation", [
    {}, {"A": 2.0}, {"A": 0.4}, {"A": -0.1, "B": 1.1},
    {"A": float("nan")}, {"A": float("inf")}, {"": 1.0}, {"A": True},
])
@pytest.mark.parametrize("operation", ["rebalance", "withdraw", "earn_and_save"])
def test_invalid_allocations_fail_without_mutating_holdings(allocation, operation):
    investor = Investor(65, 100, {"A": 100.0, "B": 0.0})
    before = dict(investor.holdings)
    with pytest.raises(ValueError):
        if operation == "rebalance":
            investor.rebalance(allocation)
        elif operation == "withdraw":
            investor.withdraw(60, allocation)
        else:
            investor.earn_and_save(0.1, allocation)
    assert investor.holdings == before


@pytest.mark.parametrize("amount,expected,remaining", [(0, 0, 100), (60, 60, 40), (100, 100, 0), (150, 100, 0)])
def test_withdraw_uses_available_holdings_and_reports_shortfall(amount, expected, remaining):
    investor = Investor(65, 0, {"A": 100.0, "B": 0.0})
    assert investor.withdraw(amount, {"A": 0.5, "B": 0.5}) == expected
    assert investor.holdings == {"A": pytest.approx(remaining), "B": 0.0}


def test_withdraw_sells_actual_holdings_outside_target_allocation():
    investor = Investor(65, 0, {"Old": 100.0})
    assert investor.withdraw(60, {"New": 1.0}) == 60
    assert investor.holdings == {"Old": 40.0}
    investor.rebalance({"New": 1.0})
    assert investor.holdings == {"New": 40.0}


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), True])
def test_invalid_withdrawal_amount_is_atomic(amount):
    investor = Investor(65, 0, {"A": 100.0})
    with pytest.raises(ValueError):
        investor.withdraw(amount, {"A": 1.0})
    assert investor.holdings == {"A": 100.0}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["rebalance", "withdraw", "earn_and_save"])
def test_invalid_existing_holdings_are_rejected(value, operation):
    investor = Investor(65, 100, {"A": value})
    with pytest.raises(ValueError):
        if operation == "rebalance":
            investor.rebalance({"A": 1.0})
        elif operation == "withdraw":
            investor.withdraw(0, {"A": 1.0})
        else:
            investor.earn_and_save(0.1, {"A": 1.0})


def test_rounding_tolerance_does_not_multiply_wealth_on_rebalance():
    investor = Investor(25, 1000, {"A": 1000.0})
    allocation = {"A": 0.6, "B": 0.4000005}
    for _ in range(1000):
        investor.rebalance(allocation)
    assert investor.total_portfolio_value == pytest.approx(1000, abs=1e-8)
    investor.earn_and_save(1, allocation)
    assert investor.total_portfolio_value == pytest.approx(2000, abs=1e-8)
    assert allocation == {"A": 0.6, "B": 0.4000005}


def test_zero_wealth_withdrawal_returns_zero_without_dividing():
    investor = Investor(65, 0, {"A": 0.0})
    assert investor.withdraw(0, {"A": 1.0}) == 0
    assert investor.withdraw(100, {"A": 1.0}) == 0


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_savings_rate_cannot_create_or_remove_wealth(rate):
    investor = Investor(25, 100, {"A": 100.0})
    with pytest.raises(ValueError):
        investor.earn_and_save(rate, {"A": 1.0})
    assert investor.holdings == {"A": 100.0}


def test_colliding_balanced_labels_conserve_zero_return_lifecycle_wealth():
    config = SimulationConfig(starting_age=25, retirement_age=27, end_age=28,
                              initial_salary=1000, salary_growth_rate=0,
                              savings_rate=1, withdrawal_rate=0)
    class ZeroMarket:
        def get_annual_returns(self):
            return {"A": 0.0}
    result = Simulator(config).run_stochastic(BalancedStrategy("A", "A"),
                                             num_trials=1, market_engine=ZeroMarket())
    assert result == pytest.approx([2000])


def test_invalid_custom_allocation_rejected_before_simulated_cash_flows():
    class BadStrategy(Strategy):
        def _get_allocation(self, age):
            return {"A": 2.0}
    with pytest.raises(ValueError, match="sum to 1.0"):
        Simulator(SimulationConfig()).run_stochastic(BadStrategy(), num_trials=1)


@pytest.mark.parametrize("rate", [-1.1, float("nan"), float("inf"), True])
def test_invalid_salary_growth_is_atomic(rate):
    investor = Investor(25, 100)
    with pytest.raises(ValueError):
        investor.grow_salary(rate)
    assert investor.current_salary == 100


def test_contribution_overflow_does_not_partially_mutate_holdings():
    investor = Investor(25, 1e308, {"A": 1e308, "B": 0})
    with pytest.raises(ValueError, match="finite"):
        investor.earn_and_save(1, {"A": 0.9, "B": 0.1})
    assert investor.holdings == {"A": 1e308, "B": 0}
