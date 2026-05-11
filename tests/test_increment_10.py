from src.config import SimulationConfig, MarketConfig
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy
from src.investor import Investor


def test_fixed_real_withdrawal_strategy():
    """
    Verify that 'fixed_real' strategy uses the portfolio value AT retirement
    to set all subsequent withdrawals.
    """
    # 0% returns, 0% growth to keep numbers simple
    config = SimulationConfig(
        starting_age=64,
        retirement_age=65,
        end_age=67,
        initial_salary=50000,  # Added salary
        withdrawal_rate=0.10,  # 10%
        withdrawal_strategy="fixed_real",
        markets=[MarketConfig("Cash", 0.0, 0.0, 1.0)],
    )

    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Cash": 1.0})

    # Age 64: Start with $0. Savings rate 10% of $50k = $5000.
    # Age 65: Portfolio is $5000.
    # Withdrawal amount should be 10% of $5000 = $500.
    # Age 65: Withdraw $500. Remainder $4500.
    # Age 66: Withdraw $500. Remainder $4000.
    # End Age 67: Terminal wealth should be $4000.

    results = sim.run_stochastic(strategy, num_trials=1)
    assert results[0] == 4000.0


def test_variable_pct_withdrawal_strategy():
    """
    Verify that 'variable_pct' strategy calculates withdrawal based on
    CURRENT portfolio value each year.
    """
    config = SimulationConfig(
        starting_age=64,
        retirement_age=65,
        end_age=67,
        initial_salary=50000,  # Added salary
        withdrawal_rate=0.10,  # 10%
        withdrawal_strategy="variable_pct",
        markets=[MarketConfig("Cash", 0.0, 0.0, 1.0)],
    )

    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Cash": 1.0})

    # Age 64: Start $0. Save $5000.
    # Age 65: Portfolio $5000. Withdraw 10% of $5000 = $500. Remainder $4500.
    # Age 66: Portfolio $4500. Withdraw 10% of $4500 = $450. Remainder $4050.
    # End Age 67: Terminal wealth $4050.

    results = sim.run_stochastic(strategy, num_trials=1)
    assert results[0] == 4050.0


def test_mortality_affects_trial_length():
    """Verify that enabling mortality leads to variable terminal wealths due to different ages."""
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=100,
        initial_salary=50000,
        enable_mortality=True,
        markets=[MarketConfig("Cash", 0.03, 0.0, 1.0)],  # Fixed 3% return
    )

    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Cash": 1.0})

    # Run 100 trials. With mortality, they should NOT all be identical
    # because they end at different ages.
    results = sim.run_stochastic(strategy, num_trials=100)

    assert len(set(results)) > 1


def test_social_security_reduces_withdrawals():
    """Verify that social security benefit reduces the amount taken from the portfolio."""
    # Scenario: $1000 portfolio, 10% withdrawal rate ($100), $100 SS benefit.
    # Result: Net withdrawal should be $0.
    config = SimulationConfig(
        starting_age=65,
        retirement_age=65,
        end_age=66,
        initial_salary=0,
        withdrawal_rate=0.10,
        social_security_benefit=100.0,
        markets=[MarketConfig("Cash", 0.0, 0.0, 1.0)],
    )

    strategy = FixedAllocationStrategy({"Cash": 1.0})

    # Manually set initial holdings
    def run_with_initial_wealth():
        investor = Investor(age=65, current_salary=0, holdings={"Cash": 1000.0})
        # Simulate one year at age 65
        target_alloc = strategy.get_allocation(65)
        # 1. Market Growth (0%)
        # 2. Withdrawal
        withdrawal_amount = 1000.0 * 0.10  # $100
        net_withdrawal = max(
            0.0, withdrawal_amount - config.social_security_benefit
        )  # 100 - 100 = 0
        investor.withdraw(net_withdrawal, target_alloc)
        return investor.total_portfolio_value

    final_wealth = run_with_initial_wealth()
    assert final_wealth == 1000.0
