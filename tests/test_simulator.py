import pytest
from src.config import SimulationConfig
from src.simulator import Simulator


def test_zero_growth_scenario():
    """
    Mathematical verification of the lifecycle loop.
    With 0% return and 0% salary growth, the terminal wealth should follow:
    Savings = (Years Working) * (Salary * Savings Rate)
    Terminal Wealth = Savings * (1 - Withdrawal Rate) ^ (Retirement Years)
    """
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=100,
        initial_salary=50000.0,
        salary_growth_rate=0.0,  # Fixed salary for easy math
        savings_rate=0.10,  # $5000/year
        withdrawal_rate=0.04,  # 4% of portfolio/year
    )

    working_years = config.retirement_age - config.starting_age  # 40 years
    total_savings = working_years * (
        config.initial_salary * config.savings_rate
    )  # $200,000

    retirement_years = config.end_age - config.retirement_age  # 35 years
    expected_terminal_wealth = total_savings * (
        (1 - config.withdrawal_rate) ** retirement_years
    )

    simulator = Simulator(config)
    actual_terminal_wealth = simulator.run_deterministic(annual_return=0.0)

    # Using pytest.approx for floating point comparisons
    assert actual_terminal_wealth == pytest.approx(expected_terminal_wealth)


def test_accumulation_only():
    """Verify that portfolio equals total savings when retirement and end age are the same."""
    config = SimulationConfig(
        starting_age=20,
        retirement_age=30,
        end_age=30,
        initial_salary=10000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
    )
    # 10 years * ($10000 * 0.1) = $10000
    simulator = Simulator(config)
    assert simulator.run_deterministic(annual_return=0.0) == pytest.approx(10000.0)
