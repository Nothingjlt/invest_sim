import pytest
from src.metrics import Metrics, SavingsGapCalculator
from src.config import SimulationConfig, MarketConfig
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy


def test_basic_metrics():
    """Verify probability of ruin and percentile calculations."""
    wealths = [0.0, 0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

    assert Metrics.probability_of_ruin(wealths) == 0.2
    assert Metrics.percentile(wealths, 50) == pytest.approx(
        350.0
    )  # Median of 300 and 400
    assert Metrics.percentile(wealths, 0) == 0.0
    assert Metrics.percentile(wealths, 100) == 800.0


def test_savings_gap_logic():
    """
    Verify the binary search for savings rate.
    Target: $100,000 terminal wealth.
    Fixed Salary: $50,000.
    Years: 40.
    Return: 0.07.

    Using FV of an Annuity (simplified for end-of-period contributions):
    PMT = 50000 * rate
    r = 0.07
    n = 40
    FV = PMT * ((1 + r)^n - 1) / r
    """
    market = MarketConfig(
        "Stock", 0.07, 0.0, 1.0
    )  # 0 volatility for deterministic test

    def simulator_factory(savings_rate):
        config = SimulationConfig(
            markets=[market],
            savings_rate=savings_rate,
            salary_growth_rate=0.0,
            starting_age=25,
            retirement_age=65,
            end_age=65,  # Accumulation only
        )
        return Simulator(config)

    r = 0.07
    n = 40
    # To get $1,000,000 for easier numbers:
    target_wealth = 1000000.0

    # PMT = target * r / ((1+r)^n - 1)
    # PMT = 1000000 * 0.07 / ((1.07)^40 - 1)
    pmt_required = target_wealth * r / ((1 + r) ** n - 1)
    expected_rate = pmt_required / 50000.0  # ~0.0093 or 0.93%

    calc = SavingsGapCalculator(simulator_factory)
    strategy = FixedAllocationStrategy({"Stock": 1.0})

    required_rate = calc.find_required_savings_rate(
        target_5th_percentile=target_wealth,
        strategy=strategy,
        num_trials=1,
        tolerance=0.01,
    )

    assert required_rate == pytest.approx(expected_rate, rel=1e-2)


def test_ruin_sensitivity():
    """Verify that very high withdrawal rates trigger high ruin probability."""
    # (Simplified test setup)
    wealths = [-10.0, 0.0, 5.0, 100.0]
    assert Metrics.probability_of_ruin(wealths) == 0.5
