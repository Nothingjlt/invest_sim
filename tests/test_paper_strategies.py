import pytest
from src.config import MarketConfig, SimulationConfig
from src.simulator import Simulator
from src.strategy import PaperOptimalStrategy, PaperTDFStrategy, BalancedStrategy


def test_balanced_strategy():
    strategy = BalancedStrategy()
    alloc = strategy.get_allocation(40)
    assert alloc["Domestic Stock"] == pytest.approx(0.60)
    assert alloc["Bonds"] == pytest.approx(0.40)


def test_paper_market_config_weights_match_fixed_weight_policy():
    weights = {
        market.name: market.weight
        for market in SimulationConfig.get_paper_market_configs()
    }
    assert weights == pytest.approx({
        "Domestic Stock": 0.34,
        "International Stock": 0.66,
        "Bonds": 0.0,
        "Bills": 0.0,
    })


def test_paper_optimal_strategy_uses_paper_fixed_weight_all_equity_policy():
    intl_assets = {"International Stock": 1.0}
    strategy = PaperOptimalStrategy(
        retire_age=65, dom_label="Domestic Stock", intl_assets=intl_assets
    )

    # Table III's optimal fixed-weight policy is 34% domestic / 66% international.
    for age in (25, 40, 65, 67.5, 90):
        allocation = strategy.get_allocation(age)
        assert allocation == pytest.approx({
            "Domestic Stock": 0.34,
            "International Stock": 0.66,
        })
        assert sum(allocation.values()) == pytest.approx(1.0)
        assert "Bonds" not in allocation
        assert "Bills" not in allocation


def test_paper_tdf_linear_approximation_endpoints_and_midpoint():
    intl_assets = {"International Stock": 1.0}
    strategy = PaperTDFStrategy(
        start_age=25, retire_age=65, dom_label="Domestic Stock", intl_assets=intl_assets
    )

    # Approximation start: 54/36/10/0.
    alloc_25 = strategy.get_allocation(25)
    assert alloc_25["Domestic Stock"] == pytest.approx(0.54)
    assert alloc_25["International Stock"] == pytest.approx(0.36)
    assert alloc_25["Bonds"] == pytest.approx(0.10)
    assert alloc_25.get("Bills", 0) == 0

    # Approximation endpoint: 10/7/73/10.
    alloc_65 = strategy.get_allocation(65)
    assert alloc_65["Domestic Stock"] == pytest.approx(0.10)
    assert alloc_65["International Stock"] == pytest.approx(0.07)
    assert alloc_65["Bonds"] == pytest.approx(0.73)
    assert alloc_65["Bills"] == pytest.approx(0.10)

    # Midpoint (45): midpoint of this linear approximation, not a Figure 1 datum.
    alloc_45 = strategy.get_allocation(45)
    # Dom: 0.54 -> 0.10. Mid is 0.32
    # Intl: 0.36 -> 0.07. Mid is 0.215
    # Bonds: 0.10 -> 0.73. Mid is 0.415
    # Bills: 0.00 -> 0.10. Mid is 0.05
    assert alloc_45["Domestic Stock"] == pytest.approx(0.32)
    assert alloc_45["International Stock"] == pytest.approx(0.215)
    assert alloc_45["Bonds"] == pytest.approx(0.415)
    assert alloc_45["Bills"] == pytest.approx(0.05)


def test_paper_optimal_strategy_stochastic_runs_deterministically():
    markets = [
        MarketConfig(
            name="Domestic Stock",
            expected_return=0.0,
            volatility=0.0,
            weight=0.34,
        ),
        MarketConfig(
            name="International Stock",
            expected_return=0.0,
            volatility=0.0,
            weight=0.66,
        ),
        MarketConfig(
            name="Bills",
            expected_return=0.0,
            volatility=0.0,
            weight=0.0,
        ),
    ]
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=66,
        initial_salary=50000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
        withdrawal_rate=0.04,
        withdrawal_strategy="fixed_real",
        markets=markets,
    )
    sim = Simulator(config)
    strategy = PaperOptimalStrategy(
        retire_age=65,
        dom_label="Domestic Stock",
        intl_assets={"International Stock": 1.0},
        bills_label="Bills",
    )

    results = sim.run_stochastic(strategy, num_trials=2)
    assert len(results) == 2
    assert results[0] == pytest.approx(results[1])
    assert all(value >= 0.0 for value in results)


def test_paper_tdf_strategy_stochastic_runs_deterministically():
    markets = [
        MarketConfig(
            name="Domestic Stock",
            expected_return=0.0,
            volatility=0.0,
            weight=0.10,
        ),
        MarketConfig(
            name="International Stock",
            expected_return=0.0,
            volatility=0.0,
            weight=0.07,
        ),
        MarketConfig(
            name="Bonds",
            expected_return=0.0,
            volatility=0.0,
            weight=0.73,
        ),
        MarketConfig(
            name="Bills",
            expected_return=0.0,
            volatility=0.0,
            weight=0.10,
        ),
    ]
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=66,
        initial_salary=50000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
        withdrawal_rate=0.04,
        withdrawal_strategy="fixed_real",
        markets=markets,
    )
    sim = Simulator(config)
    strategy = PaperTDFStrategy(
        start_age=25,
        retire_age=65,
        dom_label="Domestic Stock",
        intl_assets={"International Stock": 1.0},
        bond_label="Bonds",
        bills_label="Bills",
    )

    results = sim.run_stochastic(strategy, num_trials=2)
    assert len(results) == 2
    assert results[0] == pytest.approx(results[1])
    assert all(value >= 0.0 for value in results)
