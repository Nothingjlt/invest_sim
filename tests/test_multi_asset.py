import pytest
from src.investor import Investor


def test_portfolio_rebalancing():
    """
    Verify that rebalance() correctly redistributes wealth.
    Start: 50/50 split of $1000 ($500 each).
    Step 1: Apply 20% return to Stocks ($600), 0% to Bonds ($500). Total = $1100.
    Step 2: Rebalance back to 50/50.
    Result: Both assets should be $550.
    """
    investor = Investor(
        age=25, current_salary=50000, holdings={"Stocks": 500.0, "Bonds": 500.0}
    )

    # 20% return on Stocks, 0% on Bonds
    investor.apply_returns({"Stocks": 0.20, "Bonds": 0.0})

    assert investor.holdings["Stocks"] == pytest.approx(600.0)
    assert investor.total_portfolio_value == pytest.approx(1100.0)

    # Rebalance
    target_allocation = {"Stocks": 0.5, "Bonds": 0.5}
    investor.rebalance(target_allocation)

    assert investor.holdings["Stocks"] == pytest.approx(550.0)
    assert investor.holdings["Bonds"] == pytest.approx(550.0)
    assert investor.total_portfolio_value == pytest.approx(1100.0)


def test_multi_asset_simulator_run():
    """Verify the simulator can run a stochastic trial with multiple assets."""
    from src.config import SimulationConfig, MarketConfig
    from src.simulator import Simulator
    from src.strategy import FixedAllocationStrategy

    markets = [
        MarketConfig("Domestic", 0.07, 0.15, 0.5),
        MarketConfig("International", 0.07, 0.15, 0.5),
    ]
    config = SimulationConfig(markets=markets, end_age=30)  # Short run
    simulator = Simulator(config)

    strategy = FixedAllocationStrategy({"Domestic": 0.5, "International": 0.5})
    results = simulator.run_stochastic(strategy, num_trials=5)

    assert len(results) == 5
    for r in results:
        assert r > 0
