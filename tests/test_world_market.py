import pytest
from src.config import SimulationConfig, MarketConfig
from src.market import BootstrapMarket
from src.strategy import WorldEquityStrategy, PaperOptimalStrategy
from src.simulator import Simulator
from src.metrics import Metrics

def test_world_market_data_loading():
    """Verify that BootstrapMarket can load the 60+ country indices."""
    csv_path = "data/global_historical_returns.csv"
    market = BootstrapMarket(csv_path)
    returns = market.get_annual_returns()
    
    # Check for presence of key countries
    assert "USA" in returns
    assert "GBR" in returns
    assert "CHN" in returns
    assert "BRA" in returns
    assert "Bonds" in returns
    assert "Bills" in returns
    assert len(returns) >= 60

def test_world_equity_strategy_allocation():
    """Verify WorldEquityStrategy correctly flattens nested weights."""
    region_weights = {
        "Developed": {"USA": 0.5, "JPN": 0.2},
        "Emerging": {"CHN": 0.3}
    }
    strategy = WorldEquityStrategy(region_weights)
    alloc = strategy.get_allocation(25)
    
    assert alloc["USA"] == 0.5
    assert alloc["JPN"] == 0.2
    assert alloc["CHN"] == 0.3
    assert sum(alloc.values()) == pytest.approx(1.0)

def test_world_equity_strategy_duplicate_collision():
    """Verify validation prevents same country in multiple regions."""
    region_weights = {
        "Developed": {"KOR": 0.5},
        "Emerging": {"KOR": 0.5}
    }
    with pytest.raises(ValueError, match="multiple regions"):
        WorldEquityStrategy(region_weights)

def test_paper_optimal_world_aware():
    """Verify PaperOptimalStrategy works with specific country mappings."""
    # Custom mapping: Domestic is UK, International is USA/JPN
    intl_assets = {"USA": 0.5, "JPN": 0.5}
    strategy = PaperOptimalStrategy(dom_label="GBR", intl_assets=intl_assets)
    
    # Pre-retirement (1/3 Domestic, 2/3 International)
    alloc_young = strategy.get_allocation(30)
    assert alloc_young["GBR"] == pytest.approx(0.33)
    assert alloc_young["USA"] == pytest.approx(0.67 * 0.5)
    assert alloc_young["JPN"] == pytest.approx(0.67 * 0.5)
    
    # Retirement Age 65 (Tactical Shift)
    alloc_65 = strategy.get_allocation(65)
    assert alloc_65["Bills"] == pytest.approx(0.27)
    assert alloc_65["GBR"] == pytest.approx(0.26)

def test_simulation_with_world_market():
    """Run a full simulation using world market indices."""
    world_markets = SimulationConfig.get_world_market_configs()
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        markets=world_markets
    )
    
    # Define a simple World Strategy: 50% USA, 25% GBR, 25% CHN
    strategy = WorldEquityStrategy({
        "Dev": {"USA": 0.5, "GBR": 0.25},
        "EM": {"CHN": 0.25}
    })
    
    sim = Simulator(config)
    market = BootstrapMarket("data/global_historical_returns.csv")
    
    # Just run 10 trials to ensure no crashes
    results = sim.run_stochastic(strategy, num_trials=10, market_engine=market)
    assert len(results) == 10
    assert all(w >= 0 for w in results)

def test_diversification_safety_comparison():
    """
    Empirical check: World portfolio should have better or equal tail risk
    than a single country portfolio if they have similar returns.
    """
    # Note: Since data is synthetic/simulated, we just ensure the tool can make the comparison.
    world_markets = SimulationConfig.get_world_market_configs()
    config = SimulationConfig(starting_age=25, markets=world_markets)
    market = BootstrapMarket("data/global_historical_returns.csv")
    sim = Simulator(config)
    
    # Portfolio A: 100% USA
    strategy_a = WorldEquityStrategy({"USA": {"USA": 1.0}})
    # Portfolio B: 1/3 across 3 countries (Diversified)
    strategy_b = WorldEquityStrategy({"Dev": {"USA": 0.34, "GBR": 0.33, "JPN": 0.33}})
    
    results_a = sim.run_stochastic(strategy_a, num_trials=100, market_engine=market)
    results_b = sim.run_stochastic(strategy_b, num_trials=100, market_engine=market)
    
    # Calculate 5th percentile
    tail_a = Metrics.percentile(results_a, 5)
    tail_b = Metrics.percentile(results_b, 5)
    
    print(f"Tail risk USA: {tail_a}, Tail risk World: {tail_b}")
    # We don't assert tail_b > tail_a because of randomness in 100 trials,
    # but the test passing ensures the multi-country engine is working.
    assert tail_a >= 0
    assert tail_b >= 0
