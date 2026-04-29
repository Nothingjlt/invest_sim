import pytest
from src.strategy import FixedAllocationStrategy, GlidePathStrategy

def test_fixed_allocation_strategy():
    """Verify that a fixed strategy remains constant across all ages."""
    target = {"Stocks": 1.0}
    strategy = FixedAllocationStrategy(target)
    
    assert strategy.get_allocation(25) == target
    assert strategy.get_allocation(65) == target
    assert strategy.get_allocation(100) == target

def test_glide_path_strategy_linear_shift():
    """
    Verify the linear shift from 90% stocks at age 25 to 30% stocks at age 65.
    (60% drop over 40 years = 1.5% drop per year)
    """
    strategy = GlidePathStrategy(
        start_age=25, 
        retire_age=65, 
        start_equity=0.90, 
        end_equity=0.30
    )
    
    # Boundary tests
    assert strategy.get_allocation(25)["Stocks"] == pytest.approx(0.90)
    assert strategy.get_allocation(65)["Stocks"] == pytest.approx(0.30)
    assert strategy.get_allocation(20)["Stocks"] == pytest.approx(0.90) # Before start age
    assert strategy.get_allocation(70)["Stocks"] == pytest.approx(0.30) # After retirement age

    # Midpoint test (Age 45 should be exactly midway between 0.90 and 0.30 = 0.60)
    assert strategy.get_allocation(45)["Stocks"] == pytest.approx(0.60)
    
    # Check bonds
    assert strategy.get_allocation(45)["Bonds"] == pytest.approx(0.40)

def test_glide_path_granular_assets():
    """Verify that GlidePathStrategy handles multiple equity and bond assets."""
    equity_assets = {"Domestic": 0.5, "International": 0.5}
    bond_assets = {"Gov Bonds": 0.7, "Corp Bonds": 0.3}
    
    strategy = GlidePathStrategy(
        start_age=25,
        retire_age=65,
        start_equity=1.0, # 100% equity at start
        end_equity=0.0,   # 0% equity at retirement
        equity_assets=equity_assets,
        bond_assets=bond_assets
    )
    
    # Age 25: 100% Equity (50/50 Domestic/Intl)
    alloc_25 = strategy.get_allocation(25)
    assert alloc_25["Domestic"] == pytest.approx(0.5)
    assert alloc_25["International"] == pytest.approx(0.5)
    assert alloc_25.get("Gov Bonds", 0) == 0
    
    # Age 65: 100% Bonds (70/30 Gov/Corp)
    alloc_65 = strategy.get_allocation(65)
    assert alloc_65["Gov Bonds"] == pytest.approx(0.7)
    assert alloc_65["Corp Bonds"] == pytest.approx(0.3)
    assert alloc_65.get("Domestic", 0) == 0
    
    # Age 45: 50% Equity, 50% Bonds
    # Equity part: 0.5 * 0.5 = 0.25 Domestic, 0.25 International
    # Bond part: 0.5 * 0.7 = 0.35 Gov, 0.5 * 0.3 = 0.15 Corp
    alloc_45 = strategy.get_allocation(45)
    assert alloc_45["Domestic"] == pytest.approx(0.25)
    assert alloc_45["International"] == pytest.approx(0.25)
    assert alloc_45["Gov Bonds"] == pytest.approx(0.35)
    assert alloc_45["Corp Bonds"] == pytest.approx(0.15)

def test_glide_path_invalid_weights():
    """Confirm ValueError is raised if asset weights do not sum to 1.0."""
    with pytest.raises(ValueError, match="Equity asset weights must sum to 1.0"):
        GlidePathStrategy(25, 65, equity_assets={"A": 0.5})
        
    with pytest.raises(ValueError, match="Bond asset weights must sum to 1.0"):
        GlidePathStrategy(25, 65, bond_assets={"B": 1.1})
