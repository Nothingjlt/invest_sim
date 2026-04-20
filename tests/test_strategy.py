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
