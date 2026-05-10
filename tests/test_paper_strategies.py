import pytest
from src.strategy import PaperOptimalStrategy, PaperTDFStrategy, BalancedStrategy

def test_balanced_strategy():
    strategy = BalancedStrategy()
    alloc = strategy.get_allocation(40)
    assert alloc["Domestic Stock"] == pytest.approx(0.60)
    assert alloc["Bonds"] == pytest.approx(0.40)

def test_paper_optimal_strategy_lifecycle():
    # Pass custom labels to match the test's expectations
    intl_assets = {"International Stock": 1.0}
    strategy = PaperOptimalStrategy(retire_age=65, dom_label="Domestic Stock", intl_assets=intl_assets)
    
    # Pre-retirement: 100% equity (33/67)
    alloc_40 = strategy.get_allocation(40)
    assert alloc_40["Domestic Stock"] == pytest.approx(0.33)
    assert alloc_40["International Stock"] == pytest.approx(0.67)
    assert alloc_40.get("Bills", 0) == 0
    
    # At retirement (65): Tactical cash cushion
    alloc_65 = strategy.get_allocation(65)
    assert alloc_65["Domestic Stock"] == pytest.approx(0.26)
    assert alloc_65["International Stock"] == pytest.approx(0.47)
    assert alloc_65["Bills"] == pytest.approx(0.27)
    
    # Mid-transition (67.5): Midway between 65 and 70
    # Bills: 27 -> 0. Midpoint is 13.5
    alloc_67_5 = strategy.get_allocation(67.5) # Using float age for testing interpolation
    assert alloc_67_5["Bills"] == pytest.approx(0.135)
    
    # Post-transition (70+): Returns to 100% equity
    alloc_75 = strategy.get_allocation(75)
    assert alloc_75["Domestic Stock"] == pytest.approx(0.33)
    assert alloc_75["International Stock"] == pytest.approx(0.67)
    assert alloc_75.get("Bills", 0) == 0

def test_paper_tdf_strategy_lifecycle():
    intl_assets = {"International Stock": 1.0}
    strategy = PaperTDFStrategy(start_age=25, retire_age=65, dom_label="Domestic Stock", intl_assets=intl_assets)
    
    # Start (25): 54/36/10/0
    alloc_25 = strategy.get_allocation(25)
    assert alloc_25["Domestic Stock"] == pytest.approx(0.54)
    assert alloc_25["International Stock"] == pytest.approx(0.36)
    assert alloc_25["Bonds"] == pytest.approx(0.10)
    assert alloc_25.get("Bills", 0) == 0
    
    # End (65): 10/7/73/10
    alloc_65 = strategy.get_allocation(65)
    assert alloc_65["Domestic Stock"] == pytest.approx(0.10)
    assert alloc_65["International Stock"] == pytest.approx(0.07)
    assert alloc_65["Bonds"] == pytest.approx(0.73)
    assert alloc_65["Bills"] == pytest.approx(0.10)
    
    # Midpoint (45): Midway between 25 and 65
    alloc_45 = strategy.get_allocation(45)
    # Dom: 0.54 -> 0.10. Mid is 0.32
    # Intl: 0.36 -> 0.07. Mid is 0.215
    # Bonds: 0.10 -> 0.73. Mid is 0.415
    # Bills: 0.00 -> 0.10. Mid is 0.05
    assert alloc_45["Domestic Stock"] == pytest.approx(0.32)
    assert alloc_45["International Stock"] == pytest.approx(0.215)
    assert alloc_45["Bonds"] == pytest.approx(0.415)
    assert alloc_45["Bills"] == pytest.approx(0.05)
