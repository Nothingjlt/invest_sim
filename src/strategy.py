from abc import ABC, abstractmethod
from typing import Dict

class Strategy(ABC):
    """Abstract base class for investment strategies."""
    @abstractmethod
    def get_allocation(self, age: int) -> Dict[str, float]:
        """Returns the target portfolio allocation for a given age."""
        pass

class FixedAllocationStrategy(Strategy):
    """A strategy where the allocation remains constant (e.g., 100% Equity)."""
    def __init__(self, target_allocation: Dict[str, float]):
        self.target_allocation = target_allocation

    def get_allocation(self, age: int) -> Dict[str, float]:
        return self.target_allocation

class GlidePathStrategy(Strategy):
    """
    A traditional glide path strategy (Target Date Fund).
    Shifts from stocks to bonds linearly between a starting age and a retirement age.
    Now supports granular asset classes within the equity and bond portions.
    """
    def __init__(
        self, 
        start_age: int, 
        retire_age: int, 
        start_equity: float = 0.90, 
        end_equity: float = 0.30,
        equity_assets: Dict[str, float] = None,
        bond_assets: Dict[str, float] = None
    ):
        self.start_age = start_age
        self.retire_age = retire_age
        self.start_equity = start_equity
        self.end_equity = end_equity
        
        # Default to "Stocks" and "Bonds" for backward compatibility
        self.equity_assets = equity_assets or {"Stocks": 1.0}
        self.bond_assets = bond_assets or {"Bonds": 1.0}
        
        # Validation
        if abs(sum(self.equity_assets.values()) - 1.0) > 1e-6:
            raise ValueError("Equity asset weights must sum to 1.0")
        if abs(sum(self.bond_assets.values()) - 1.0) > 1e-6:
            raise ValueError("Bond asset weights must sum to 1.0")

    def get_allocation(self, age: int) -> Dict[str, float]:
        """Calculates the granular asset split for a given age."""
        if age <= self.start_age:
            equity_pct = self.start_equity
        elif age >= self.retire_age:
            equity_pct = self.end_equity
        else:
            # Linear interpolation between start and retirement age
            progress = (age - self.start_age) / (self.retire_age - self.start_age)
            equity_pct = self.start_equity - progress * (self.start_equity - self.end_equity)
        
        bond_pct = 1.0 - equity_pct
        
        # Combine granular assets
        allocation = {}
        for asset, weight in self.equity_assets.items():
            allocation[asset] = equity_pct * weight
        for asset, weight in self.bond_assets.items():
            allocation[asset] = allocation.get(asset, 0.0) + (bond_pct * weight)
            
        return allocation
