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

class BalancedStrategy(Strategy):
    """A traditional balanced strategy (60% Domestic Stocks, 40% Bonds)."""
    def __init__(self, domestic_label: str = "Domestic Stock", bond_label: str = "Bonds"):
        self.domestic_label = domestic_label
        self.bond_label = bond_label

    def get_allocation(self, age: int) -> Dict[str, float]:
        return {
            self.domestic_label: 0.60,
            self.bond_label: 0.40
        }

class PaperOptimalStrategy(Strategy):
    """
    The optimal age-based strategy recommended by the paper.
    - 100% Equity (33% Domestic, 67% Intl) for most of the lifecycle.
    - Tactical cash (Bills) allocation near retirement.
    """
    def __init__(
        self, 
        retire_age: int = 65,
        dom_label: str = "Domestic Stock",
        intl_label: str = "International Stock",
        bills_label: str = "Bills"
    ):
        self.retire_age = retire_age
        self.dom_label = dom_label
        self.intl_label = intl_label
        self.bills_label = bills_label

    def get_allocation(self, age: int) -> Dict[str, float]:
        # Before retirement or after age 70: standard 100% equity split
        if age < self.retire_age or age >= 70:
            return {self.dom_label: 0.33, self.intl_label: 0.67}
        
        # Transition at retirement (Age 65-70)
        # Age 65: 26% Dom, 47% Intl, 27% Bills
        # Age 70: 33% Dom, 67% Intl, 0% Bills
        progress = (age - self.retire_age) / (70 - self.retire_age)
        
        bills = 0.27 * (1 - progress)
        dom = 0.26 + (0.33 - 0.26) * progress
        intl = 1.0 - bills - dom
        
        return {
            self.dom_label: dom,
            self.intl_label: intl,
            self.bills_label: bills
        }

class PaperTDFStrategy(Strategy):
    """
    A representative Target Date Fund (TDF) glide path based on the paper.
    - Age 25: 54% Domestic, 36% Intl, 10% Bonds, 0% Bills.
    - Age 65: 10% Domestic, 7% Intl, 73% Bonds, 10% Bills.
    """
    def __init__(
        self,
        start_age: int = 25,
        retire_age: int = 65,
        dom_label: str = "Domestic Stock",
        intl_label: str = "International Stock",
        bond_label: str = "Bonds",
        bills_label: str = "Bills"
    ):
        self.start_age = start_age
        self.retire_age = retire_age
        self.dom_label = dom_label
        self.intl_label = intl_label
        self.bond_label = bond_label
        self.bills_label = bills_label

    def get_allocation(self, age: int) -> Dict[str, float]:
        if age <= self.start_age:
            p = 0.0
        elif age >= self.retire_age:
            p = 1.0
        else:
            p = (age - self.start_age) / (self.retire_age - self.start_age)

        # Interpolation between start and retirement
        dom = 0.54 + (0.10 - 0.54) * p
        intl = 0.36 + (0.07 - 0.36) * p
        bonds = 0.10 + (0.73 - 0.10) * p
        bills = 0.00 + (0.10 - 0.00) * p

        return {
            self.dom_label: dom,
            self.intl_label: intl,
            self.bond_label: bonds,
            self.bills_label: bills
        }
