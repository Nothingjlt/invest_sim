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
    """
    def __init__(
        self, 
        start_age: int, 
        retire_age: int, 
        start_equity: float = 0.90, 
        end_equity: float = 0.30
    ):
        self.start_age = start_age
        self.retire_age = retire_age
        self.start_equity = start_equity
        self.end_equity = end_equity

    def get_allocation(self, age: int) -> Dict[str, float]:
        """Calculates the equity/bond split for a given age."""
        if age <= self.start_age:
            equity = self.start_equity
        elif age >= self.retire_age:
            equity = self.end_equity
        else:
            # Linear interpolation between start and retirement age
            progress = (age - self.start_age) / (self.retire_age - self.start_age)
            equity = self.start_equity - progress * (self.start_equity - self.end_equity)
        
        return {
            "Stocks": equity,
            "Bonds": 1.0 - equity
        }
