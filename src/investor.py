from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Investor:
    """Tracks the state of an individual investor during the simulation."""
    age: int
    current_salary: float
    # Maps asset name (e.g., 'Stocks') to current value
    holdings: Dict[str, float] = field(default_factory=lambda: {"Total": 0.0})

    @property
    def total_portfolio_value(self) -> float:
        return sum(self.holdings.values())

    def earn_and_save(self, savings_rate: float, allocation: Dict[str, float]):
        """Adds a portion of salary to holdings based on current allocation."""
        contribution = self.current_salary * savings_rate
        for asset, weight in allocation.items():
            self.holdings[asset] = self.holdings.get(asset, 0.0) + (contribution * weight)
        return contribution

    def grow_salary(self, growth_rate: float):
        self.current_salary *= (1 + growth_rate)

    def withdraw(self, amount: float, allocation: Dict[str, float]):
        """Removes a specific amount from holdings proportionally to allocation."""
        total = self.total_portfolio_value
        if amount > total:
            actual_withdrawn = total
            self.holdings = {asset: 0.0 for asset in self.holdings}
            return actual_withdrawn
        
        # Withdraw proportionally from each asset
        for asset, weight in allocation.items():
            asset_withdrawal = amount * weight
            self.holdings[asset] = self.holdings.get(asset, 0.0) - asset_withdrawal
        return amount

    def apply_returns(self, returns: Dict[str, float]):
        """Applies annual returns to each asset class."""
        for asset, ret in returns.items():
            if asset in self.holdings:
                self.holdings[asset] *= (1 + ret)

    def rebalance(self, target_allocation: Dict[str, float]):
        """Redistributes total wealth according to target weights."""
        total = self.total_portfolio_value
        self.holdings = {asset: total * weight for asset, weight in target_allocation.items()}
