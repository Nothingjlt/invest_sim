from dataclasses import dataclass, field
from typing import Dict

from src.assets import finite_number, require_return_series, validate_allocation


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
        allocation = validate_allocation(allocation)
        self._validate_holdings()
        salary = finite_number(self.current_salary, context="current_salary")
        rate = finite_number(savings_rate, context="savings_rate")
        if salary < 0 or not 0 <= rate <= 1:
            raise ValueError("Salary must be nonnegative and savings_rate between 0 and 1.")
        contribution = salary * rate
        holdings = dict(self.holdings)
        for asset, weight in allocation.items():
            holdings[asset] = holdings.get(asset, 0.0) + (
                contribution * weight
            )
            finite_number(holdings[asset], context=f"Holding for {asset}")
        finite_number(sum(holdings.values()), context="Portfolio total")
        self.holdings = holdings
        return contribution

    def grow_salary(self, growth_rate: float):
        salary = finite_number(self.current_salary, context="current_salary")
        rate = finite_number(growth_rate, context="growth_rate")
        if salary < 0 or rate < -1:
            raise ValueError("Salary must be nonnegative and growth_rate at least -1.")
        self.current_salary = finite_number(salary * (1 + rate), context="New salary")

    def _validate_holdings(self):
        for asset, value in self.holdings.items():
            if finite_number(value, context=f"Holding for {asset}") < 0:
                raise ValueError("Holdings must be nonnegative.")
        finite_number(self.total_portfolio_value, context="Portfolio total")

    def withdraw(self, amount: float, allocation: Dict[str, float]):
        """Withdraw pro rata from actual holdings, capped at available wealth.

        The allocation remains a validated target for subsequent rebalancing;
        selling against target weights before that rebalance could short an
        absent or underweight asset. Return the amount actually funded.
        """
        validate_allocation(allocation)
        amount = finite_number(amount, context="Withdrawal amount")
        if amount < 0:
            raise ValueError("Withdrawal amount must be nonnegative.")
        self._validate_holdings()
        total = self.total_portfolio_value
        if amount >= total:
            actual_withdrawn = total
            self.holdings = {asset: 0.0 for asset in self.holdings}
            return actual_withdrawn

        remaining_fraction = (total - amount) / total
        self.holdings = {
            asset: value * remaining_fraction for asset, value in self.holdings.items()
        }
        return amount

    def apply_returns(self, returns: Dict[str, float]):
        """Apply valid simple returns atomically to held assets."""
        self._validate_holdings()
        require_return_series(
            (asset for asset, value in self.holdings.items() if value != 0.0),
            returns,
            context=f"Investor at age {self.age}",
        )
        holdings = dict(self.holdings)
        for asset, ret in returns.items():
            if asset in self.holdings:
                ret = finite_number(ret, context=f"Return for {asset}")
                if ret < -1:
                    raise ValueError(f"Return for {asset} must be at least -1.")
                holdings[asset] = finite_number(
                    holdings[asset] * (1 + ret), context=f"Holding for {asset}"
                )
        finite_number(sum(holdings.values()), context="Portfolio total")
        self.holdings = holdings

    def rebalance(self, target_allocation: Dict[str, float]):
        """Redistributes total wealth according to target weights."""
        target_allocation = validate_allocation(target_allocation)
        self._validate_holdings()
        total = self.total_portfolio_value
        self.holdings = {
            asset: total * weight for asset, weight in target_allocation.items()
        }
