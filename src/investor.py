from dataclasses import dataclass
from src.config import SimulationConfig

@dataclass
class Investor:
    """Tracks the state of an individual investor during the simulation."""
    age: int
    current_salary: float
    portfolio_value: float = 0.0

    def earn_and_save(self, savings_rate: float):
        """Adds a portion of the current salary to the portfolio."""
        contribution = self.current_salary * savings_rate
        self.portfolio_value += contribution
        return contribution

    def grow_salary(self, growth_rate: float):
        """Increases salary by the growth rate."""
        self.current_salary *= (1 + growth_rate)

    def withdraw(self, amount: float):
        """Removes a specific amount from the portfolio."""
        if amount > self.portfolio_value:
            # Portfolio is ruined (hit zero or below)
            withdrawn = self.portfolio_value
            self.portfolio_value = 0.0
            return withdrawn
        
        self.portfolio_value -= amount
        return amount
