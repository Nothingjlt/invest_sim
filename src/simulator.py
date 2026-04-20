from src.config import SimulationConfig
from src.investor import Investor

class Simulator:
    """Orchestrates the lifecycle simulation for a single investor."""
    def __init__(self, config: SimulationConfig):
        self.config = config

    def run_deterministic(self, annual_return: float = 0.0) -> float:
        """
        Runs a single lifecycle simulation with a fixed annual return.
        Used for mathematical verification (Increment 2).
        """
        investor = Investor(
            age=self.config.starting_age,
            current_salary=self.config.initial_salary
        )

        # Main Lifecycle Loop
        while investor.age < self.config.end_age:
            # Step 1: Market Growth (Beginning of year)
            investor.portfolio_value *= (1 + annual_return)

            # Step 2: Income/Savings or Withdrawal
            if investor.age < self.config.retirement_age:
                # Accumulation Phase
                investor.earn_and_save(self.config.savings_rate)
                investor.grow_salary(self.config.salary_growth_rate)
            else:
                # Decumulation Phase
                withdrawal_amount = investor.portfolio_value * self.config.withdrawal_rate
                investor.withdraw(withdrawal_amount)

            # Step 3: Aging
            investor.age += 1

        return investor.portfolio_value
