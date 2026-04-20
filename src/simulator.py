from typing import List
from src.config import SimulationConfig
from src.investor import Investor
from src.market import Market

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

    def run_stochastic(self, num_trials: int = 1000) -> List[float]:
        """
        Runs multiple lifecycle simulations using stochastic market returns.
        """
        terminal_wealths = []
        # For Increment 3, we use the first market in the config as a baseline.
        market_config = self.config.markets[0]
        
        for _ in range(num_trials):
            market = Market(market_config)
            investor = Investor(
                age=self.config.starting_age,
                current_salary=self.config.initial_salary
            )

            while investor.age < self.config.end_age:
                # Stochastic Market Growth
                annual_return = market.get_annual_return()
                investor.portfolio_value *= (1 + annual_return)

                # Savings or Withdrawal
                if investor.age < self.config.retirement_age:
                    invest.earn_and_save(self.config.savings_rate)
                    investor.grow_salary(self.config.salary_growth_rate)
                else:
                    withdrawal_amount = investor.portfolio_value * self.config.withdrawal_rate
                    investor.withdraw(withdrawal_amount)

                investor.age += 1
            
            terminal_wealths.append(investor.portfolio_value)
            
        return terminal_wealths
