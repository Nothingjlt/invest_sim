from typing import List, Dict
from src.config import SimulationConfig
from src.investor import Investor
from src.market import Market, SyntheticMarket
from src.strategy import Strategy, FixedAllocationStrategy

class Simulator:
    """Orchestrates the lifecycle simulation for a single investor."""
    def __init__(self, config: SimulationConfig):
        self.config = config

    def run_deterministic(self, annual_return: float = 0.0) -> float:
        """
        Runs a single lifecycle simulation with a fixed annual return.
        Used for mathematical verification (Increment 2).
        Works with the multi-asset Investor by using the first market's name.
        """
        # Create a simple 100% allocation to the first market
        market_name = self.config.markets[0].name
        strategy = FixedAllocationStrategy({market_name: 1.0})
        target_alloc = strategy.get_allocation(self.config.starting_age)

        investor = Investor(
            age=self.config.starting_age,
            current_salary=self.config.initial_salary,
            holdings={market_name: 0.0}
        )

        # Main Lifecycle Loop
        while investor.age < self.config.end_age:
            # Step 1: Market Growth
            investor.apply_returns({market_name: annual_return})

            # Step 2: Income/Savings or Withdrawal
            if investor.age < self.config.retirement_age:
                # Accumulation Phase
                investor.earn_and_save(self.config.savings_rate, target_alloc)
                investor.grow_salary(self.config.salary_growth_rate)
            else:
                # Decumulation Phase
                withdrawal_amount = investor.total_portfolio_value * self.config.withdrawal_rate
                investor.withdraw(withdrawal_amount, target_alloc)

            # Step 3: Aging
            investor.age += 1

        return investor.total_portfolio_value

    def run_stochastic(
        self, 
        strategy: Strategy, 
        num_trials: int = 1000, 
        market_engine: Market = None
    ) -> List[float]:
        """
        Runs multiple lifecycle simulations with strategy-based rebalancing.
        If market_engine is not provided, defaults to SyntheticMarket using config.
        """
        terminal_wealths = []
        
        for _ in range(num_trials):
            # Use provided engine or default to Synthetic
            market = market_engine if market_engine else SyntheticMarket(self.config.markets)
            
            # If bootstrap, start a new path
            if hasattr(market, 'start_new_path'):
                market.start_new_path()

            investor = Investor(
                age=self.config.starting_age,
                current_salary=self.config.initial_salary,
                # Initialize holdings with zeros for all assets
                holdings={m.name: 0.0 for m in self.config.markets}
            )

            while investor.age < self.config.end_age:
                # 1. Determine target allocation for current age
                target_alloc = strategy.get_allocation(investor.age)
                
                # 2. Market Growth (applied to existing holdings)
                annual_returns = market.get_annual_returns()
                investor.apply_returns(annual_returns)

                # 3. Income/Savings or Withdrawal
                if investor.age < self.config.retirement_age:
                    investor.earn_and_save(self.config.savings_rate, target_alloc)
                    investor.grow_salary(self.config.salary_growth_rate)
                else:
                    withdrawal_amount = investor.total_portfolio_value * self.config.withdrawal_rate
                    investor.withdraw(withdrawal_amount, target_alloc)

                # 4. Annual Rebalancing
                # Crucial step: move money between assets to maintain target allocation
                investor.rebalance(target_alloc)

                investor.age += 1
            
            terminal_wealths.append(investor.total_portfolio_value)
            
        return terminal_wealths
