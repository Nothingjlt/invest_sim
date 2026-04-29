from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MarketConfig:
    """Configuration for a synthetic market or asset class."""
    name: str
    expected_return: float  # Annualized decimal (e.g., 0.07 for 7%)
    volatility: float       # Annualized standard deviation
    weight: float           # Portfolio weight (0.0 to 1.0)

@dataclass
class SimulationConfig:
    """Global parameters for the lifecycle simulation."""
    # Timeline
    starting_age: int = 25
    retirement_age: int = 65
    end_age: int = 100
    
    # Financials
    initial_salary: float = 50000.0
    salary_growth_rate: float = 0.03  # Real salary growth (above inflation)
    savings_rate: float = 0.10       # Percentage of salary saved (e.g., 10%)
    withdrawal_rate: float = 0.04    # Percentage of retirement portfolio withdrawn annually
    
    # Markets (Synthetic placeholders for now)
    # Allows 'choice of markets to invest' by defining asset classes here.
    markets: List[MarketConfig] = field(default_factory=lambda: [
        MarketConfig(name="Domestic Stock", expected_return=0.07, volatility=0.15, weight=1.0)
    ])

    def validate(self):
        """Basic validation to ensure parameters are logically consistent."""
        if self.starting_age >= self.retirement_age:
            raise ValueError("Starting age must be before retirement age.")
        if self.retirement_age >= self.end_age:
            raise ValueError("Retirement age must be before end age.")
        if sum(m.weight for m in self.markets) != 1.0:
            # We will handle rebalancing in later increments, 
            # but for now, weights must sum to 100%.
            raise ValueError("Total market weights must sum to 1.0.")

    @staticmethod
    def get_paper_market_configs() -> List[MarketConfig]:
        """
        Returns the 4 standard asset classes used in the paper:
        Domestic Stock, International Stock, Bonds, and Bills.
        Return values are approximate annualized real returns based on paper data.
        """
        return [
            MarketConfig(name="Domestic Stock", expected_return=0.05, volatility=0.17, weight=0.33),
            MarketConfig(name="International Stock", expected_return=0.07, volatility=0.23, weight=0.67),
            MarketConfig(name="Bonds", expected_return=0.01, volatility=0.10, weight=0.0),
            MarketConfig(name="Bills", expected_return=0.00, volatility=0.02, weight=0.0)
        ]
