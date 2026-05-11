from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MarketConfig:
    """Configuration for a synthetic market or asset class."""

    name: str
    expected_return: float  # Annualized decimal (e.g., 0.07 for 7%)
    volatility: float  # Annualized standard deviation
    weight: float  # Portfolio weight (0.0 to 1.0)


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
    savings_rate: float = 0.10  # Percentage of salary saved (e.g., 10%)
    withdrawal_rate: float = (
        0.04  # Percentage of retirement portfolio withdrawn annually
    )
    withdrawal_strategy: str = "variable_pct"  # "variable_pct" or "fixed_real"
    social_security_benefit: float = 0.0  # Annual real benefit
    enable_mortality: bool = False  # Use variable lifespans

    # Markets (Synthetic placeholders for now)
    # Allows 'choice of markets to invest' by defining asset classes here.
    markets: List[MarketConfig] = field(
        default_factory=lambda: [
            MarketConfig(
                name="Domestic Stock", expected_return=0.07, volatility=0.15, weight=1.0
            )
        ]
    )

    # Classification Transition Registry (Historical Accuracy)
    # Maps country names to the year they were first classified as 'Developed'.
    # Used to prevent double-counting in World portfolios.
    DEVELOPED_TRANSITIONS: Dict[str, int] = field(
        default_factory=lambda: {
            "USA": 1890,
            "GBR": 1890,
            "FRA": 1890,
            "DEU": 1890,
            "JPN": 1930,
            "CAN": 1891,
            "AUS": 1901,
            "MEX": 1994,
            "KOR": 1996,
            "POL": 1996,
            "GRC": 1920,
            "TUR": 1948,
            "CHL": 2010,
            "CZE": 1995,
            "HUN": 1996,
        }
    )

    def validate(self):
        """Basic validation to ensure parameters are logically consistent."""
        if self.starting_age >= self.retirement_age:
            raise ValueError("Starting age must be before retirement age.")
        if self.retirement_age >= self.end_age:
            raise ValueError("Retirement age must be before end age.")
        if sum(m.weight for m in self.markets) != 1.0:
            raise ValueError("Total market weights must sum to 1.0.")

        # Check for Market Classification Collisions
        # Ensure no country is treated as both Developed and Emerging in the same context
        asset_names = [m.name for m in self.markets]
        if len(asset_names) != len(set(asset_names)):
            raise ValueError("Duplicate asset names detected in market configuration.")

    @staticmethod
    def get_paper_market_configs() -> List[MarketConfig]:
        """
        Returns the 4 standard asset classes used in the paper:
        Domestic Stock, International Stock, Bonds, and Bills.
        Return values are approximate annualized real returns based on paper data.
        """
        return [
            MarketConfig(
                name="Domestic Stock",
                expected_return=0.05,
                volatility=0.17,
                weight=0.33,
            ),
            MarketConfig(
                name="International Stock",
                expected_return=0.07,
                volatility=0.23,
                weight=0.67,
            ),
            MarketConfig(
                name="Bonds", expected_return=0.01, volatility=0.10, weight=0.0
            ),
            MarketConfig(
                name="Bills", expected_return=0.00, volatility=0.02, weight=0.0
            ),
        ]

    @staticmethod
    def get_world_market_configs() -> List[MarketConfig]:
        """
        Returns a broad set of world market configurations for Developed and Emerging indices.
        Based on pooled statistical moments from global research.
        """
        configs = []
        # Representative Developed Markets (Pooled averages)
        for country in ["USA", "GBR", "JPN", "DEU", "FRA", "CAN", "AUS"]:
            configs.append(
                MarketConfig(
                    name=country, expected_return=0.05, volatility=0.17, weight=0.0
                )
            )

        # Representative Emerging Markets (Pooled averages)
        for country in ["CHN", "IND", "BRA", "ZAF", "ARE"]:
            configs.append(
                MarketConfig(
                    name=country, expected_return=0.10, volatility=0.21, weight=0.0
                )
            )

        # Global Fixed Income
        configs.append(
            MarketConfig(
                name="Bonds", expected_return=0.01, volatility=0.10, weight=0.0
            )
        )
        configs.append(
            MarketConfig(
                name="Bills", expected_return=0.00, volatility=0.02, weight=0.0
            )
        )

        return configs
