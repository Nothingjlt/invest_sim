from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.assets import (
    DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS,
    finite_number, validate_allocation,
)


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

    # Withdrawal modifiers
    withdrawal_inflation_adjusted: bool = False
    # If True, grow the fixed_real withdrawal (and optional cap/floor) by realized
    # inflation sourced from the market engine each year.

    withdrawal_cap: Optional[float] = None
    # Maximum gross withdrawal per year (same real units as initial_salary).
    # Applied before the social-security offset.

    withdrawal_cap_inflation_adjusted: bool = False
    # If True, grow withdrawal_cap by realized inflation each year.

    withdrawal_floor: Optional[float] = None
    # Minimum real expenditure per year. The portfolio makes up any shortfall
    # not covered by social_security_benefit.

    withdrawal_floor_inflation_adjusted: bool = False
    # If True, grow withdrawal_floor by realized inflation each year.

    decumulation_granularity: str = "annual"
    # Controls the time-step used during the retirement (decumulation) phase.
    # "annual"  – one withdrawal per year (default, backward-compatible).
    # "monthly" – withdrawals at the beginning of each calendar month. If the
    #              market supplies annual returns, each annual return is
    #              smoothed into twelve equal compounded monthly returns:
    #              r_m = (1 + R_annual)^(1/12) - 1.

    # Markets (Synthetic placeholders for now)
    # Allows 'choice of markets to invest' by defining asset classes here.
    markets: List[MarketConfig] = field(
        default_factory=lambda: [
            MarketConfig(
                name=DOMESTIC_STOCK, expected_return=0.07, volatility=0.15, weight=1.0
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
        for name in ("starting_age", "retirement_age", "end_age"):
            age = getattr(self, name)
            if isinstance(age, bool) or not isinstance(age, int) or age <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.starting_age >= self.retirement_age:
            raise ValueError("Starting age must be before retirement age.")
        if self.retirement_age >= self.end_age:
            raise ValueError("Retirement age must be before end age.")
        if not isinstance(self.markets, (list, tuple)) or not self.markets:
            raise ValueError("markets must contain at least one MarketConfig.")
        for market in self.markets:
            if not isinstance(market, MarketConfig):
                raise ValueError("markets must contain MarketConfig entries.")
            if not isinstance(market.name, str) or not market.name.strip():
                raise ValueError("Market asset names must be nonempty strings.")
        # Check for Market Classification Collisions
        # Ensure no country is treated as both Developed and Emerging in the same context
        asset_names = [m.name for m in self.markets]
        if len(asset_names) != len(set(asset_names)):
            raise ValueError("Duplicate asset names detected in market configuration.")
        validate_allocation(
            {m.name: m.weight for m in self.markets}, context="Total market"
        )
        for market in self.markets:
            expected_return = finite_number(market.expected_return, context="expected_return")
            if expected_return <= -1:
                raise ValueError("expected_return must be greater than -1.")
            if finite_number(market.volatility, context="volatility") < 0:
                raise ValueError("volatility must be nonnegative.")

        for name in ("initial_salary", "social_security_benefit", "withdrawal_cap", "withdrawal_floor"):
            value = getattr(self, name)
            if value is None and name in {"withdrawal_cap", "withdrawal_floor"}:
                continue
            if finite_number(value, context=name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in ("savings_rate", "withdrawal_rate"):
            if not 0 <= finite_number(getattr(self, name), context=name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if finite_number(self.salary_growth_rate, context="salary_growth_rate") < -1:
            raise ValueError("salary_growth_rate must be at least -1.")
        if not isinstance(self.withdrawal_strategy, str) or self.withdrawal_strategy not in {"variable_pct", "fixed_real"}:
            raise ValueError("withdrawal_strategy must be 'variable_pct' or 'fixed_real'.")
        for name in (
            "enable_mortality", "withdrawal_inflation_adjusted",
            "withdrawal_cap_inflation_adjusted", "withdrawal_floor_inflation_adjusted",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool.")

        # Decumulation granularity
        if not isinstance(self.decumulation_granularity, str) or self.decumulation_granularity not in {"annual", "monthly"}:
            raise ValueError(
                f"decumulation_granularity must be 'annual' or 'monthly', "
                f"got {self.decumulation_granularity!r}."
            )

        # Withdrawal cap/floor consistency
        if (
            self.withdrawal_cap is not None
            and self.withdrawal_floor is not None
            and self.withdrawal_cap < self.withdrawal_floor
        ):
            raise ValueError(
                "withdrawal_cap must be >= withdrawal_floor when both are set."
            )

    @staticmethod
    def get_paper_market_configs() -> List[MarketConfig]:
        """
        Returns the 4 standard asset classes used in the paper:
        Domestic Stock, International Stock, Bonds, and Bills.
        Return values are approximate annualized real returns based on paper data.
        """
        return [
            MarketConfig(
                name=DOMESTIC_STOCK,
                expected_return=0.05,
                volatility=0.17,
                weight=0.34,
            ),
            MarketConfig(
                name=INTERNATIONAL_STOCK,
                expected_return=0.07,
                volatility=0.23,
                weight=0.66,
            ),
            MarketConfig(
                name=BONDS, expected_return=0.01, volatility=0.10, weight=0.0
            ),
            MarketConfig(
                name=BILLS, expected_return=0.00, volatility=0.02, weight=0.0
            ),
        ]

    @staticmethod
    def get_paper_bootstrap_config(perspective: str = "USA") -> 'SimulationConfig':
        """
        Returns a SimulationConfig pre-configured with the paper's standard asset classes.
        """
        return SimulationConfig(
            markets=SimulationConfig.get_paper_market_configs()
        )


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
                name=BONDS, expected_return=0.01, volatility=0.10, weight=0.0
            )
        )
        configs.append(
            MarketConfig(
                name=BILLS, expected_return=0.00, volatility=0.02, weight=0.0
            )
        )

        return configs
