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


class WorldEquityStrategy(Strategy):
    """
    A sophisticated global equity strategy that allocates across regions and countries.
    Supports nested weights for hierarchical allocation.
    """

    def __init__(self, region_weights: Dict[str, Dict[str, float]]):
        self.region_weights = region_weights
        self._validate_weights()

    def _validate_weights(self):
        total_weight = 0.0
        seen_countries = set()
        for region, countries in self.region_weights.items():
            region_sum = sum(countries.values())
            if abs(region_sum - 1.0) > 1e-6:
                # If region sum isn't 1.0, we assume the user provided absolute portfolio weights
                # instead of relative weights within the region.
                pass

            for country in countries:
                if country in seen_countries:
                    raise ValueError(
                        f"Country {country} specified in multiple regions."
                    )
                seen_countries.add(country)

            total_weight += sum(countries.values())

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Total portfolio weight must sum to 1.0 (got {total_weight})"
            )

    def get_allocation(self, age: int) -> Dict[str, float]:
        flat_allocation = {}
        for region, countries in self.region_weights.items():
            for country, weight in countries.items():
                flat_allocation[country] = weight
        return flat_allocation


class GlidePathStrategy(Strategy):
    """
    A traditional glide path strategy (Target Date Fund).
    Shifts from stocks to bonds linearly between a starting age and a retirement age.
    Now supports granular asset classes within the equity and bond portions.
    """

    def __init__(
        self,
        start_age: int,
        retire_age: int,
        start_equity: float = 0.90,
        end_equity: float = 0.30,
        equity_assets: Dict[str, float] | None = None,
        bond_assets: Dict[str, float] | None = None,
    ):
        self.start_age = start_age
        self.retire_age = retire_age
        self.start_equity = start_equity
        self.end_equity = end_equity

        # Default to "Stocks" and "Bonds" for backward compatibility
        self.equity_assets = equity_assets or {"Stocks": 1.0}
        self.bond_assets = bond_assets or {"Bonds": 1.0}

        # Validation
        if abs(sum(self.equity_assets.values()) - 1.0) > 1e-6:
            raise ValueError("Equity asset weights must sum to 1.0")
        if abs(sum(self.bond_assets.values()) - 1.0) > 1e-6:
            raise ValueError("Bond asset weights must sum to 1.0")

    def get_allocation(self, age: int) -> Dict[str, float]:
        """Calculates the granular asset split for a given age."""
        if age <= self.start_age:
            equity_pct = self.start_equity
        elif age >= self.retire_age:
            equity_pct = self.end_equity
        else:
            # Linear interpolation between start and retirement age
            progress = (age - self.start_age) / (self.retire_age - self.start_age)
            equity_pct = self.start_equity - progress * (
                self.start_equity - self.end_equity
            )

        bond_pct = 1.0 - equity_pct

        # Combine granular assets
        allocation = {}
        for asset, weight in self.equity_assets.items():
            allocation[asset] = equity_pct * weight
        for asset, weight in self.bond_assets.items():
            allocation[asset] = allocation.get(asset, 0.0) + (bond_pct * weight)

        return allocation


class BalancedStrategy(Strategy):
    """A traditional balanced strategy (60% Domestic Stocks, 40% Bonds)."""

    def __init__(
        self, domestic_label: str = "Domestic Stock", bond_label: str = "Bonds"
    ):
        self.domestic_label = domestic_label
        self.bond_label = bond_label

    def get_allocation(self, age: int) -> Dict[str, float]:
        return {self.domestic_label: 0.60, self.bond_label: 0.40}


class PaperOptimalStrategy(Strategy):
    """
    The optimal age-based strategy recommended by the paper.
    - 100% Equity (33% Domestic, 67% Intl) for most of the lifecycle.
    - Tactical cash (Bills) allocation near retirement.
    """

    def __init__(
        self,
        retire_age: int = 65,
        dom_label: str = "USA",
        intl_assets: Dict[str, float] | None = None,
        bills_label: str = "Bills",
    ):
        self.retire_age = retire_age
        self.dom_label = dom_label
        # Default to a broad international developed set if not provided
        self.intl_assets = intl_assets or {
            "GBR": 0.3,
            "JPN": 0.3,
            "FRA": 0.2,
            "DEU": 0.2,
        }
        self.bills_label = bills_label

        # Validation for intl_assets
        if abs(sum(self.intl_assets.values()) - 1.0) > 1e-6:
            raise ValueError("International asset weights must sum to 1.0")

    def get_allocation(self, age: int) -> Dict[str, float]:
        # Before retirement or after age 70: standard 100% equity split (33/67)
        if age < self.retire_age or age >= 70:
            dom_weight = 0.33
            intl_weight = 0.67
        else:
            # Transition at retirement (Age 65-70)
            # Age 65: 26% Dom, 47% Intl, 27% Bills
            # Age 70: 33% Dom, 67% Intl, 0% Bills
            progress = (age - self.retire_age) / (70 - self.retire_age)
            bills_weight = 0.27 * (1 - progress)
            dom_weight = 0.26 + (0.33 - 0.26) * progress
            intl_weight = 1.0 - bills_weight - dom_weight

        # Map weights to actual assets
        allocation = {self.dom_label: dom_weight}
        for asset, rel_weight in self.intl_assets.items():
            allocation[asset] = intl_weight * rel_weight

        if age >= self.retire_age and age < 70:
            allocation[self.bills_label] = 1.0 - sum(
                v for k, v in allocation.items() if k != self.bills_label
            )

        return allocation


class PaperTDFStrategy(Strategy):
    """
    A representative Target Date Fund (TDF) glide path based on the paper.
    - Age 25: 54% Domestic, 36% Intl, 10% Bonds, 0% Bills.
    - Age 65: 10% Domestic, 7% Intl, 73% Bonds, 10% Bills.
    """

    def __init__(
        self,
        start_age: int = 25,
        retire_age: int = 65,
        dom_label: str = "USA",
        intl_assets: Dict[str, float] | None = None,
        bond_label: str = "Bonds",
        bills_label: str = "Bills",
    ):
        self.start_age = start_age
        self.retire_age = retire_age
        self.dom_label = dom_label
        self.intl_assets = intl_assets or {
            "GBR": 0.3,
            "JPN": 0.3,
            "FRA": 0.2,
            "DEU": 0.2,
        }
        self.bond_label = bond_label
        self.bills_label = bills_label

        if abs(sum(self.intl_assets.values()) - 1.0) > 1e-6:
            raise ValueError("International asset weights must sum to 1.0")

    def get_allocation(self, age: int) -> Dict[str, float]:
        if age <= self.start_age:
            p = 0.0
        elif age >= self.retire_age:
            p = 1.0
        else:
            p = (age - self.start_age) / (self.retire_age - self.start_age)

        # Interpolation between start and retirement
        dom_w = 0.54 + (0.10 - 0.54) * p
        intl_w = 0.36 + (0.07 - 0.36) * p
        bonds_w = 0.10 + (0.73 - 0.10) * p
        bills_w = 0.00 + (0.10 - 0.00) * p

        allocation = {
            self.dom_label: dom_w,
            self.bond_label: bonds_w,
            self.bills_label: bills_w,
        }

        for asset, rel_weight in self.intl_assets.items():
            allocation[asset] = intl_w * rel_weight

        return allocation
