from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping
from functools import wraps

from src.assets import (
    DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS,
    finite_number, validate_allocation,
)


def _validate_timeline(start_age, retire_age):
    for name, age in [("start_age", start_age), ("retire_age", retire_age)]:
        if finite_number(age, context=name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if start_age >= retire_age:
        raise ValueError("start_age must be before retire_age.")


class Strategy(ABC):
    """Abstract base class for investment strategies.

    Composite strategies add component weights when holding labels overlap.
    WorldEquityStrategy rejects countries repeated across regions, since region
    entries are absolute holdings rather than separate portfolio components.
    """

    @staticmethod
    def _validate_age(func):
        @wraps(func)
        def wrapper(self, age: float, *args, **kwargs):
            if finite_number(age, context="Age") <= 0:
                raise ValueError("Age must be positive.")
            return func(self, age, *args, **kwargs)
        return wrapper

    @_validate_age
    def get_allocation(self, age: float) -> Dict[str, float]:
        """Returns the target portfolio allocation for a given age."""
        return validate_allocation(self._get_allocation(age))

    def resolve_allocation_for_market(
        self, allocation: Dict[str, float], available_assets: Iterable[str]
    ) -> Dict[str, float]:
        """Resolve an allocation against the current market's return series."""
        return validate_allocation(allocation)

    @abstractmethod
    def _get_allocation(self, age: float) -> Dict[str, float]:
        """Subclasses implement this to provide the allocation logic."""
        pass


class FixedAllocationStrategy(Strategy):
    """A strategy where the allocation remains constant (e.g., 100% Equity)."""

    def __init__(self, target_allocation: Dict[str, float]):
        self.target_allocation = validate_allocation(target_allocation)

    def _get_allocation(self, age: float) -> Dict[str, float]:
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
        if not isinstance(self.region_weights, Mapping):
            raise ValueError("Region weights must be a mapping.")
        flat_allocation = {}
        seen_countries = set()
        for region, countries in self.region_weights.items():
            if not isinstance(countries, Mapping):
                raise ValueError(f"Country weights for {region} must be a mapping.")
            for country in countries:
                if country in seen_countries:
                    raise ValueError(
                        f"Country {country} specified in multiple regions."
                    )
                seen_countries.add(country)
            flat_allocation.update(countries)
        validate_allocation(flat_allocation, context="Total portfolio")

    def _get_allocation(self, age: float) -> Dict[str, float]:
        self._validate_weights()
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
        self.equity_assets = validate_allocation(
            {"Stocks": 1.0} if equity_assets is None else equity_assets,
            context="Equity asset",
        )
        self.bond_assets = validate_allocation(
            {"Bonds": 1.0} if bond_assets is None else bond_assets,
            context="Bond asset",
        )
        self._validate_parameters()

    def _validate_parameters(self):
        _validate_timeline(self.start_age, self.retire_age)
        for name in ("start_equity", "end_equity"):
            if not 0 <= finite_number(getattr(self, name), context=name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        self.equity_assets = validate_allocation(self.equity_assets, context="Equity asset")
        self.bond_assets = validate_allocation(self.bond_assets, context="Bond asset")

    def _get_allocation(self, age: float) -> Dict[str, float]:
        """Calculates the granular asset split for a given age."""
        self._validate_parameters()
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
        self, domestic_label: str = DOMESTIC_STOCK, bond_label: str = BONDS
    ):
        self.domestic_label = domestic_label
        self.bond_label = bond_label

    def _get_allocation(self, age: float) -> Dict[str, float]:
        # Components with the same holding label add their weights.
        allocation = {self.domestic_label: 0.60}
        allocation[self.bond_label] = allocation.get(self.bond_label, 0.0) + 0.40
        return allocation


class _PaperStrategy(Strategy):
    _country_international_assets = {"GBR": 0.3, "JPN": 0.3, "FRA": 0.2, "DEU": 0.2}

    def resolve_allocation_for_market(
        self, allocation: Dict[str, float], available_assets: Iterable[str]
    ) -> Dict[str, float]:
        allocation = validate_allocation(allocation)
        # Subclasses can provide their own constructor and allocation hook.
        # Only resolve defaults recorded by a paper strategy constructor.
        if not hasattr(self, "_default_domestic") or not hasattr(self, "_default_international"):
            return allocation
        available = set(available_assets)
        explicit_assets = set()
        if not self._default_domestic:
            explicit_assets.add(self.dom_label)
        if not self._default_international:
            explicit_assets.update(self.intl_assets)
        if hasattr(self, "bond_label"):
            # The TDF uses bonds and bills as active components. The fixed
            # optimal strategy retains bills_label only for call compatibility.
            explicit_assets.add(self.bills_label)
            explicit_assets.add(self.bond_label)

        replacements = {}
        if (
            self._default_domestic and DOMESTIC_STOCK not in explicit_assets
            and DOMESTIC_STOCK not in available and "USA" in available
        ):
            replacements[DOMESTIC_STOCK] = {"USA": 1.0}
        if (
            self._default_international and INTERNATIONAL_STOCK not in explicit_assets
            and INTERNATIONAL_STOCK not in available
            and self._country_international_assets.keys() <= available
        ):
            replacements[INTERNATIONAL_STOCK] = self._country_international_assets

        # Keep the countries as holdings so monthly conversion and rebalancing
        # operate on each constituent rather than an averaged annual return.
        resolved = {}
        for asset, weight in allocation.items():
            for constituent, relative_weight in replacements.get(asset, {asset: 1.0}).items():
                resolved[constituent] = resolved.get(constituent, 0.0) + weight * relative_weight
        return validate_allocation(resolved, context="Resolved allocation")


class PaperOptimalStrategy(_PaperStrategy):
    """
    Fixed-weight all-equity approximation of the paper's optimal portfolio.

    The paper reports 34% domestic stocks and 66% international stocks for its
    optimal fixed-weight strategy, with no bonds or bills. Its separate optimal
    age-based strategy varies by age across 13 allocation windows; reproducing
    that schedule requires age-specific weights not represented by this class.

    ``retire_age`` and ``bills_label`` remain accepted for call compatibility;
    this strategy has no retirement-date cash transition.
    """

    def __init__(
        self,
        retire_age: int = 65,
        dom_label: str | None = None,
        intl_assets: Dict[str, float] | None = None,
        bills_label: str = BILLS,
    ):
        self.retire_age = retire_age
        self._default_domestic = dom_label is None
        self._default_international = intl_assets is None
        self.dom_label = DOMESTIC_STOCK if dom_label is None else dom_label
        self.intl_assets = (
            {INTERNATIONAL_STOCK: 1.0}
            if intl_assets is None
            else dict(intl_assets)
        )
        self.bills_label = bills_label

        self.intl_assets = validate_allocation(self.intl_assets, context="International asset")

    def _get_allocation(self, age: float) -> Dict[str, float]:
        self.intl_assets = validate_allocation(self.intl_assets, context="International asset")
        # Table III reports the paper's optimal fixed-weight allocation as 34/66.
        # Map that weight to the caller's selected international constituents.
        allocation = {self.dom_label: 0.34}
        for asset, rel_weight in self.intl_assets.items():
            allocation[asset] = allocation.get(asset, 0.0) + 0.66 * rel_weight

        return allocation


class PaperTDFStrategy(_PaperStrategy):
    """
    Linear approximation between the TDF allocation extremes reported in the paper.

    Table III reports allocation ranges, while the age-specific path is shown in
    Figure 1. This class does not reproduce that plotted curve; it interpolates
    between the configured starting and retirement ages, then holds the terminal
    allocation constant.

    - Age 25: 54% Domestic, 36% Intl, 10% Bonds, 0% Bills.
    - Age 65: 10% Domestic, 7% Intl, 73% Bonds, 10% Bills.
    """

    def __init__(
        self,
        start_age: int = 25,
        retire_age: int = 65,
        dom_label: str | None = None,
        intl_assets: Dict[str, float] | None = None,
        bond_label: str = BONDS,
        bills_label: str = BILLS,
    ):
        self.start_age = start_age
        self.retire_age = retire_age
        self._default_domestic = dom_label is None
        self._default_international = intl_assets is None
        self.dom_label = DOMESTIC_STOCK if dom_label is None else dom_label
        self.intl_assets = (
            {INTERNATIONAL_STOCK: 1.0}
            if intl_assets is None
            else dict(intl_assets)
        )
        self.bond_label = bond_label
        self.bills_label = bills_label

        _validate_timeline(self.start_age, self.retire_age)
        self.intl_assets = validate_allocation(self.intl_assets, context="International asset")

    def _get_allocation(self, age: float) -> Dict[str, float]:
        _validate_timeline(self.start_age, self.retire_age)
        self.intl_assets = validate_allocation(self.intl_assets, context="International asset")
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

        # Components can refer to the same holding; preserve each one's weight.
        allocation = {self.dom_label: dom_w}
        for asset, weight in [(self.bond_label, bonds_w), (self.bills_label, bills_w)]:
            allocation[asset] = allocation.get(asset, 0.0) + weight

        for asset, rel_weight in self.intl_assets.items():
            allocation[asset] = allocation.get(asset, 0.0) + intl_w * rel_weight

        return allocation
