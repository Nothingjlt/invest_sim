from abc import ABC, abstractmethod
import math
import random
import pandas as pd
from typing import List, Dict
from src.config import MarketConfig
from src.provenance import DataProvenance, csv_provenance


class Market(ABC):
    """Base class for market return generators."""

    # Custom engines may override this with an explicit source declaration.
    provenance = DataProvenance()

    @abstractmethod
    def get_annual_returns(self) -> Dict[str, float]:
        raise NotImplementedError


class SyntheticMarket(Market):
    """Generates bounded simple returns with lognormal gross returns."""

    provenance = DataProvenance("synthetic", "SyntheticMarket lognormal model")

    def __init__(self, market_configs: List[MarketConfig], seed: int | None = None):
        self.configs = market_configs
        self._lognormal_params = []
        for config in self.configs:
            try:
                expected_return = float(config.expected_return)
                volatility = float(config.volatility)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Market '{config.name}' expected_return and volatility "
                    "must be finite numbers."
                ) from exc

            if not math.isfinite(expected_return) or expected_return <= -1.0:
                raise ValueError(
                    f"Market '{config.name}' expected_return must be finite "
                    "and greater than -1."
                )
            if not math.isfinite(volatility) or volatility < 0.0:
                raise ValueError(
                    f"Market '{config.name}' volatility must be finite and "
                    "non-negative."
                )

            # Convert arithmetic simple-return moments to the parameters of
            # log gross returns. This keeps the configured mean and standard
            # deviation while ensuring every sampled gross return is positive.
            relative_volatility = volatility / (1.0 + expected_return)
            log_variance = 2.0 * math.log(math.hypot(1.0, relative_volatility))
            log_mean = math.log1p(expected_return) - 0.5 * log_variance
            if not math.isfinite(log_mean) or not math.isfinite(log_variance):
                raise ValueError(
                    f"Market '{config.name}' expected_return and volatility "
                    "are outside the supported numeric range."
                )
            self._lognormal_params.append(
                (config.name, expected_return, log_mean, math.sqrt(log_variance))
            )

        if seed is not None:
            random.seed(seed)

    def get_annual_returns(self) -> Dict[str, float]:
        returns = {}
        for name, expected_return, log_mean, log_volatility in self._lognormal_params:
            if log_volatility == 0.0:
                returns[name] = expected_return
            else:
                returns[name] = math.exp(
                    random.gauss(log_mean, log_volatility)
                ) - 1.0
        return returns


class BootstrapMarket(Market):
    """
    Generates returns by sampling CSV data with contiguous blocks.
    The source is unverified unless explicit provenance is provided or the
    input matches the known bundled synthetic panel.
    """

    def __init__(
        self, csv_path: str, block_size: int = 10, seed: int | None = None,
        *, provenance: DataProvenance | None = None,
    ):
        self.provenance = csv_provenance(csv_path, provenance)
        self.data = pd.read_csv(csv_path)
        self.block_size = block_size
        if seed is not None:
            random.seed(seed)

        # State for current simulation path
        self.current_index = 0
        self.remaining_in_block = 0

    def start_new_path(self):
        """Reset state for a new simulation trial."""
        self.current_index = random.randint(0, len(self.data) - 1)
        self.remaining_in_block = self.block_size

    def _observed_returns(self, row: pd.Series) -> Dict[str, float]:
        """Keep observed series without discarding a partially covered year.

        An absent key lets active-asset validation reject unavailable holdings;
        a NaN value would incorrectly present the series as available. Other
        non-finite or non-numeric values are invalid input and fail closed.
        """
        returns = {}
        for col, value in row.items():
            if col == "Year" or pd.isna(value):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Bootstrap return for {col} must be numeric and finite."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Bootstrap return for {col} must be numeric and finite."
                )
            returns[col] = value
        return returns

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data from the current block."""
        if self.remaining_in_block <= 0:
            # Pick a new random start for the next block
            self.current_index = random.randint(0, len(self.data) - 1)
            self.remaining_in_block = self.block_size

        # Get data from the current index (wrap around if at end of CSV)
        idx = self.current_index % len(self.data)
        row = self.data.iloc[idx]

        returns = self._observed_returns(row)

        # Advance state
        self.current_index += 1
        self.remaining_in_block -= 1

        return returns


class StationaryBootstrapMarket(BootstrapMarket):
    """
    Generates returns using Politis & Romano (1994) Stationary Block Bootstrap.
    Block lengths are geometrically distributed with expected value equal to block_size.
    """

    def __init__(
        self, csv_path: str, block_size: int = 10, seed: int | None = None,
        *, provenance: DataProvenance | None = None,
    ):
        super().__init__(csv_path, block_size, seed, provenance=provenance)
        self.first_step = True

    def start_new_path(self):
        """Reset state for a new simulation trial."""
        self.first_step = True
        self.current_index = random.randint(0, len(self.data) - 1)

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data, jumping to a new block with probability 1/block_size."""
        if self.first_step:
            self.first_step = False
        else:
            # Jump with probability 1 / block_size
            if random.random() < (1.0 / self.block_size):
                self.current_index = random.randint(0, len(self.data) - 1)
            else:
                self.current_index = (self.current_index + 1) % len(self.data)

        row = self.data.iloc[self.current_index]
        return self._observed_returns(row)


class PerspectiveBootstrapMarket(BootstrapMarket):
    """
    Generates returns by running a perspective-country-aware bootstrap.
    Uses JSTDataLoader to build a customized panel of FX-adjusted real returns.
    """

    def __init__(
        self,
        csv_path: str,
        perspective_country: str = "USA",
        block_size: int = 10,
        stationary_bootstrap: bool = False,
        weight_method: str = "gdp",
        seed: int | None = None,
        *, provenance: DataProvenance | None = None,
    ):
        from src.data_loader import JSTDataLoader
        self.provenance = csv_provenance(csv_path, provenance)
        loader = JSTDataLoader(csv_path)
        processed_data = loader.get_processed_returns(
            perspective_country=perspective_country,
            weight_method=weight_method
        )

        self.data = processed_data
        self.block_size = block_size
        self.stationary_bootstrap = stationary_bootstrap
        if seed is not None:
            random.seed(seed)

        self.current_index = 0
        self.remaining_in_block = 0
        self.first_step = True

    def start_new_path(self):
        """Reset state for a new simulation trial."""
        self.first_step = True
        self.current_index = random.randint(0, len(self.data) - 1)
        self.remaining_in_block = self.block_size

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data using the specified bootstrap method."""
        if self.stationary_bootstrap:
            if self.first_step:
                self.first_step = False
            else:
                if random.random() < (1.0 / self.block_size):
                    self.current_index = random.randint(0, len(self.data) - 1)
                else:
                    self.current_index = (self.current_index + 1) % len(self.data)

            row = self.data.iloc[self.current_index]
            return self._observed_returns(row)
        else:
            return super().get_annual_returns()
