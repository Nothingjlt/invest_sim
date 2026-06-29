from abc import ABC, abstractmethod
import random
import pandas as pd
from typing import List, Dict
from src.config import MarketConfig


class Market(ABC):
    """Base class for market return generators."""

    @abstractmethod
    def get_annual_returns(self) -> Dict[str, float]:
        raise NotImplementedError


class SyntheticMarket(Market):
    """Generates stochastic returns for multiple assets using Normal Distributions."""

    def __init__(self, market_configs: List[MarketConfig], seed: int | None = None):
        self.configs = market_configs
        if seed is not None:
            random.seed(seed)

    def get_annual_returns(self) -> Dict[str, float]:
        returns = {}
        for config in self.configs:
            returns[config.name] = random.gauss(
                config.expected_return, config.volatility
            )
        return returns


class BootstrapMarket(Market):
    """
    Generates returns by sampling from historical CSV data.
    Uses 'Block Bootstrap' logic to preserve contiguous historical sequences.
    """

    def __init__(self, csv_path: str, block_size: int = 10, seed: int | None = None):
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

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data from the current block."""
        if self.remaining_in_block <= 0:
            # Pick a new random start for the next block
            self.current_index = random.randint(0, len(self.data) - 1)
            self.remaining_in_block = self.block_size

        # Get data from the current index (wrap around if at end of CSV)
        idx = self.current_index % len(self.data)
        row = self.data.iloc[idx]

        # Prepare returns dict (excluding the 'Year' column)
        returns = {col: row[col] for col in self.data.columns if col != "Year"}

        # Advance state
        self.current_index += 1
        self.remaining_in_block -= 1

        return returns


class StationaryBootstrapMarket(BootstrapMarket):
    """
    Generates returns using Politis & Romano (1994) Stationary Block Bootstrap.
    Block lengths are geometrically distributed with expected value equal to block_size.
    """

    def __init__(self, csv_path: str, block_size: int = 10, seed: int | None = None):
        super().__init__(csv_path, block_size, seed)
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
        returns = {col: row[col] for col in self.data.columns if col != "Year"}
        return returns


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
        seed: int | None = None
    ):
        from src.data_loader import JSTDataLoader
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
            returns = {col: row[col] for col in self.data.columns if col != "Year"}
            return returns
        else:
            return super().get_annual_returns()

