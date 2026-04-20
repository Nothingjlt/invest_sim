import random
from typing import List, Dict
from src.config import MarketConfig

class Market:
    """Generates stochastic returns for multiple assets."""
    def __init__(self, market_configs: List[MarketConfig], seed: int = None):
        self.configs = market_configs
        if seed is not None:
            random.seed(seed)

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns a dictionary mapping asset names to annual returns."""
        returns = {}
        for config in self.configs:
            returns[config.name] = random.gauss(config.expected_return, config.volatility)
        return returns
