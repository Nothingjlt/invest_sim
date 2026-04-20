import random
from src.config import MarketConfig

class Market:
    """Generates stochastic returns based on market configuration."""
    def __init__(self, market_config: MarketConfig, seed: int = None):
        self.config = market_config
        if seed is not None:
            random.seed(seed)

    def get_annual_return(self) -> float:
        """Draws a single year's return from a normal distribution."""
        # Note: Normal Distribution is a baseline; Increment 7 will implement Block Bootstrap.
        return random.gauss(self.config.expected_return, self.config.volatility)
