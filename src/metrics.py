import statistics
from typing import List
import numpy as np

class Metrics:
    """Calculates success and risk metrics from a list of simulation outcomes."""
    
    @staticmethod
    def probability_of_ruin(terminal_wealths: List[float]) -> float:
        """The percentage of simulations that ended with zero wealth."""
        if not terminal_wealths:
            return 0.0
        ruined_count = sum(1 for w in terminal_wealths if w <= 0.0)
        return ruined_count / len(terminal_wealths)

    @staticmethod
    def percentile(terminal_wealths: List[float], p: float) -> float:
        """Calculates the p-th percentile outcome."""
        if not terminal_wealths:
            return 0.0
        return float(np.percentile(terminal_wealths, p))

    @staticmethod
    def mean(terminal_wealths: List[float]) -> float:
        return statistics.mean(terminal_wealths) if terminal_wealths else 0.0

class SavingsGapCalculator:
    """Calculates the savings rate needed for one strategy to match another's safety."""
    
    def __init__(self, simulator_factory):
        self.simulator_factory = simulator_factory

    def find_required_savings_rate(
        self, 
        target_5th_percentile: float, 
        strategy, 
        num_trials: int = 1000,
        tolerance: float = 0.01
    ) -> float:
        """
        Binary search for the savings rate that achieves the target 5th percentile wealth.
        """
        low = 0.0
        high = 1.0 # 100% savings rate
        
        for _ in range(15): # Max iterations
            mid = (low + high) / 2
            
            # Create a temporary config with the mid savings rate
            simulator = self.simulator_factory(mid)
            results = simulator.run_stochastic(strategy, num_trials=num_trials)
            current_5th = Metrics.percentile(results, 5)
            
            if abs(current_5th - target_5th_percentile) / max(target_5th_percentile, 1) < tolerance:
                return mid
            
            if current_5th < target_5th_percentile:
                low = mid
            else:
                high = mid
                
        return (low + high) / 2
