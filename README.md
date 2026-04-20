# Lifecycle Investment Simulation Tool

A Python-based simulation engine inspired by the research paper **"Beyond the Status Quo: A Critical Assessment of Lifecycle Investment Advice"** (Anarkulova, Cederburg, and O'Doherty, 2023). 

This tool allows investors to compare traditional "Glide Path" (Target Date Fund) strategies against the paper's recommendation of **100% Equities with Geographic Diversification** using both synthetic and historical market data.

## Features

- **Stochastic Lifecycle Modeling**: Monte Carlo simulations tracking an investor from age 25 to 100.
- **Interchangeable Strategies**: Easily switch between `FixedAllocationStrategy` and `GlidePathStrategy`.
- **Advanced Market Engines**:
    - `SyntheticMarket`: Normal distribution modeling (Mean/Volatility).
    - `BootstrapMarket`: **Block Bootstrap** sampling from real-world historical data to preserve market cycles.
- **Success Metrics**:
    - **Probability of Ruin**: Frequency of hitting $0 in retirement.
    - **Tail Risk (5th Percentile)**: Identifying worst-case wealth outcomes.
    - **Savings Gap Calculator**: Finding the savings rate needed for one strategy to match another's safety.

## Installation

1. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(Note: Requirements include `pandas`, `numpy`, and `pytest` for testing.)*

2. **Run Tests**:
   ```bash
   PYTHONPATH=. pytest tests/
   ```

## Usage Example

The following script compares a 100% Equity strategy against a traditional Glide Path using the provided historical data.

```python
from src.config import SimulationConfig, MarketConfig
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy, GlidePathStrategy
from src.market import BootstrapMarket
from src.metrics import Metrics

# 1. Setup Configuration
config = SimulationConfig(
    starting_age=25,
    retirement_age=65,
    end_age=100,
    initial_salary=50000,
    savings_rate=0.10,
    withdrawal_rate=0.04
)

# 2. Define Strategies
equity_100 = FixedAllocationStrategy({"Domestic": 0.5, "International": 0.5})
glide_path = GlidePathStrategy(start_age=25, retire_age=65)

# 3. Initialize Historical Market Engine
market = BootstrapMarket("data/historical_returns.csv", block_size=10)
sim = Simulator(config)

# 4. Run Simulations
equity_results = sim.run_stochastic(equity_100, num_trials=1000, market_engine=market)
glide_results = sim.run_stochastic(glide_path, num_trials=1000, market_engine=market)

# 5. Compare Metrics
print(f"100% Equity - Median Wealth: ${Metrics.percentile(equity_results, 50):,.2f}")
print(f"100% Equity - 5th Percentile: ${Metrics.percentile(equity_results, 5):,.2f}")
print(f"100% Equity - Prob. of Ruin: {Metrics.probability_of_ruin(equity_results):.2%}")

print(f"\nGlide Path - Median Wealth: ${Metrics.percentile(glide_results, 50):,.2f}")
print(f"Glide Path - 5th Percentile: ${Metrics.percentile(glide_results, 5):,.2f}")
print(f"Glide Path - Prob. of Ruin: {Metrics.probability_of_ruin(glide_results):.2%}")
```

## Plugging In Real-World Data

The `BootstrapMarket` engine reads from a standard CSV format. To use your own data:

1. Create a CSV file in the `data/` directory.
2. Ensure the CSV has a **Year** column and one column for each asset class (e.g., `Domestic`, `International`, `Bonds`).
3. Returns should be in decimal format (e.g., `0.07` for 7%, `-0.05` for -5%).

**Example Format (`data/my_data.csv`):**
```csv
Year,Domestic,International,Bonds
1990,0.05,-0.02,0.08
1991,0.30,0.12,0.15
...
```

4. Update your simulation script to point to the new file:
```python
market = BootstrapMarket("data/my_data.csv", block_size=10)
```

## Project Structure

- `src/`: Core logic (Investor, Simulator, Market, Strategy, Metrics).
- `data/`: Historical return datasets.
- `tests/`: Full suite of unit and integration tests.

## Future Development

### Planned Enhancements
- **Flexible Rebalancing**: Allow for configurable rebalancing frequencies (e.g., quarterly, every 5 years) or threshold-based rebalancing (e.g., only when an asset drifts by >5%).
- **Asset Class Granularity**: Expand support for more specific international markets (Emerging vs. Developed), Small-Cap vs. Large-Cap, and alternative assets like Real Estate (REITs).
- **Visualization Suite**: Integrate `matplotlib` or `plotly` to generate:
    - Terminal wealth distribution histograms.
    - "Spaghetti plots" showing individual simulation paths over time.
    - Probability of ruin heatmaps across different withdrawal rates.
- **Taxes & Fees**: Model the impact of capital gains taxes, dividend leakage, and fund expense ratios on long-term wealth.
- **Dynamic Withdrawal Strategies**: Implement smarter decumulation rules, such as the **Guyton-Klinger Guardrails**, instead of a fixed percentage.
- **Correlated Synthetic Markets**: Improve the `SyntheticMarket` to handle asset correlations (covariance), allowing for more realistic stock/bond interactions.
- **Social Security & Pensions**: Add the ability to model non-portfolio income streams starting at specific ages.
- **Sensitivity Analysis Automation**: A runner that "sweeps" through parameters (e.g., testing every savings rate from 5% to 20%) to find optimal inflection points.
- **Web Dashboard**: Create a Streamlit or Dash interface to allow non-technical users to interact with the simulation parameters and view results in real-time.
