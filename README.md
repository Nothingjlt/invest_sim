# Lifecycle Investment Simulation Tool

A Python-based simulation engine inspired by the research paper **"Beyond the Status Quo: A Critical Assessment of Lifecycle Investment Advice"** (Anarkulova, Cederburg, and O'Doherty, 2023). 

This tool allows investors to compare traditional "Glide Path" (Target Date Fund) strategies against the paper's recommendation of **100% Equities with Geographic Diversification** using both synthetic and historical market data.

## Features

- **Stochastic Lifecycle Modeling**: Monte Carlo simulations tracking an investor from age 25 to a variable terminal age (Longevity Risk).
- **Interchangeable & Research-Based Strategies**: 
    - `FixedAllocationStrategy`: Constant asset mix.
    - `GlidePathStrategy`: Traditional Target Date Fund (TDF) approach.
    - `PaperOptimalStrategy`: The 100% Equity recommendation with tactical cash buffers from Anarkulova et al. (2023).
    - `PaperTDFStrategy`: Representative industry glide path for direct research comparison.
- **Advanced Market Engines**:
    - `SyntheticMarket`: Normal distribution modeling (Mean/Volatility).
    - `BootstrapMarket`: **Block Bootstrap** sampling from real-world historical data to preserve market cycles.
- **Longevity & Social Security**:
    - **Mortality Engine**: Simplified Gompertz mortality model for stochastic lifespans.
    - **Social Security**: Integrated as a consumption floor to model non-portfolio income.
- **Dynamic Withdrawal Rules**: Supports both "Variable Percentage" and "Fixed Real" (4% Rule) withdrawal strategies.
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

The following script compares the paper's **100% Equity (Optimal)** strategy against a **Traditional Target Date Fund (TDF)** using synthetic market parameters derived from historical developed market data.

```python
from src.config import SimulationConfig
from src.simulator import Simulator
from src.strategy import PaperOptimalStrategy, PaperTDFStrategy
from src.metrics import Metrics

# 1. Setup Configuration with Research Parameters
paper_markets = SimulationConfig.get_paper_market_configs()
config = SimulationConfig(
    starting_age=25,
    retirement_age=65,
    initial_salary=50000,
    savings_rate=0.10,
    withdrawal_rate=0.04,
    withdrawal_strategy="fixed_real", # The "4% Rule" behavior
    enable_mortality=True,            # Stochastic lifespans
    markets=paper_markets
)

# 2. Define Research-Based Strategies
optimal_strategy = PaperOptimalStrategy(retire_age=65)
tdf_strategy = PaperTDFStrategy(start_age=25, retire_age=65)

# 3. Initialize Simulator
sim = Simulator(config)

# 4. Run Monte Carlo Simulations
optimal_results = sim.run_stochastic(optimal_strategy, num_trials=1000)
tdf_results = sim.run_stochastic(tdf_strategy, num_trials=1000)

# 5. Compare Metrics
print(f"--- 100% Equity (Optimal) ---")
print(f"Median Wealth: ${Metrics.percentile(optimal_results, 50):,.2f}")
print(f"5th Percentile: ${Metrics.percentile(optimal_results, 5):,.2f}")
print(f"Prob. of Ruin: {Metrics.probability_of_ruin(optimal_results):.2%}")

print(f"\n--- Traditional TDF ---")
print(f"Median Wealth: ${Metrics.percentile(tdf_results, 50):,.2f}")
print(f"5th Percentile: ${Metrics.percentile(tdf_results, 5):,.2f}")
print(f"Prob. of Ruin: {Metrics.probability_of_ruin(tdf_results):.2%}")
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
from src.market import BootstrapMarket
market = BootstrapMarket("data/my_data.csv", block_size=10)
results = sim.run_stochastic(optimal_strategy, market_engine=market)
```

## Project Structure

- `src/`: Core logic (Investor, Simulator, Market, Strategy, Metrics).
- `data/`: Historical return datasets.
- `tests/`: Full suite of unit and integration tests.

## Future Development

### Planned Enhancements
- **Flexible Rebalancing**: Allow for configurable rebalancing frequencies (e.g., quarterly, every 5 years) or threshold-based rebalancing.
- **Asset Class Granularity**: Expand support for Emerging vs. Developed markets, Small-Cap vs. Large-Cap, and REITs.
- **Visualization Suite**: Integrate `matplotlib` or `plotly` to generate terminal wealth histograms and "spaghetti plots" of simulation paths.
- **Taxes & Fees**: Model the impact of capital gains taxes, dividend leakage, and expense ratios.
- **Sensitivity Analysis Automation**: A runner that "sweeps" through parameters (e.g., varying savings rates) to find optimal inflection points.
- **Web Dashboard**: Create a Streamlit or Dash interface for interactive simulations.

## Acknowledgements

This project was developed with the assistance of **Gemini CLI**, an AI-powered software engineering agent.
