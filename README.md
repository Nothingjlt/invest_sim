# Lifecycle Investment Simulation Tool

A Python-based simulation engine inspired by the research paper **"Beyond the Status Quo: A Critical Assessment of Lifecycle Investment Advice"** (Anarkulova, Cederburg, and O'Doherty, 2023). 

This tool allows investors to compare traditional "Glide Path" (Target Date Fund) strategies against the paper's recommendation of **100% Equities with Geographic Diversification** using both synthetic and historical market data.

## Features

- **Stochastic Lifecycle Modeling**: Monte Carlo simulations tracking an investor from age 25 to a variable terminal age (Longevity Risk).
- **World Market Architecture**: Support for 60+ country indices (Developed and Emerging) to simulate global diversification benefits.
- **Interchangeable & Research-Based Strategies**: 
    - `FixedAllocationStrategy`: Constant asset mix.
    - `WorldEquityStrategy`: Hierarchical global allocation across regions and specific countries.
    - `GlidePathStrategy`: Traditional Target Date Fund (TDF) approach with granular asset support.
    - `PaperOptimalStrategy`: The 100% Equity recommendation with tactical cash buffers from Anarkulova et al. (2023), now country-aware.
    - `PaperTDFStrategy`: Representative industry glide path for direct research comparison.
    - `BalancedStrategy`: Traditional 60/40 stock-bond benchmark.
- **Advanced Market Engines**:
    - `SyntheticMarket`: Normal distribution modeling (Mean/Volatility).
    - `BootstrapMarket`: **Dynamic Block Bootstrap** sampling that detects country columns automatically from historical CSV data.
    - `StationaryBootstrapMarket`: Geometrically distributed block sizes following Politis & Romano (1994), matching the paper's bootstrap design.
    - `PerspectiveBootstrapMarket`: Country-aware bootstrap engine that processes raw historical panel data (like JST) into perspective-adjusted real returns (FX-converted, deflated by local inflation, and GDP-weighted for international markets).
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
   *(Note: Requirements include `pandas`, `numpy`, `pytest`, and `matplotlib`)*

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
    social_security_benefit=15000,    # Annual real SS income
    enable_mortality=True,            # Stochastic lifespans
    markets=paper_markets
)

# 2. Define Research-Based Strategies (matching labels in paper_markets)
optimal_strategy = PaperOptimalStrategy(
    retire_age=65, 
    dom_label="Domestic Stock",
    intl_assets={"International Stock": 1.0}
)
tdf_strategy = PaperTDFStrategy(
    start_age=25, 
    retire_age=65, 
    dom_label="Domestic Stock",
    intl_assets={"International Stock": 1.0}
)

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

## Visualization

The project now includes a lightweight `matplotlib`-based plotting module for visualizing simulation outcomes.

- `plot_terminal_wealth_histogram(...)`
  - X axis: terminal wealth (`$`)
  - Y axis: trial count
- `plot_percentile_curve(...)`
  - X axis: percentile (`0–100`)
  - Y axis: terminal wealth (`$`)
- `plot_ruin_probability(...)`
  - X axis: strategy label
  - Y axis: ruin probability (`0–100%`)
- `plot_summary_metrics(...)`
  - X axis: metric name (`mean`, `median`, `5th percentile`, `95th percentile`)
  - Y axis: terminal wealth (`$`)
- `plot_simulation_paths(...)`
  - X axis: age
  - Y axis: portfolio value (`$`)
- `plot_withdrawal_paths(...)`
  - X axis: age
  - Y axis: actual withdrawal amount from portfolio (`$`)
- `plot_withdrawal_comparison(...)`
  - X axis: age
  - Y axis: withdrawal amount (`$`) at a specific percentile across multiple strategies

Example:

```python
from src.visualization import (
    plot_terminal_wealth_histogram,
    plot_percentile_curve,
    plot_ruin_probability,
    plot_summary_metrics,
    plot_simulation_paths,
    plot_withdrawal_paths,
    plot_withdrawal_comparison,
)

# after running sim.run_stochastic(..., track_paths=True) for one or more strategies
plot_terminal_wealth_histogram(optimal_results.terminal_wealths, strategy_name="Optimal Strategy", save_path="optimal_terminal_wealth.png")
plot_percentile_curve(tdf_results.terminal_wealths, strategy_name="TDF Strategy", save_path="tdf_percentile_curve.png")
plot_ruin_probability(
    {
        "Optimal Strategy": optimal_results.terminal_wealths,
        "TDF Strategy": tdf_results.terminal_wealths,
    },
    save_path="comparison_ruin_probability.png",
)
plot_summary_metrics(
    {
        "Optimal Strategy": optimal_results.terminal_wealths,
        "TDF Strategy": tdf_results.terminal_wealths,
    },
    save_path="comparison_summary_metrics.png",
)
plot_simulation_paths(
    optimal_results.paths,
    strategy_name="Optimal Strategy",
    save_path="optimal_simulation_paths.png",
)
plot_withdrawal_paths(
    optimal_results.withdrawal_paths,
    strategy_name="Optimal Strategy",
    save_path="optimal_withdrawal_paths.png",
)
plot_withdrawal_comparison(
    {
        "Optimal Strategy": optimal_results.withdrawal_paths,
        "TDF Strategy": tdf_results.withdrawal_paths,
    },
    percentile=10.0,
    save_path="comparison_withdrawal_percentiles.png",
)
```

## Plugging In Real-World Data

The `BootstrapMarket` engine reads from a standard CSV format and dynamically detects all available country or asset indices. You can provide as many or as few countries as you like.

1. Create a CSV file in the `data/` directory.
2. Ensure the CSV has a **Year** column and one column for each country or asset (e.g., `USA`, `GBR`, `CHN`, `Bonds`, `Bills`).
3. Returns should be in decimal format (e.g., `0.07` for 7%, `-0.05` for -5%).

**Example Format (`data/global_historical_returns.csv`):**
```csv
Year,USA,GBR,JPN,CHN,BRA,Bonds,Bills
1990,0.05,-0.02,0.12,0.08,0.15,0.08,0.03
1991,0.30,0.12,0.05,0.22,0.45,0.15,0.04
...
```

4. Update your simulation script to point to the new file:
```python
from src.market import BootstrapMarket
market = BootstrapMarket("data/global_historical_returns.csv", block_size=10)
results = sim.run_stochastic(optimal_strategy, market_engine=market)
```

*(Note: When using `BootstrapMarket`, ensure your strategy's `dom_label` and `intl_assets` match the column names in your CSV, such as `"USA"`, `"GBR"`, etc.)*

## Real-World JST Macrohistory Database Integration

You can now run simulations backed by actual historical macroeconomic and market data spanning from **1870 to 2023** across 18 developed nations via the **Jordà-Schularick-Taylor (JST) Macrohistory Database**.

### Downloading the JST Data
Run the automated fetcher script to retrieve the JST dataset:
```bash
python fetch_jst_data.py
```
This script downloads the latest release (R6) from macrohistory.net and saves it to `data/raw/jst_dataset.csv`. (If the automated download fails due to network constraints, follow the instructions in the script's output to download and save the XLSX manually, then re-run the script).

### Using the Paper-Faithful Market Engine
Use `PerspectiveBootstrapMarket` to dynamically convert the raw long panel data into domestic-investor perspective returns:
```python
from src.market import PerspectiveBootstrapMarket

# Configure USA-perspective, stationary geometric block bootstrap (10-year mean block)
market_engine = PerspectiveBootstrapMarket(
    csv_path="data/raw/jst_dataset.csv",
    perspective_country="USA",
    block_size=10,
    stationary_bootstrap=True,
    weight_method="gdp"
)
```
This dynamically calculates:
- **Domestic Stock**: Real equity returns in domestic currency.
- **International Stock**: GDP-in-USD weighted average of non-domestic equity returns, FX-converted to the domestic currency, and deflated by domestic inflation.
- **Bonds**: Real 10-year government bond returns.
- **Bills**: Real short-term treasury bill returns.


## Project Structure

- `src/`: Core logic (Investor, Simulator, Market, Strategy, Metrics).
- `data/`: Global return datasets and historical indices.
- `tests/`: Full suite of unit and integration tests.
- `generate_global_data.py`: Utility to generate multi-country synthetic historical paths.

## Future Development

### Planned Enhancements
- **Flexible Rebalancing**: Allow for configurable rebalancing frequencies (e.g., quarterly, every 5 years) or threshold-based rebalancing.
- **Asset Class Granularity**: Expand support for Emerging vs. Developed markets, Small-Cap vs. Large-Cap, and REITs.
- **Visualization Enhancements**: The basic visualization suite and path plot support are implemented, but nicer, more premium plots (e.g., customized CSS/Matplotlib styles, advanced visual configurations, or interactive Dash/Streamlit figures) are still required.
- **Taxes & Fees**: Model the impact of capital gains taxes, dividend leakage, and expense ratios.
- **Sensitivity Analysis Automation**: A runner that "sweeps" through parameters (e.g., varying savings rates) to find optimal inflection points.
- **Web Dashboard**: Create a Streamlit or Dash interface for interactive simulations.
- **Withdrawl amount updates** *(done)*: Withdrawal can now be inflation-adjusted (using realized CPI from market data), capped at a configurable maximum, or bound from below by a minimum real expenditure floor. Both cap and floor can themselves be inflation-adjusted. See `SimulationConfig` fields: `withdrawal_inflation_adjusted`, `withdrawal_cap`, `withdrawal_cap_inflation_adjusted`, `withdrawal_floor`, `withdrawal_floor_inflation_adjusted`.
- **Variable withdrawal timing granularity**: The paper specifies that variable-pct withdrawals are evaluated at the *beginning of each month*. The current simulator runs in annual steps, which approximates this. A future increment should introduce monthly sub-steps in the decumulation loop to exactly replicate the paper's timing semantics.

## Acknowledgements

This project was developed with the assistance of **Gemini CLI**, an AI-powered software engineering agent.
