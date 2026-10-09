# Lifecycle Investment Simulation Tool

A Python-based simulation engine inspired by **"Beyond the Status Quo: A Critical Assessment of Lifecycle Investment Advice"** by Anarkulova, Cederburg, and O'Doherty (reference version dated August 27, 2026).

This tool allows investors to compare traditional "Glide Path" (Target Date Fund) strategies against the paper's all-equity strategies using both synthetic and historical market data.

## Features

- **Stochastic Lifecycle Modeling**: Monte Carlo simulations tracking an investor from age 25 to a variable terminal age (Longevity Risk).
- **World Market Architecture**: Support for 60+ country indices (Developed and Emerging) to simulate global diversification benefits.
- **Interchangeable & Research-Based Strategies**: 
    - `FixedAllocationStrategy`: Constant asset mix.
    - `WorldEquityStrategy`: Hierarchical global allocation across regions and specific countries.
    - `GlidePathStrategy`: Traditional Target Date Fund (TDF) approach with granular asset support.
    - `PaperOptimalStrategy`: A country-aware fixed-weight approximation of the paper's 34% domestic / 66% international equity strategy. It does not implement the separate age-based optimizer's 13 allocation windows.
    - `PaperTDFStrategy`: Linear approximation between the reported TDF allocation extremes; it does not reconstruct the age-specific curve in Figure 1.
    - `BalancedStrategy`: Traditional 60/40 stock-bond benchmark.
- **Advanced Market Engines**:
    - `SyntheticMarket`: Normal distribution modeling (Mean/Volatility).
    - `BootstrapMarket`: **Dynamic Block Bootstrap** sampling that detects country columns automatically from CSV data, with explicit data-source provenance.
    - `StationaryBootstrapMarket`: Geometrically distributed block sizes following Politis & Romano (1994), inspired by—but not a full reproduction of—the paper's bootstrap design; see [GAPS.md](GAPS.md).
    - `PerspectiveBootstrapMarket`: Country-aware bootstrap engine that processes raw historical panel data (like JST) into perspective-adjusted real returns (FX-converted, deflated by local inflation, and GDP-weighted as a proxy for international markets); see [GAPS.md](GAPS.md) for known paper-fidelity gaps.
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
   pytest tests/
   ```
   *(A root `conftest.py` automatically adds the project root to `sys.path`.)*


## Market descriptors and strategy allocations

`MarketConfig(name, expected_return, volatility)` describes an available asset
and its synthetic return assumptions. `SimulationConfig.markets` defines the
investment universe. The default configuration, `get_paper_market_configs()`,
and `get_world_market_configs()` supply descriptors with `weight=None`, so their
output passes configuration validation without assigning a portfolio policy.

Every strategy owns its target weights through `get_allocation(age)`, including
`WorldEquityStrategy`, `PaperOptimalStrategy`, `PaperTDFStrategy`,
`BalancedStrategy`, `FixedAllocationStrategy`, `GlidePathStrategy`, and custom
`Strategy` subclasses. The simulator uses the strategy's resolved allocation
for contributions, rebalancing, and withdrawals. Paper strategies keep their
logical Domestic/International Stock sleeves when those return series exist;
with country-only series, their default resolver maps domestic stock to USA
and international stock to GBR/JPN/FRA/DEU using its existing constituent mix.
Explicit strategy label mappings retain their existing behavior.

```python
from src.config import SimulationConfig
from src.strategy import WorldEquityStrategy

config = SimulationConfig(markets=SimulationConfig.get_world_market_configs())
config.validate()  # No placeholder weights or default investment policy needed.
strategy = WorldEquityStrategy({"Developed": {"USA": 0.5, "GBR": 0.3, "JPN": 0.2}})
```

For compatibility, `MarketConfig(..., weight=...)` and the fourth positional
argument remain supported as legacy allocation metadata. If any market has a
numeric weight, every market must have one, and the weights must be finite,
nonnegative, and sum to one (within the existing rounding tolerance). All-zero
weights and mixtures of omitted/numeric weights are rejected. Legacy weights
never override a strategy. To migrate descriptor-only callers, omit all weights
or set all of them to `None`; move portfolio allocations into a strategy, such as
`FixedAllocationStrategy`. Code extracting numeric weights from the paper/world
factories must now obtain them from the chosen strategy. The paper factory no
longer embeds `PaperOptimalStrategy`'s 34/66 policy.

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
print(f"--- 100% Equity (Optimal) | {optimal_results.provenance.label} ---")
print(f"Median Wealth: ${Metrics.percentile(optimal_results, 50):,.2f}")
print(f"5th Percentile: ${Metrics.percentile(optimal_results, 5):,.2f}")
print(f"Prob. of Ruin: {Metrics.probability_of_ruin(optimal_results):.2%}")

print(f"\n--- Traditional TDF | {tdf_results.provenance.label} ---")
print(f"Median Wealth: ${Metrics.percentile(tdf_results, 50):,.2f}")
print(f"5th Percentile: ${Metrics.percentile(tdf_results, 5):,.2f}")
print(f"Prob. of Ruin: {Metrics.probability_of_ruin(tdf_results):.2%}")
```

## Data Provenance

Every `run_stochastic` result exposes `.provenance`, an immutable `DataProvenance`
record with `kind`, `source`, and an optional input-file `sha256` digest. Its
`.label` is suitable for report headings and plot titles. The label describes the
**input data**; even a simulation using historical observations produces simulated
outcomes, not an observed investment track record.

- **`synthetic`**: model-generated data. This includes `SyntheticMarket` and the
  bundled `data/global_historical_returns.csv`, which is generated by
  `generate_global_data.py` despite its filename. The bundled panel is recognized
  by its exact SHA-256 digest, including when copied or renamed. It cannot be
  declared historical.
- **`historical`**: an explicit caller declaration identifying the actual source
  and release. This label does not independently certify the data's authenticity
  or establish replication of the paper.
- **`unknown`**: origin has not been declared. Arbitrary CSVs, JST-format files,
  and custom engines default to this label; filenames never imply historical data.

CSV engines accept `provenance=DataProvenance(...)` and attach the input digest.
If you supply an expected `sha256`, a mismatch raises an error. Missing CSV inputs
raise `FileNotFoundError`, including through `PerspectiveBootstrapMarket`; the
market will not create substitute observations. The direct `JSTDataLoader` also
fails on a missing file; it no longer creates a dummy JST panel. Existing files
previously generated by the legacy JST loader have no trustworthy historical
declaration: replace them with sourced data before declaring them historical.

Bootstrap rows may have partial coverage. Blank/`NaN` observations are omitted
from that sampled year's return mapping; if an active holding or target has no
observed return, validation raises before growth or contributions. Missing data
is never interpreted as a 0% return. Non-numeric or infinite return values are
rejected as invalid input.

The default result remains list-compatible: `isinstance(results, list)` is true,
and existing metrics, indexing, and iteration work. With `track_paths=True`, the
result retains its three fields (`terminal_wealths`, `paths`, `withdrawal_paths`)
and supports existing tuple unpacking. Both the result and its `terminal_wealths`
expose the same provenance. Custom market engines can declare a `provenance`
attribute using the same record.

Use `results.to_dict()` to save outcomes **with** their source metadata:

```python
import json

print(optimal_results.provenance.label)
serialized_result = json.dumps(optimal_results.to_dict())
```

Ordinary list conversion, slicing, or `json.dumps(results)` follows standard list
behavior and omits metadata; `to_dict()` is the explicit export API for both
result forms. Plotting functions accept raw values, so include `.provenance.label`
in the supplied strategy label when presenting results.

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

optimal_results = sim.run_stochastic(optimal_strategy, num_trials=1000, track_paths=True)
tdf_results = sim.run_stochastic(tdf_strategy, num_trials=1000, track_paths=True)
optimal_label = f"Optimal Strategy | {optimal_results.provenance.label}"
tdf_label = f"TDF Strategy | {tdf_results.provenance.label}"
plot_terminal_wealth_histogram(optimal_results.terminal_wealths, strategy_name=optimal_label, save_path="optimal_terminal_wealth.png")
plot_percentile_curve(tdf_results.terminal_wealths, strategy_name=tdf_label, save_path="tdf_percentile_curve.png")
plot_ruin_probability(
    {
        optimal_label: optimal_results.terminal_wealths,
        tdf_label: tdf_results.terminal_wealths,
    },
    save_path="comparison_ruin_probability.png",
)
plot_summary_metrics(
    {
        optimal_label: optimal_results.terminal_wealths,
        tdf_label: tdf_results.terminal_wealths,
    },
    save_path="comparison_summary_metrics.png",
)
plot_simulation_paths(
    optimal_results.paths,
    strategy_name=optimal_label,
    save_path="optimal_simulation_paths.png",
)
plot_withdrawal_paths(
    optimal_results.withdrawal_paths,
    strategy_name=optimal_label,
    save_path="optimal_withdrawal_paths.png",
)
plot_withdrawal_comparison(
    {
        optimal_label: optimal_results.withdrawal_paths,
        tdf_label: tdf_results.withdrawal_paths,
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

**Illustrative Format (`data/my_returns.csv`; numbers below are examples):**
```csv
Year,USA,GBR,JPN,CHN,BRA,Bonds,Bills
1990,0.05,-0.02,0.12,0.08,0.15,0.08,0.03
1991,0.30,0.12,0.05,0.22,0.45,0.15,0.04
...
```

4. Declare the source when constructing the market. For the bundled generated
   panel (and for any newly generated panel), use synthetic provenance:
```python
from src.market import BootstrapMarket
from src.provenance import DataProvenance

market = BootstrapMarket(
    "data/global_historical_returns.csv", block_size=10,
    provenance=DataProvenance("synthetic", "generate_global_data.py"),
)
results = sim.run_stochastic(optimal_strategy, market_engine=market)
print(results.provenance.label)
```

For actual historical observations, supply `DataProvenance("historical",
"Your data provider, release, and construction method")` only after establishing
their origin. Omit the declaration to keep unfamiliar data labeled unverified.

*(Note: When using `BootstrapMarket`, ensure your strategy's `dom_label` and `intl_assets` match the column names in your CSV, such as `"USA"`, `"GBR"`, etc.)*

The legacy `BootstrapMarket`, `StationaryBootstrapMarket`, and
`PerspectiveBootstrapMarket` constructors require a positive integer
`block_size`, a non-empty input panel (and, for the perspective engine, a
non-empty processed panel), and numeric finite return observations no lower
than `-1`. Blank/NaN cells remain unavailable observations and are omitted so
country-coverage validation can distinguish missing data from malformed data.
The simulator applies the same return validation to custom market engines
before either annual or monthly processing. A missing `Inflation` field or an
explicit `Inflation=None` uses zero inflation; nonnumeric, non-finite, or
below-`-1` inflation metadata is rejected before it reaches arithmetic.

## Real-World JST Macrohistory Database Integration

The project can process the **Jordà-Schularick-Taylor (JST) Macrohistory Database**. Release R6 provides annual data for 18 advanced economies beginning in 1870; country and variable coverage vary, and the return-data fetcher describes its coverage through 2020. This is distinct from the paper's 39-country monthly dataset.

### Downloading the JST Data
Run the automated fetcher script to retrieve the JST dataset:
```bash
python fetch_jst_data.py
```
This script downloads the latest release (R6) from macrohistory.net and saves it to `data/raw/jst_dataset.csv`. (If the automated download fails due to network constraints, follow the instructions in the script's output to download and save the XLSX manually, then re-run the script).

### Using the Perspective Market Engine
Use `PerspectiveBootstrapMarket` to dynamically convert the raw long panel data into domestic-investor perspective returns:
```python
from src.market import PerspectiveBootstrapMarket
from src.provenance import DataProvenance

# Configure USA-perspective, stationary geometric block bootstrap (10-year mean block)
# Declare historical provenance only after confirming the downloaded source/release.
market_engine = PerspectiveBootstrapMarket(
    csv_path="data/raw/jst_dataset.csv",
    perspective_country="USA",
    block_size=10,
    stationary_bootstrap=True,
    weight_method="gdp",
    provenance=DataProvenance("historical", "JST Macrohistory Database R6"),
)
```
This dynamically calculates:
- **Domestic Stock**: Real equity returns in domestic currency.
- **International Stock (current legacy method)**: non-domestic equity returns weighted by current-year raw GDP/FX, converted to the domestic currency, and deflated by domestic inflation.
- **Bonds**: Real 10-year government bond returns.
- **Bills**: Real short-term treasury bill returns.

The legacy perspective loader uses the paper's country sample registry. Both
the perspective country and each foreign market must be eligible in the return
year `t` and the preceding year `t-1`. Chile's eligible intervals are
1927--1970 and 2010--2023; observations in 1971--2009 are unavailable to this
path even when present in the CSV. The lag requirement also excludes 1927 and
2010 from Chile's usable return years. Missing in-sample observations remain
missing: they are never filled with zero, and an incomplete four-asset vector
is omitted under the existing legacy coverage rule.

`CountryMetadataRegistry.get_sample_periods(iso)` returns a tuple of inclusive
`(start_year, end_year)` intervals, matching the values now stored in
`COUNTRY_SAMPLE_PERIODS`. Use `is_valid_year_for_country(iso, year)` for
membership. The compatible `get_sample_period(iso)` still returns the outer
bounds (for Chile, `(1927, 2023)`), which must not be used to test eligibility.
Unknown codes return `()` from the plural accessor, `(None, None)` from the
legacy accessor, and false membership. The loader preserves its observed-panel
fallback for an unknown perspective, but excludes unknown foreign markets.
The pooled loader uses observed JST coverage independently of this registry.

The current `weight_method="gdp"` is an unvalidated GDP proxy, not the paper's
lagged USD market-cap weighting: it uses contemporaneous GDP divided by FX,
without country-specific GDP scale normalization, and falls back to equal
weights when GDP is missing. Do not assume those current cross-country weights
are comparable. Use `PooledPerspectiveBootstrapMarket` for the documented
scale-normalized lagged-GDP baseline; see [GAPS.md](GAPS.md) for its coverage
rules and remaining alternatives.

### Broad pooled-country bootstrap

`PooledPerspectiveBootstrapMarket` provides the annual broad-pooling path. It
uses complete, real return vectors from country perspectives, samples country
and date at geometric block starts, and restarts at country-history gaps and
endpoints. The default gives each eligible country equal probability at each
block start. `block_size` is the mean of the untruncated geometric request
(10 annual observations by default). Gaps and endpoints truncate blocks, so
observed blocks can be shorter; a simulated path ending can also shorten the
final block. Equal start probabilities do not imply equal realized observation
exposure: short histories contribute fewer observations per start on average,
and long contiguous segments can dominate observed path years.

```python
from src.market import PooledPerspectiveBootstrapMarket
from src.provenance import DataProvenance

market = PooledPerspectiveBootstrapMarket(
    csv_path="data/raw/jst_dataset.csv",
    block_size=10,
    country_sampling="equal_country",
    weight_method="gdp_lagged",
    gdp_lag=2,
    min_weight_coverage=1.0,
    seed=42,
    provenance=DataProvenance("historical", "JST Macrohistory Database R6"),
)

# After running a simulation, inspect the sampled country/year exposure.
diagnostics = market.get_diagnostics()
last_observation = market.last_observation
market_universe = market.market_universe_report
```

Diagnostics distinguish `intended_country_start_probabilities`,
`realized_country_start_counts`, and `realized_country_observation_counts`.
`observed_block_length_counts` records realized block lengths, including
boundary truncation and path-end censoring.

The baseline weights use nominal GDP converted with JST's `xrusd` field after
applying the R6 country-specific scale multipliers in
[`JST_R6_GDP_SCALE_FACTORS`](src/data_loader.py). This is labeled **GDP
converted using JST FX**: historical FX observations combine source/timing
conventions, so they should not be described as uniform annual-average USD GDP.
GDP is an economic-size proxy, not listed-market capitalization, and the
two-year lag does not make revised historical data point-in-time.

By default, the loader includes foreign markets with at least one usable
equity-return observation. JST R6 has no equity-return observations for Canada
or Ireland, so those markets are omitted from the default foreign basket and
shown in `market.market_universe_report`. Pass `foreign_countries` to choose a
different universe. Within the selected universe, the default requires full
foreign-weight return coverage for each country-year. Lower
`min_weight_coverage` values explicitly permit renormalization over observed
markets; inspect `market.coverage_report` and `market.get_diagnostics()` when
doing so. `weight_method="equal"` and
`country_sampling="observation_weighted"` are implemented comparison settings.
World Bank/WFE and Kuvshinov–Zimmermann capitalization remain documented
alternatives; they are not automatically downloaded or mixed into the long
series. No paid dataset is required for the JST baseline.

The detailed design and open alternatives are in [GAPS.md](GAPS.md). It
distinguishes international-equity weights from country-selection weights;
peer/region-weighted, one-country-per-path, fixed-country, hybrid, and IID
pooling remain unimplemented. Broad pooling represents a mixture of historical
domestic environments; it does not by itself estimate the future experience of
a particular small country. Annual inputs also do not provide observed monthly
sequence risk. The pooled engine returns real returns without an `Inflation`
series; keep simulator spending and returns in consistent real units. Its
monthly decumulation mode smooths annual returns into equal monthly compound
returns and does not recover observed monthly sequence risk.

JST is available at no cost under [CC BY-NC-SA 4.0](https://www.macrohistory.net/database/licence-terms/).
Check its non-commercial and share-alike terms before redistributing the source
or derived data. The World Bank/WFE listed-market-cap series is available from
1975 onward, with country-specific gaps, under [CC BY 4.0](https://data.worldbank.org/indicator/CM.MKT.LCAP.CD); it supports a shorter capitalization-weighted comparison, not the full JST history.


## Project Structure

- `src/`: Core logic (Investor, Simulator, Market, Strategy, Metrics).
- `data/`: The bundled synthetic return panel and any user-supplied datasets.
- `tests/`: Full suite of unit and integration tests.
- `generate_global_data.py`: Utility to generate multi-country synthetic return paths.

## Future Development

### Planned Enhancements
- **Flexible Rebalancing**: Allow for configurable rebalancing frequencies (e.g., quarterly, every 5 years) or threshold-based rebalancing.
- **Asset Class Granularity**: Expand support for Emerging vs. Developed markets, Small-Cap vs. Large-Cap, and REITs.
- **Visualization Enhancements**: The basic visualization suite and path plot support are implemented, but nicer, more premium plots (e.g., customized CSS/Matplotlib styles, advanced visual configurations, or interactive Dash/Streamlit figures) are still required.
- **Taxes & Fees**: Model the impact of capital gains taxes, dividend leakage, and expense ratios.
- **Sensitivity Analysis Automation**: A runner that "sweeps" through parameters (e.g., varying savings rates) to find optimal inflection points.
- **Web Dashboard**: Create a Streamlit or Dash interface for interactive simulations.
- **Withdrawl amount updates** *(done)*: Withdrawal can now be inflation-adjusted (using realized CPI from market data), capped at a configurable maximum, or bound from below by a minimum real expenditure floor. Both cap and floor can themselves be inflation-adjusted. See `SimulationConfig` fields: `withdrawal_inflation_adjusted`, `withdrawal_cap`, `withdrawal_cap_inflation_adjusted`, `withdrawal_floor`, `withdrawal_floor_inflation_adjusted`.
- **Variable withdrawal timing granularity** *(done)*: The paper specifies that variable-pct withdrawals are evaluated at the *beginning of each month*. A new `decumulation_granularity` field on `SimulationConfig` (default `"annual"`) controls this. Set it to `"monthly"` to enable 12 monthly sub-steps per retirement year where each annual return is converted to a monthly compounded rate `r_m = (1 + R)^(1/12) − 1` and the withdrawal is evaluated at the beginning of each month before growth is applied. All cap, floor, and Social Security offsets are proportionally divided by 12 within each monthly sub-step; inflation adjustments occur once at year-end.

## Acknowledgements

This project was developed with the assistance of **Gemini CLI**, an AI-powered software engineering agent.
