# Open Gaps

## Paper-compatible handling of incomplete return histories

**Status:** Annual broad-pooling baseline implemented; exact monthly paper replication remains open.

The new `PooledPerspectiveBootstrapMarket` builds complete annual JST
country-perspective vectors and samples contiguous blocks within a country.
It is intentionally a public-data annual analogue rather than a reproduction
of the paper's monthly 39-country sampler.

### Evidence

The bundled [`data/global_historical_returns.csv`](data/global_historical_returns.csv)
is a synthetic, annual, wide-format panel. It contains 100 rows for 1924--2023,
with 63 country columns and `Bonds`/`Bills`. At the current revision:

- all 100 rows have at least one missing return;
- 3,841 of 6,500 return cells are missing;
- 23 country columns are entirely empty;
- for the default paper-style basket (`USA`, `GBR`, `JPN`, `FRA`, and `DEU`,
  plus fixed income), six rows are incomplete: 1924--1929, because Japan's
  defined sample begins in 1930.

These blanks are primarily created intentionally by
[`generate_global_data.py`](generate_global_data.py), which masks countries
outside their configured sample periods. The file is explicitly labeled
synthetic in the README; it is not the paper's historical JST panel.

### Paper approach

The paper uses a long country-month panel. Each retained country-month has a
complete vector of domestic stocks, international stocks, bonds, and bills.
Countries have different sample periods, so unavailable country-months are not
represented as partially populated observations. The paper reports 91% coverage
of potential developed-country-months and 31,801 usable country-month
observations.

Its stationary block bootstrap samples consecutive return vectors from one
country. If that country's history ends before the requested block is filled,
the remaining block is continued from the beginning of another randomly chosen
country's sample. The four asset returns remain jointly aligned within each
country-month observation.

### Remaining paper-replication gap

The direct [`BootstrapMarket`](src/market.py) and
`StationaryBootstrapMarket` and `PerspectiveBootstrapMarket` retain their
legacy global-row/fixed-perspective behavior. The new pooled engine does not
implement the paper's monthly frequency, 39-country input, or its unfinished
block continuation from the start of another country's sample.

The annual engine fails closed on incomplete asset vectors and does not treat
missing returns as zero. Its boundary restart rule differs from the paper.

The implemented direction is a **broad pooled-country model** for obtaining
more historical environments than a short local series can provide. It is an
analogue model: results describe outcomes across a declared mixture of
countries, not calibrated probabilities for any one small country. See
[Broad pooled-country model](#broad-pooled-country-model-implemented) below.

### Remaining paper-replication work

1. Add a monthly public or licensed input panel if monthly sequence risk and
   paper-level frequency are required.
2. Decide whether a future paper-replication mode should continue unfinished
   blocks from another country's first observation or retain the annual
   engine's restart-at-gap/end rule.
3. Keep the paper-specific sample registry and developed-country eligibility
   separate from the source-coverage eligibility used by the JST pool.

### Validation criteria still open

- Missing values never become implicit zero returns.
- Every sampled pooled observation contains a complete four-asset return vector.
- A sampled block stays within one country's contiguous observed segment; gaps
  and endpoints trigger a fresh country/date draw in the annual engine.
- Coverage counts and excluded country-periods are reproducible from the input
  data and visible to callers.
- Seeded reference paths demonstrate the intended behavior for both synthetic
  fixtures and a declared historical dataset.

## Related implementation and paper-fidelity gaps

### International-stock weighting

**Status:** Lagged-GDP public baseline implemented in pooled engine; legacy
single-perspective weights remain unvalidated; capitalization alternatives are
not integrated.

The paper's Appendix A.3 value-weights international stock returns using each
country's lagged total market capitalization in USD. The legacy
[`PerspectiveBootstrapMarket`](src/market.py) path uses same-year GDP divided
by FX without country scale normalization and falls back to unit weights when
GDP is missing; those weights remain unvalidated. The new pooled path instead
uses a two-year-lagged GDP proxy with explicit R6 unit multipliers and strict
coverage by default. GDP remains an economic-size proxy, not the paper's
market-cap method.

#### Public-data baseline and alternatives

The implemented default uses two-year-lagged nominal GDP converted using JST
FX as a long-history proxy for the international-equity basket. For investor
country `i`, foreign market `j`, and return year `t`:

`size[j,t] = gdp[j,t-2] * scale[j] / xrusd[j,t-2]`

`weight[i,j,t] = size[j,t] / sum(size[k,t])` for declared foreign markets
`k != i`. The implementation uses those weights on year-`t` foreign equity
returns, converts each return to investor `i`'s currency, and deflates by `i`'s
CPI. The two-year lag is a conservative modeling assumption, not a true
point-in-time guarantee: JST history is revised and does not supply historical
data vintages.

For the official JST R6 file, `JST_R6_GDP_SCALE_FACTORS` in
[`src/data_loader.py`](src/data_loader.py) converts the stored series to
absolute local-currency units: `1e6` for AUS, BEL, FIN, IRL, NLD, NOR, PRT,
ESP, SWE, and CHE; `1e9` for CAN, DNK, FRA, DEU, ITA, GBR, and USA; and `1e12`
for JPN. These multipliers follow the [JST R6 main-data documentation](https://www.macrohistory.net/app/download/9834516169/JST_documentationR6.pdf)
and were checked against the official R6 Stata download. Custom or future
releases can pass `gdp_scale_factors` or a per-row `gdp_scale` column; unknown
countries fail closed rather than receiving guessed weights.

`xrusd` is local-currency units per USD, but JST historical FX combines source
and timing conventions and should not be described as uniform annual-average
USD GDP. Label this baseline **GDP converted using JST FX**. The two-year lag
does not correct historical revisions or differences in FX timing. The JST
main R6 file does not expose a standalone market-cap series; its returns
supplement documents sources used for some composite-return calculations, not
a complete distributed cap-weight panel. See [JST R6 returns documentation](https://www.macrohistory.net/app/download/9918957869/JST_RORE_Documentation_R6.pdf).
No paid dataset is needed for the GDP baseline.

GDP measures economic output, not the value of listed equity. This baseline
must be labeled **GDP-weighted international equities** and must not be
presented as the paper's market-cap-weighted portfolio. Missing GDP or FX
inputs never receive an implicit unit weight. By default, the loader derives
the foreign-market universe from countries with at least one usable equity
return in the input and exposes selected and omitted markets through
`market_universe_report`. In the official JST R6 returns file, Canada and
Ireland have no equity-return observations, so they are omitted from the
default foreign basket; they also cannot supply a domestic perspective vector.
Pass `foreign_countries` to declare another universe explicitly.

Within the declared universe, the default requires every foreign market's
lagged GDP/FX weight inputs and a usable year-`t` equity return. A missing
market-year therefore excludes that investor-country-year. A lower
`min_weight_coverage` explicitly permits renormalizing over observed return
markets; `WeightCoverage`, market counts, exclusions, and country-year
coverage remain available in diagnostics. For the full 17-country foreign
basket, strict 100% coverage produces no vectors because Canada's and
Ireland's equity-return series are empty. At a 90% threshold, the official R6
file yields 1,434 vectors across 16 investor perspectives from 1924 through
2020, with gaps. The accepted vectors have no Canadian or Irish equity returns;
their weight remains counted as missing coverage, and the other observed
markets are renormalized. This differs from the default, which excludes
markets with no return series before computing the declared basket weights.
These are availability results, not evidence that a 90% threshold is
statistically optimal.

Public alternatives have different interpretations:

| International-equity weighting | Use and tradeoff |
| --- | --- |
| Equal weight by eligible foreign market | Simple, transparent sensitivity case; gives small markets much more weight relative to their size. |
| **Two-year-lagged GDP converted using JST FX (implemented baseline)** | Long-history public economic-size proxy. Scale-normalized for R6, but not listed-market capitalization or point-in-time data. |
| Listed-market capitalization (World Bank/WFE) | More directly measures listed equity and is available in current USD under CC BY 4.0. WDI indicator `CM.MKT.LCAP.CD` is annual, displayed from 1975, with country-specific gaps. Use the last value demonstrably available before year `t`; if release timing is unknown, use a conservative additional lag. Check market definition/source transitions. [World Bank indicator](https://data.worldbank.org/indicator/CM.MKT.LCAP.CD); [series metadata](https://databank.worldbank.org/metadataglossary/World-Development-Indicators/series/CM.MKT.LCAP.CD). |
| Long-run capitalization reconstruction (Kuvshinov–Zimmermann) | Author page provides Excel/Stata downloads for a study of 17 advanced economies over 1870–2016. More directly relevant to equity weights and much longer than WDI, but reconstructed, not a full match to JST's 18 countries, and reuse-license/redistribution terms are not established here. Audit file units, definitions, coverage, provenance, and license before use. [Author data page](https://dkuvshinov.com/); [paper and appendix](https://www.dkuvshinov.com/wp-content/uploads/2018/09/big_bang_latest.pdf). |
| Paper-matched GFD capitalization | Closest to the paper's exact source and weights, but paid/licensed and dependent on access to the authors' source definitions. Not required for this project baseline. |
| Specified fund or index returns | Closest to a real user's investable product; history, currency, dividends, fees, and licensing are product-specific. |

Do not switch from GDP to market-cap weights partway through the primary
long-history series without labeling the change. Compare capitalization,
GDP, and equal weights on identical country-years in their overlap. JST's R6
returns documentation reports using market-capitalization-to-GDP data from
multiple sources when constructing composite-return series; this is a lead to
investigate, not an assumption that a complete cap series is exposed in the
main JST file. See [JST R6 returns documentation](https://www.macrohistory.net/app/download/9918957869/JST_RORE_Documentation_R6.pdf).
World Bank/WFE data are under [CC BY 4.0](https://data.worldbank.org/indicator/CM.MKT.LCAP.CD).
JST's no-cost data are licensed [CC BY-NC-SA 4.0](https://www.macrohistory.net/database/licence-terms/),
so its non-commercial and share-alike terms must be considered before bundling
data or derived panels. A public download is not by itself proof of an open
redistribution license, especially for the long-run capitalization files.

The pooled input path supports `gdp_lagged` and `equal`; market-cap alternatives
remain documented but require an audited separate input series. The legacy
single-perspective loader still uses its earlier unvalidated same-year GDP
proxy. Do not describe either GDP method as the paper's market-cap weighting.

### Disjoint country sample periods

**Status:** Open

`CountryMetadataRegistry` currently represents each country with one continuous
`(start_year, end_year)` interval. The paper has countries that leave and later
re-enter the developed-country sample; for example, Chile has 1927--1970 and
2010--2023 periods. The current registry therefore treats the intervening years
as eligible and can contaminate coverage calculations and international-return
weights.

Replace the single interval with a list of eligible intervals (or an equivalent
predicate), preserve the distinction between an unavailable period and a true
missing observation, and add tests for Chile and the other reclassified/re-entry
countries.

### Bootstrap input validation

**Status:** Completed

The legacy CSV bootstrap constructors now fail fast on invalid or unusable
inputs:

- `block_size` must be a positive integer for fixed, stationary, and legacy
  perspective bootstraps;
- raw and processed perspective panels must contain at least one row; and
- non-missing return cells must be numeric and finite, with simple returns
  below `-1` rejected before sampling.

Explicit missing/NaN cells in legacy CSV panels remain unavailable observations:
they are omitted from the sampled mapping so existing coverage validation can
report a missing held asset. This is distinct from a present non-finite value
such as infinity, which is rejected.

### Inconsistent monthly handling of invalid returns

**Status:** Completed

The simulator now validates each custom market observation once, immediately
after it is returned and before either annual or monthly processing. Annual and
monthly paths both reject nonnumeric/non-finite values and simple returns below
`-1`; the monthly path only accepts `-1` as an actual total loss and never
coerces a lower value into one.

Missing `Inflation` and an explicit `Inflation=None` both mean zero inflation,
preserving the existing fallback. Present nonnumeric/non-finite inflation is
invalid and fails before arithmetic; numeric inflation below `-1` is also
rejected because it would produce a non-positive adjustment factor.

### Pre-existing world-market configuration gap

**Status:** Open; pre-existing on `master`, not introduced by this PR

`SimulationConfig.get_world_market_configs()` returns every market with
`weight=0.0`, while `SimulationConfig.validate()` requires the market weights to
form a valid allocation. Calling `SimulationConfig(markets=...).validate()` on
the factory output therefore fails before a strategy can supply its actual
portfolio allocation. Existing tests work around this by changing at least one
weight, but there is no test defining the intended factory contract.

Decide whether the factory should return a valid default allocation or whether
market descriptors should be separated from portfolio allocations. Then add a
regression test that exercises the chosen contract; do not silently change this
as part of the paper-bootstrap work.

### Additional tests needed for remaining open gaps

The open gaps above should be covered by:

- multi-row stationary and fixed-block fixtures that exercise country-history
  boundaries and continuation behavior;
- lagged market-cap weighting and disjoint country-eligibility fixtures;
- a factory-contract test for `get_world_market_configs()`; and
- a lightweight documentation/review check that paper-compatibility claims
  continue to link to this file.

### Weak or non-discriminating existing tests

**Status:** Open

Several existing tests can pass without proving the behavior their names or
comments suggest:

- `test_withdrawal_cap_inflation_adjusts` checks that the second withdrawal is
  monotonic and below a bound, but does not assert the exact inflation-adjusted
  cap or withdrawal amount;
- the monthly Social Security test uses a benefit deliberately multiplied by
  12, so it does not cover ordinary annual-benefit coverage or partial
  portfolio funding; and
- `test_diversification_safety_comparison` computes two tail metrics but only
  asserts that both are non-negative, so it does not test the claimed
  diversification comparison.

Strengthen these tests with exact expected values and scenarios that can fail
  when the relevant behavior regresses. For the diversification test, either
  assert a deterministic property using a controlled fixture or rename it as a
  smoke test and add a separate comparative test with a justified invariant.

## Broad pooled-country model (implemented)

**Status:** Annual JST baseline implemented; non-GDP weight integrations and
several alternative pooling estimands remain open.

### Question and interpretation

The goal is to use other countries' histories to broaden the return
environments available to an investor whose own country has a short financial
record. One observation is a complete annual return vector from one country's
investor perspective. A British observation contains British real domestic
stocks, British real international stocks, British real bonds, and British
real bills; the analogous Japanese observation uses Japanese purchasing power
and currency conventions. Returns, inflation metadata, country, year, and
source identity stay together.

This estimates strategy outcomes across a declared mixture of historical
domestic environments. It does not establish that those countries are equally
likely to describe the target country's future. A foreign country's domestic
bond return is an analogue for the target country's bond risk, not an observed
target-country return. Pooling local-real returns also changes the simulated
price-level environment when the sampled country changes. Keep this limitation
visible in labels and reports. A target-country history remains useful as a
local reference even when it is too short to use alone.

JST R6 offers annual data for 18 advanced economies since 1870, with country
and variable coverage varying over time. It is a useful public starting
universe, not the paper's 39-country monthly panel. The repository does not
bundle a raw JST file. Call `get_pooled_coverage_report()` after loading the
user-supplied R6 CSV to inspect country-year eligibility and exclusions. The
JST license is non-commercial/share-alike; [license terms](https://www.macrohistory.net/database/licence-terms/)
apply to redistribution and derived data.

### Implemented baseline

1. `JSTDataLoader.get_pooled_processed_returns()` builds one row per usable
   perspective-country year, with the four real returns plus country, year,
   inflation, weight-year, and basket-coverage metadata. Eligibility follows
   observed source data and requires consecutive calendar years for CPI/FX
   transformations; it does not apply the paper's 39-country registry.
2. The foreign-market universe is independently configurable from the set of
   perspective countries. GDP weighting requires valid lagged GDP, FX, and a
   scale factor for every declared foreign market. Return-basket coverage is
   100% by default; a lower `min_weight_coverage` explicitly permits
   renormalization over markets with valid returns. The coverage report records
   input counts, weight coverage, inclusion, and exclusion reason.
3. Incomplete four-asset vectors are excluded, never changed to zero. The
   sampler splits each country at every missing calendar year and never treats
   rows across a gap as consecutive.
4. At every block start, `PooledPerspectiveBootstrapMarket` chooses a country
   uniformly, then a valid start year uniformly within that country. It follows
   consecutive years in that segment. `block_size` (10 by default) is the mean
   of the untruncated geometric request. Gaps and endpoints truncate observed
   blocks; at a block end, gap, or endpoint it draws a fresh country and start.
   A simulated path ending can also shorten the final block. The instance-local
   seeded RNG does not alter global random state.
5. `country_sampling="observation_weighted"` is also implemented for
   comparison. It chooses uniformly among all usable country-year starts.
6. `get_diagnostics()` distinguishes `intended_country_start_probabilities`,
   `realized_country_start_counts`, and `realized_country_observation_counts`.
   It also reports country-year counts, valid segment lengths,
   `observed_block_length_counts` after boundary truncation and path-end
   censoring, natural block ends, gap/endpoint restarts, and source coverage.
   `last_observation` identifies the most recently sampled country/year and
   weighting metadata.

The implemented equal-country rule gives each included country equal
probability at each block start, assuming equal relevance for those starts.
It does not give equal realized observation exposure: short histories
contribute fewer observations per start on average because gaps and endpoints
truncate blocks, while long contiguous segments can dominate observed path
years. It is separate from international-equity weights: GDP
weights define the foreign markets inside one country's international-stock
return; country-selection weights define which domestic environment supplies a
bootstrap block.

Because JST asset returns are real, the pooled engine returns only the four
asset returns and retains CPI/inflation as diagnostics. The simulator rejects
inflation-adjustment flags when used with this real-return engine. Annual data
cannot establish monthly sequence risk: the simulator's monthly decumulation
mode still converts one annual return into twelve equal compounded returns.

### Other country-pooling models

Each row below answers a different question. Pooling is not just a way to fill
missing values.

| Pooling model | Country selection | Interpretation and tradeoff |
| --- | --- | --- |
| **Equal-country block starts (implemented baseline)** | Choose a usable country uniformly, then a valid date uniformly within it. | Equal probability at each block start; realized observation exposure depends on boundary truncation. Short histories contribute fewer observations per start on average, and long segments can dominate observed path years. |
| Observation-weighted starts (implemented alternative) | Choose uniformly among all usable country-period rows. | A random recorded country-period; long histories contribute more block starts. Closer to sampling a row from the full pooled panel. |
| One country per simulated path | Choose one country, then sample its segments for the entire lifecycle. | Preserves persistent country differences; limited by that country's usable record and can require repeated block restarts. |
| Peer/similarity-weighted pool | Weight countries using declared features such as currency regime, income, financial-market depth, or sovereign-risk characteristics. | More target-relevant in principle, but peer definitions are judgment calls and can omit important crises. Defer until a target country is specified. |
| Region or regime strata | Select a region or currency/institutional regime, then country and date within it. | Makes group mix explicit; results depend on group definitions and probabilities. |
| Fixed target-country perspective | Use only the target country's local real vector, while retaining its global international-equity basket. | Most directly tied to local inflation, currency and bonds, but has few independent long-horizon histories. Keep as a benchmark. |
| Local/pooled hybrid | Mix complete local and pooled blocks, or choose local versus pooled model once per full path. | Explicit compromise. Mixing within paths allows country environments to switch; choosing a model per path preserves persistent local-versus-pooled uncertainty. The mixture share must be shown and varied. |
| IID pooled observations | Draw individual country-period rows independently. | A useful contrast that destroys multi-period dependencies and underrepresents long crises; not the initial default. |

The paper uses a stationary bootstrap with geometric block lengths averaging
120 months, keeps each four-asset vector intact, and continues an unfinished
block from the beginning of another randomly selected country's history.
The implemented annual sampler instead restarts at country gaps/endpoints. A named
paper-style continuation mode can be compared later; the two boundary rules
create different synthetic transitions and realized country exposure. See
[the paper's simulation procedure, §4.4](https://finance-conference.wpcarey.asu.edu/sites/default/files/2025-01/Beyond%20the%20Status%20Quo.pdf).

### Remaining validation and data gates

The repository does not pin or bundle JST source data. Before interpreting a
run, record the R6 file checksum, country-by-year coverage report, license, and
the exact `gdp_scale_factors` overrides (if any). The default excludes markets
with no observed equity-return series and is strict for missing GDP/FX inputs
and missing returns within the selected foreign universe. Lower return
coverage thresholds must be declared and reported. Validation remains open:
add hand-worked fixtures for lagged GDP weights, FX/CPI transformations,
complete-vector alignment, gaps, block restarts, RNG isolation, and diagnostics.
Compare the baseline with local, peer/region, one-country-path, hybrid,
equal-equity-weight, World Bank/WFE capitalization, and (if cleared for use)
Kuvshinov–Zimmermann capitalization scenarios on matched data where possible.

Report strategy comparisons on the same sampled paths. Separate Monte Carlo
error from uncertainty about the weighting, pooling, eligibility, missing-data,
and boundary assumptions. More simulation trials reduce only the first.
