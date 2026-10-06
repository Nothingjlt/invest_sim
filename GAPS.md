# Open Gaps

## Paper-compatible handling of incomplete return histories

**Status:** Open

The current return panel and bootstrap engines do not yet implement the paper's
missing-data and country-block bootstrap design.

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

### Current implementation gap

The direct [`BootstrapMarket`](src/market.py) and
`StationaryBootstrapMarket` operate on a global, wide row index. They omit blank
fields from a sampled row and fail closed if an active holding has no return;
they do not switch to another country's history when a country-specific block
ends. The [`PerspectiveBootstrapMarket`](src/market.py) path currently
complete-case filters its derived four-column annual output in
[`JSTDataLoader`](src/data_loader.py), but it still does not preserve the
paper's country-specific monthly block structure.

This is safer than treating a missing return as 0% growth, but it is not yet a
paper-faithful missing-row or block-bootstrap implementation.

### Proposed next steps

1. Add a validated long-form historical input schema with one country-period
   observation and all four required asset returns present.
2. Keep country sample bounds explicit and exclude incomplete country-periods
   during preprocessing rather than encoding them as usable partial rows.
3. Implement a country-aware stationary/block sampler that draws consecutive
   observations within a country and, at a country-history boundary, starts the
   remaining block at the beginning of another randomly selected country's
   sample, matching the paper.
4. Add coverage diagnostics showing excluded observations by country and period,
   and retain provenance distinguishing the bundled synthetic panel from a
   verified historical release.
5. Add seeded tests for complete four-asset vectors, country-boundary fallback,
   no within-row asset mixing, and rejection of incomplete active observations.

### Acceptance criteria

- Missing values never become implicit zero returns.
- Every sampled paper-style observation contains all four asset returns.
- A block does not mix countries within an observation and handles country
  history boundaries using the documented fallback.
- Coverage counts and excluded country-periods are reproducible from the input
  data and visible to callers.
- Seeded reference paths demonstrate the intended behavior for both synthetic
  fixtures and a declared historical dataset.

## Related implementation and paper-fidelity gaps

### International-stock weighting

**Status:** Open

The paper's Appendix A.3 value-weights international stock returns using each
country's lagged total market capitalization in USD. The current
[`JSTDataLoader`](src/data_loader.py) instead uses contemporaneous GDP divided
by the current exchange rate when `weight_method="gdp"`. This is a useful
proxy, but it is not the paper's weighting method and can change every derived
international-stock return.

The historical input schema and processing path should eventually expose the
paper's market-capitalization series, implement the lagged USD weights, and add
reference tests against hand-calculated country-period examples. Until then,
documentation should describe GDP weighting as a proxy rather than a paper
replication.

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

**Status:** Open

CSV bootstrap constructors currently accept invalid or unusable inputs too late:

- `block_size=0` is accepted; stationary sampling later divides by zero, while
  fixed-block sampling silently starts a new block for every draw;
- empty panels fail later with low-level indexing errors; and
- `_observed_returns` rejects non-finite values but permits simple returns below
  `-1`, which are economically invalid and may only fail once an affected asset
  is held.

Add constructor-level validation for positive integer block sizes, fail clearly
on empty panels (including an empty processed perspective panel), and reject
returns below `-1` before sampling or simulation. Add focused tests for each
failure mode and for both fixed and stationary engines.

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

### Additional tests needed

The open gaps above should be covered by:

- multi-row stationary and fixed-block fixtures that exercise country-history
  boundaries and continuation behavior;
- lagged market-cap weighting and disjoint country-eligibility fixtures;
- invalid block sizes, empty panels, and returns below `-1`;
- a factory-contract test for `get_world_market_configs()`; and
- a lightweight documentation/review check that paper-compatibility claims
  continue to link to this file.
