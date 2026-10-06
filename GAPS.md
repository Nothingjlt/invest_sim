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
