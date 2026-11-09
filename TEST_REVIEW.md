# Independent Test Review

**Date:** October 4, 2026  
**Reviewed revision:** `6178218` on `master`, including the current uncommitted policy corrections  
**Reviewer:** independent sub-agent using GPT-6 Astra, xhigh reasoning  
**Scope:** all 111 test functions in 17 test modules, the code under test, and the supplied paper text at `/workspace/invest_sim_review/paper.txt`.

The independent reviewer made no changes. The complete suite passed: **399 cases in 6.42 seconds**. A separate in-memory mutation replacing the stationary-bootstrap return methods with the fixed-block sampler also passed all 399 cases in 4.43 seconds, showing that the suite currently does not distinguish those sampling methods.

## Findings

### Fixed-weight correction

The recent correction is supported by the paper. Section 4.1 describes 13 age windows for the optimal age-based policy (paper text line 751); Table III separately reports the optimal fixed-weight policy as 34% domestic and 66% international, with no bonds or bills (lines 1214 and 2579). The revised `PaperOptimalStrategy`, its market defaults, and the updated allocation, country-resolution, and bootstrap expectations now reflect the fixed-weight policy. The previous retirement cash transition was unsupported by this paper version.

The exact 13 age-window weights are not tabulated in the supplied text; Table III reports ranges. `PaperOptimalStrategy` should therefore not be presented as the age-based optimizer. The TDF section likewise points to a curve in Figure 1 while Table III provides ranges rather than point-by-point weights. The revised code and tests appropriately identify `PaperTDFStrategy` as a linear approximation rather than a reconstruction of that figure.

**Subsequent architecture update:** market factories now return asset descriptors
without portfolio weights. The 34/66 policy belongs to `PaperOptimalStrategy`;
every other strategy likewise owns its allocation. Factory/configuration
regressions and independently calculated strategy ownership outcomes are covered
in `tests/test_config.py` and `tests/test_allocation_ownership.py`. Complete
explicit legacy market weights remain validated metadata and never override a
strategy. The review above describes the earlier revision.

### Compatibility regression found and fixed before merge

The independent audit found that the shared market resolver treated `PaperOptimalStrategy`'s compatibility-only `bills_label` as an explicitly requested asset at [src/strategy.py:174](src/strategy.py#L174). With country-only return series, the default strategy resolved successfully, while either of these failed because resolution of a default equity sleeve was suppressed:

```python
PaperOptimalStrategy(bills_label="Domestic Stock")
PaperOptimalStrategy(bills_label="International Stock")
```

**Resolution on this PR branch:** the resolver now treats bills and bonds labels as active components only for the TDF strategy. The initial overlap cases covered unused labels `USA` and `JPN` but not the canonical aggregate names. The added [test_optimal_ignored_bills_label_does_not_change_country_resolution](tests/test_paper_market_resolution.py#L170) covers both canonical names and both country labels in annual and monthly simulations. The TDF's active bills-label behavior remains covered.

### Tests with weak or ineffective assertions

- [test_social_security_reduces_withdrawals](tests/test_increment_10.py#L85) implements the offset in the test and calls `Investor.withdraw(0)`; it does not exercise simulator Social Security behavior.
- [test_withdrawal_cap_inflation_adjusts](tests/test_simulator.py#L251) accepts unchanged caps of `50, 50`. Disabling cap indexation still passes; the stated 5% case should assert `50, 52.5`.
- The monthly Social Security test at [tests/test_simulator.py:559](tests/test_simulator.py#L559) overfunds the withdrawal by a factor of twelve. Removing the production `/12` still passes. Use partial coverage and assert the exact residual portfolio draw.
- [test_zero_return_fixed_real_total_withdrawal](tests/test_simulator.py#L430) checks equality across years rather than the expected annual total of `$240`.
- The combined cap/floor test at [tests/test_simulator.py:360](tests/test_simulator.py#L360) does not force either bound to bind; the natural draw is already inside the range.
- [test_monthly_vs_annual_produces_different_results_under_volatility](tests/test_simulator.py#L483) uses constant returns despite its name. For its proportional-withdrawal setup, the annual and monthly retention factors are `(1 + R) * (1 - q)` and `(1 + R) * (1 - q/12)^12`; the monthly factor is larger for `0 < q < 1`, contrary to the test comment.
- The two stochastic paper-strategy smoke tests use zero returns and check only repeatability/nonnegativity rather than the independently calculated terminal wealth.
- [test_diversification_safety_comparison](tests/test_world_market.py#L81) does not compare the portfolios, uses unpaired random paths, and only checks that quantiles are nonnegative.
- Most plotting tests check labels or artist counts, not plotted values. Preserve and extend the meaningful numeric median assertion in `test_plot_withdrawal_comparison`.

### Paper-model mismatches and unsupported claims

The current simulator offers useful simplified rules, but several tests must not be treated as validation of the paper's economic model:

- Equations (3)–(4) define monthly, age- and wealth-dependent saving and withdrawal policies (paper text lines 767–804). Current tests cover fixed annual saving/withdrawal rates.
- Equation (11) defines consumption as `max(D + SS, SSI)` (line 831). The simulator instead subtracts configured Social Security from a spending target and uses a portfolio-funded floor. Keep such tests as extension tests, not paper-equation tests.
- Section 4.3 describes pooled monthly return sampling and geometric blocks with a 120-month average, including a country-boundary continuation rule (line 1003). Single-row bootstrap fixtures cannot test those behaviors.
- Appendix A.3, equations (A5)–(A6), uses lagged USD market-capitalization weights for international returns (line 3324). [test_data_loader.py:49](tests/test_data_loader.py#L49) tests contemporaneous GDP/FX weights instead.
- [test_calibrated_returns_statistical_tolerance](tests/test_calibration.py#L6) treats the paper's monthly geometric means as annual arithmetic means multiplied by twelve. Table II defines the reported statistic at paper text line 2497. For example, the USA monthly geometric mean of 0.52% compounds to about 6.42% annual geometric growth, not a 6.24% annual arithmetic mean. The purported 1,000-year sample retains only 134 USA/GBR observations after eligibility masking and does not test correlations, serial dependence, or tails.
- README claims at [README.md:21](README.md#L21), [README.md:217](README.md#L217), and a paper-timing comment at [src/simulator.py:179](src/simulator.py#L179) overstate how closely the stationary sampler, GDP-weighted engine, and monthly simulation match the paper. Clarify these as approximations/extensions unless the underlying definitions are implemented.

### Additional uncovered implementation defects

These behaviors were reproduced during the review and are not currently guarded by tests:

| Area | Observed behavior | Useful missing test |
| --- | --- | --- |
| Allocation validation | `PaperOptimalStrategy(intl_assets={"A": -0.1, "B": 1.1})` accepts a negative holding; NaN weights can also pass validation. | Reject negative and nonfinite constituent weights and allocations. |
| Label collision | `BalancedStrategy("A", "A")` silently returns a 40% allocation. | Reject collisions or explicitly define merge behavior. |
| Withdrawal atomicity | `test_investor_withdraw_and_rebalance_supports_new_retirement_assets` creates negative Bills holdings before rebalancing, then hides them. | Assert holdings immediately after withdrawal and specify shortfall behavior. |
| API parity | One contribution followed by two zero-return fixed-real withdrawal years yields `$4,050` deterministically versus `$4,000` stochastically. | Compare public deterministic and stochastic cash-flow contracts. |
| Monthly metadata | Monthly return conversion raises `TypeError` when `Inflation` is `None`. | Test missing, `None`, NaN, and nonfinite metadata values. |
| Historical eligibility | Chile data from 1980 is accepted despite the paper's Table II sample periods of 1927–1970 and 2010–2023. | Test country-specific eligibility intervals and disjoint sample windows. |

The historical-eligibility finding above is now addressed: the registry stores
disjoint intervals, and `tests/test_data_loader.py` verifies Chile's excluded
gap, eligibility at both return endpoints, and missing in-sample coverage.
The other findings in this review retain their separate status.

Other high-value gaps include partial insolvency withdrawals, exact monthly cap/floor indexation, invalid returns from CSV and custom engines, seeded RNG isolation, loader failures/empty panels, non-USA FX perspectives, and savings-target infeasibility.

## Test-module disposition

The adapter/resolution and bootstrap-compatibility suites contain 299 of 399 cases (about 75%). Preserve the distinct provenance, alias, overlap, reuse, and subclass contracts, but consider reducing repetitive combinations over one-row data that cannot distinguish samplers.

| Module | Functions / cases | Independent assessment |
| --- | ---: | --- |
| `test_allocation_timing.py` | 3 / 4 | Strong cash-flow and rebalancing regressions; keep. |
| `test_asset_names.py` | 9 / 16 | Useful missing-key and atomicity checks; current retirement-switch fixture is appropriate. |
| `test_bootstrap.py` | 5 / 5 | Fixed-block checks help; stationary/perspective behavior is under-tested. |
| `test_bootstrap_compatibility.py` | 7 / 48 | Revised numerical expectations are coherent; reduce duplicate one-row engine cases. |
| `test_calibration.py` | 1 / 1 | Paper statistic and annualization are misinterpreted; redefine or label as synthetic calibration. |
| `test_config.py` | 5 / 5 | Basic validation only; add malformed-boundary and factory/API consistency cases. |
| `test_data_loader.py` | 3 / 3 | Basic FX arithmetic is useful; weighting, eligibility, non-USA perspective, and failures are missing. |
| `test_increment_10.py` | 4 / 4 | Some generic withdrawal examples help; manual Social Security assertion is ineffective. |
| `test_market.py` | 5 / 11 | Useful moments/bounds/replay checks; add RNG isolation and custom-return safety. |
| `test_metrics.py` | 3 / 3 | Basic arithmetic is useful; infeasible targets and paper welfare are absent. |
| `test_multi_asset.py` | 2 / 2 | Keep explicit rebalancing arithmetic; consolidate random positivity smoke. |
| `test_paper_market_resolution.py` | 18 / 251 | Distinct resolver contracts are useful; many combinations are repetitive and share oracles. |
| `test_paper_strategies.py` | 6 / 6 | Fixed policy and approximation labels are now sound; stochastic smoke assertions remain weak. |
| `test_simulator.py` | 19 / 19 | Good scaffolding; strengthen exact amounts, insolvency, and cross-feature cases. |
| `test_strategy.py` | 6 / 6 | Useful ordinary arithmetic; malformed-input and alias boundaries are missing. |
| `test_visualization.py` | 9 / 9 | Mostly presentation smoke; verify the numeric plotted series. |
| `test_world_market.py` | 6 / 6 | Country mapping correction is sound; diversification comparison is ineffective. |

## Suggested order of work

1. Keep the `bills_label` country-resolution regression covered; validate allocation finiteness/sign constraints and decide label-collision behavior.
2. Strengthen the exact withdrawal, indexation, Social Security, and cap/floor tests so disabling the intended production behavior fails.
3. Correct calibration and documentation claims; clearly label simplified behavior as an extension rather than paper reproduction.
4. Add controlled multi-row bootstrap and market-weight fixtures, plus data eligibility and failure cases.
5. Treat the full paper model—age-window schedule, saving/withdrawal optimization, household SS/survivor rules, pooled monthly data, utility, and outcome measures—as a separate implementation and benchmark effort.

The test suite is valuable for software regression coverage, but its current passing result does not establish paper fidelity or numerical reproduction of the paper's reported optimization results.
