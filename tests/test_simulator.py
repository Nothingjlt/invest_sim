import pytest
from typing import Dict
from src.config import SimulationConfig, MarketConfig
from src.investor import Investor
from src.market import Market
from src.simulator import Simulator, SimulationResult
from src.strategy import FixedAllocationStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

class ConstantMarket(Market):
    """Test double: returns fixed asset returns and an optional inflation rate."""

    def __init__(self, returns: Dict[str, float], inflation: float | None = 0.0):
        self._returns = dict(returns)
        if inflation is not None:
            self._returns["Inflation"] = inflation

    def get_annual_returns(self) -> Dict[str, float]:
        return dict(self._returns)


def _short_config(**kwargs) -> SimulationConfig:
    """A minimal 1-year-retirement config for tightly controlled tests."""
    defaults = dict(
        starting_age=25,
        retirement_age=30,
        end_age=32,
        initial_salary=10_000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
        withdrawal_rate=0.04,
        markets=[MarketConfig(name="Stocks", expected_return=0.0, volatility=0.0, weight=1.0)],
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def test_zero_growth_scenario():
    """
    Mathematical verification of the lifecycle loop.
    With 0% return and 0% salary growth, the terminal wealth should follow:
    Savings = (Years Working) * (Salary * Savings Rate)
    Terminal Wealth = Savings * (1 - Withdrawal Rate) ^ (Retirement Years)
    """
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=100,
        initial_salary=50000.0,
        salary_growth_rate=0.0,  # Fixed salary for easy math
        savings_rate=0.10,  # $5000/year
        withdrawal_rate=0.04,  # 4% of portfolio/year
    )

    working_years = config.retirement_age - config.starting_age  # 40 years
    total_savings = working_years * (
        config.initial_salary * config.savings_rate
    )  # $200,000

    retirement_years = config.end_age - config.retirement_age  # 35 years
    expected_terminal_wealth = total_savings * (
        (1 - config.withdrawal_rate) ** retirement_years
    )

    simulator = Simulator(config)
    actual_terminal_wealth = simulator.run_deterministic(annual_return=0.0)

    # Using pytest.approx for floating point comparisons
    assert actual_terminal_wealth == pytest.approx(expected_terminal_wealth)


def test_accumulation_only():
    """Verify that portfolio equals total savings when retirement and end age are the same."""
    config = SimulationConfig(
        starting_age=20,
        retirement_age=30,
        end_age=30,
        initial_salary=10000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
    )
    # 10 years * ($10000 * 0.1) = $10000
    simulator = Simulator(config)
    assert simulator.run_deterministic(annual_return=0.0) == pytest.approx(10000.0)


def test_investor_withdraw_and_rebalance_supports_new_retirement_assets():
    investor = Investor(
        age=65,
        current_salary=0.0,
        holdings={"Domestic Stock": 100.0, "International Stock": 200.0},
    )
    target_alloc = {"Domestic Stock": 0.26, "International Stock": 0.47, "Bills": 0.27}
    investor.withdraw(60.0, target_alloc)
    investor.rebalance(target_alloc)

    assert investor.total_portfolio_value == pytest.approx(240.0)
    assert all(value >= 0.0 for value in investor.holdings.values())


# ── Path tracking ──────────────────────────────────────────────────────────────

def _minimal_config() -> SimulationConfig:
    """A short, deterministic config for path-tracking tests."""
    return SimulationConfig(
        starting_age=25,
        retirement_age=30,
        end_age=32,
        initial_salary=10_000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
        withdrawal_rate=0.04,
        markets=[MarketConfig(name="Stocks", expected_return=0.0, volatility=0.0, weight=1.0)],
    )


def test_path_tracking_disabled_by_default():
    """Without track_paths outcomes remain list-compatible and identify their source."""
    config = _minimal_config()
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    result = sim.run_stochastic(strategy, num_trials=3)

    assert isinstance(result, list), "Default return type must be List[float]"
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)
    assert result.provenance.kind == "synthetic"


def test_path_tracking_shape_and_values():
    """track_paths=True returns SimulationResult with correct shape and values."""
    config = _minimal_config()
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})

    num_trials = 5
    result = sim.run_stochastic(strategy, num_trials=num_trials, track_paths=True)

    assert isinstance(result, SimulationResult)
    assert result.provenance.kind == "synthetic"
    assert result.terminal_wealths.provenance == result.provenance
    assert len(result.terminal_wealths) == num_trials
    assert len(result.paths) == num_trials
    assert len(result.withdrawal_paths) == num_trials

    expected_steps = config.end_age - config.starting_age + 1  # +1 for t=0 snapshot
    for i, path in enumerate(result.paths):
        # Each path must have exactly (years simulated + 1) entries.
        assert len(path) == expected_steps, (
            f"Trial {i}: expected {expected_steps} path entries, got {len(path)}"
        )
        # The last entry in the path must match terminal_wealths.
        assert path[-1] == pytest.approx(result.terminal_wealths[i])
        # All values must be non-negative (no debt modelled).
        assert all(v >= 0.0 for v in path), f"Trial {i} contains negative wealth"
        # Path must be monotonically non-decreasing during accumulation phase
        # (0% return, 0% growth, only contributions — wealth can only stay flat or rise).
        accumulation_end = config.retirement_age - config.starting_age  # step index
        for j in range(1, accumulation_end + 1):
            assert path[j] >= path[j - 1] - 1e-9, (
                f"Trial {i}: wealth decreased during accumulation at step {j}"
            )

    for i, w_path in enumerate(result.withdrawal_paths):
        assert len(w_path) == expected_steps, (
            f"Trial {i}: expected {expected_steps} withdrawal path entries, got {len(w_path)}"
        )
        # All values must be non-negative
        assert all(v >= 0.0 for v in w_path), f"Trial {i} contains negative withdrawal"
        
        # Accumulation phase: ages 25 to 29 (indices 0 to 5) should have 0.0 withdrawal.
        accumulation_end = config.retirement_age - config.starting_age  # index 5 (age 30)
        assert all(w == 0.0 for w in w_path[:accumulation_end + 1]), (
            f"Trial {i} has non-zero withdrawal during accumulation: {w_path[:accumulation_end + 1]}"
        )

        # Decumulation phase: ages 30 and 31 (indices 6 and 7) should have positive withdrawals.
        assert all(w > 0.0 for w in w_path[accumulation_end + 1:]), (
            f"Trial {i} has non-positive withdrawal during decumulation: {w_path[accumulation_end + 1:]}"
        )


# ── Withdrawal modifier tests ──────────────────────────────────────────────────

def test_fixed_real_inflation_adjusts_withdrawal():
    """fixed_real withdrawal grows each year by realized inflation when flag is set."""
    inflation_rate = 0.05  # 5%
    config = _short_config(
        withdrawal_strategy="fixed_real",
        withdrawal_inflation_adjusted=True,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=inflation_rate)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    # w[0..5] are accumulation zeros; w[6] is retirement-year-1, w[7] is year-2
    ret_start = config.retirement_age - config.starting_age  # index 5 (snapshot at age 30)
    w1 = w[ret_start + 1]  # first retirement withdrawal (age 30)
    w2 = w[ret_start + 2]  # second retirement withdrawal (age 31)

    assert w1 > 0.0, "First retirement withdrawal should be positive"
    assert w2 == pytest.approx(w1 * (1 + inflation_rate), rel=1e-6), (
        f"Expected withdrawal to grow by {inflation_rate:.0%}: {w1} -> {w2}"
    )


def test_fixed_real_no_inflation_when_flag_off():
    """When withdrawal_inflation_adjusted=False (default), withdrawal stays constant."""
    config = _short_config(
        withdrawal_strategy="fixed_real",
        withdrawal_inflation_adjusted=False,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=0.10)  # high inflation, ignored

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    w1 = w[ret_start + 1]
    w2 = w[ret_start + 2]

    assert w1 > 0.0
    assert w2 == pytest.approx(w1, rel=1e-9), "Withdrawal should be flat when flag is off"


def test_withdrawal_cap_clamps_variable_pct():
    """Withdrawal cap limits the amount drawn even with a large portfolio."""
    # Portfolio at retirement: 5 working years × $1000/yr = $5000
    # variable_pct 4% of $5000 = $200 gross. Cap at $100.
    config = _short_config(
        withdrawal_strategy="variable_pct",
        withdrawal_rate=0.04,
        withdrawal_cap=100.0,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=0.0)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    for withdrawal in w[ret_start + 1:]:
        assert withdrawal <= 100.0 + 1e-9, f"Withdrawal {withdrawal} exceeded cap of 100"


def test_withdrawal_cap_inflation_adjusts():
    """When withdrawal_cap_inflation_adjusted=True, the cap grows each year."""
    inflation_rate = 0.05
    initial_cap = 50.0
    config = _short_config(
        withdrawal_strategy="variable_pct",
        withdrawal_rate=0.99,       # Very aggressive — always hits the cap
        withdrawal_cap=initial_cap,
        withdrawal_cap_inflation_adjusted=True,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=inflation_rate)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    w1 = w[ret_start + 1]
    w2 = w[ret_start + 2]

    # Year-2 cap = initial_cap * (1 + inflation_rate); withdrawal should be close to year-2 cap
    expected_cap_yr2 = initial_cap * (1 + inflation_rate)
    # w1 is bounded by initial cap; w2 by the grown cap
    assert w2 >= w1 - 1e-9, "Cap-adjusted withdrawal should be at least as large in year 2"
    assert w2 <= expected_cap_yr2 + 1e-9


def test_withdrawal_floor_draws_from_portfolio():
    """If social security doesn't cover the floor, portfolio makes up the difference."""
    # No SS, floor = 200. variable_pct 4% of a $5000 portfolio = $200 already meets the floor.
    # Use floor = 300 to force a top-up beyond the natural 4% withdrawal.
    config = _short_config(
        withdrawal_strategy="variable_pct",
        withdrawal_rate=0.04,
        social_security_benefit=0.0,
        withdrawal_floor=300.0,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=0.0)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    for withdrawal in w[ret_start + 1:]:
        assert withdrawal >= 300.0 - 1e-9, (
            f"Withdrawal {withdrawal} fell below floor of 300"
        )


def test_withdrawal_floor_no_draw_if_ss_covers():
    """If social security fully covers the floor, no extra portfolio draw is needed."""
    # SS = 500, floor = 300. Natural 4% withdrawal is small but SS > floor, so
    # net_from_portfolio = max(0, natural - SS) and floor_from_portfolio = max(0, 300 - 500) = 0
    config = _short_config(
        withdrawal_strategy="variable_pct",
        withdrawal_rate=0.04,
        social_security_benefit=500.0,
        withdrawal_floor=300.0,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=0.0)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    # SS covers everything — the portfolio withdrawal should be 0
    for withdrawal in w[ret_start + 1:]:
        assert withdrawal == pytest.approx(0.0, abs=1e-9), (
            f"Expected no portfolio draw when SS covers floor; got {withdrawal}"
        )


def test_withdrawal_floor_inflation_adjusts():
    """When withdrawal_floor_inflation_adjusted=True, the floor grows each year."""
    inflation_rate = 0.05
    initial_floor = 50.0
    # Very small withdrawal_rate ensures variable_pct < floor always
    config = _short_config(
        withdrawal_strategy="variable_pct",
        withdrawal_rate=0.001,
        social_security_benefit=0.0,
        withdrawal_floor=initial_floor,
        withdrawal_floor_inflation_adjusted=True,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=inflation_rate)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    w1 = w[ret_start + 1]
    w2 = w[ret_start + 2]

    expected_floor_yr2 = initial_floor * (1 + inflation_rate)
    assert w1 == pytest.approx(initial_floor, rel=1e-6), (
        f"Year-1 withdrawal should equal initial floor; got {w1}"
    )
    assert w2 == pytest.approx(expected_floor_yr2, rel=1e-6), (
        f"Year-2 withdrawal should equal inflated floor {expected_floor_yr2}; got {w2}"
    )


def test_cap_and_floor_combined():
    """With both cap and floor active, withdrawal is bounded in both directions."""
    # floor=100, cap=200. Natural 4% of ~$500 portfolio = $20, which is below floor.
    # After floor is applied, withdrawal = 100. Well below cap of 200.
    config = _short_config(
        withdrawal_strategy="variable_pct",
        withdrawal_rate=0.04,
        social_security_benefit=0.0,
        withdrawal_cap=200.0,
        withdrawal_floor=100.0,
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    market = ConstantMarket({"Stocks": 0.0}, inflation=0.0)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    for withdrawal in w[ret_start + 1:]:
        assert withdrawal >= 100.0 - 1e-9, f"Withdrawal {withdrawal} below floor"
        assert withdrawal <= 200.0 + 1e-9, f"Withdrawal {withdrawal} above cap"


def test_no_inflation_key_defaults_to_zero():
    """Market with no 'Inflation' key runs without error; trackers stay flat."""
    config = _short_config(
        withdrawal_strategy="fixed_real",
        withdrawal_inflation_adjusted=True,  # Flag is on, but market won't provide inflation
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    # inflation=None means no 'Inflation' key emitted
    market = ConstantMarket({"Stocks": 0.0}, inflation=None)

    result = sim.run_stochastic(strategy, num_trials=1, market_engine=market, track_paths=True)

    w = result.withdrawal_paths[0]
    ret_start = config.retirement_age - config.starting_age
    w1 = w[ret_start + 1]
    w2 = w[ret_start + 2]

    # With 0% effective inflation, fixed_real withdrawal must stay constant
    assert w1 > 0.0
    assert w2 == pytest.approx(w1, rel=1e-9), (
        "With no inflation key, withdrawal should stay flat (zero inflation default)"
    )


# ── Monthly Decumulation Tests ────────────────────────────────────────────────

class TestMonthlyDecumulation:
    """Tests for decumulation_granularity='monthly'."""

    def _monthly_config(self, **overrides) -> SimulationConfig:
        defaults = dict(
            starting_age=25,
            retirement_age=30,
            end_age=33,
            initial_salary=12_000.0,
            salary_growth_rate=0.0,
            savings_rate=0.10,
            withdrawal_rate=0.04,
            decumulation_granularity="monthly",
            markets=[MarketConfig(name="Stocks", expected_return=0.0,
                                  volatility=0.0, weight=1.0)],
        )
        defaults.update(overrides)
        return SimulationConfig(**defaults)

    def test_zero_return_fixed_real_total_withdrawal(self):
        """
        With 0% return and fixed_real, the total amount withdrawn from the
        portfolio over each retirement year should equal
        withdrawal_rate * initial_portfolio  (no SS, no cap, no floor).
        Monthly sub-stepping must sum to the same annual figure.
        """
        config = self._monthly_config(withdrawal_strategy="fixed_real")
        sim = Simulator(config)
        strategy = FixedAllocationStrategy({"Stocks": 1.0})
        market = ConstantMarket({"Stocks": 0.0}, inflation=None)

        result = sim.run_stochastic(
            strategy, num_trials=1, market_engine=market, track_paths=True
        )

        wdrawals = result.withdrawal_paths[0]
        # retirement starts at index = retirement_age - starting_age = 5
        ret_start = config.retirement_age - config.starting_age

        # Each retirement year should withdraw exactly the same fixed amount
        w_year1 = wdrawals[ret_start + 1]
        w_year2 = wdrawals[ret_start + 2]

        assert w_year1 > 0.0, "Expected non-zero withdrawal in first retirement year"
        assert w_year2 == pytest.approx(w_year1, rel=1e-9), (
            "Fixed-real withdrawal must stay constant under 0% return and 0% inflation"
        )

    def test_zero_return_portfolio_decreases_monotonically(self):
        """
        With 0% return and variable_pct withdrawals, the portfolio must
        strictly decrease each retirement year (no magical growth).
        """
        config = self._monthly_config(withdrawal_strategy="variable_pct")
        sim = Simulator(config)
        strategy = FixedAllocationStrategy({"Stocks": 1.0})
        market = ConstantMarket({"Stocks": 0.0}, inflation=None)

        result = sim.run_stochastic(
            strategy, num_trials=1, market_engine=market, track_paths=True
        )

        path = result.paths[0]
        ret_start = config.retirement_age - config.starting_age
        retirement_path = path[ret_start:]

        for i in range(1, len(retirement_path)):
            assert retirement_path[i] < retirement_path[i - 1], (
                f"Portfolio must decrease in retirement (year {i}): "
                f"{retirement_path[i]:.2f} >= {retirement_path[i-1]:.2f}"
            )

    def test_monthly_vs_annual_produces_different_results_under_volatility(self):
        """
        Under non-zero returns, monthly and annual granularities must produce
        numerically different terminal wealth distributions (sequence-of-returns
        timing differs), while both remaining positive under moderate growth.
        """
        base_kw = dict(
            starting_age=25,
            retirement_age=30,
            end_age=40,
            initial_salary=12_000.0,
            salary_growth_rate=0.0,
            savings_rate=0.10,
            withdrawal_rate=0.04,
            markets=[MarketConfig(name="Stocks", expected_return=0.0,
                                  volatility=0.0, weight=1.0)],
        )
        config_annual = SimulationConfig(
            decumulation_granularity="annual", **base_kw
        )
        config_monthly = SimulationConfig(
            decumulation_granularity="monthly", **base_kw
        )

        # Use a positive constant market so both survive
        market_a = ConstantMarket({"Stocks": 0.07}, inflation=None)
        market_m = ConstantMarket({"Stocks": 0.07}, inflation=None)

        strat = FixedAllocationStrategy({"Stocks": 1.0})

        tw_annual = Simulator(config_annual).run_stochastic(
            strat, num_trials=1, market_engine=market_a
        )
        tw_monthly = Simulator(config_monthly).run_stochastic(
            strat, num_trials=1, market_engine=market_m
        )

        # Monthly withdrawals happen before growth each month, so the portfolio
        # should be slightly smaller (withdrawal-first effect).
        assert tw_annual[0] != pytest.approx(tw_monthly[0], rel=1e-6), (
            "Annual and monthly granularity must yield different terminal wealth"
        )
        assert tw_annual[0] > 0.0
        assert tw_monthly[0] > 0.0

    def test_monthly_withdrawal_paths_length_matches_annual(self):
        """
        withdrawal_paths entries must have the same length regardless of
        decumulation_granularity (both are year-indexed).
        """
        base_kw = dict(
            starting_age=25,
            retirement_age=30,
            end_age=33,
            initial_salary=12_000.0,
            salary_growth_rate=0.0,
            savings_rate=0.10,
            withdrawal_rate=0.04,
            markets=[MarketConfig(name="Stocks", expected_return=0.0,
                                  volatility=0.0, weight=1.0)],
        )
        market = ConstantMarket({"Stocks": 0.0}, inflation=None)
        strat = FixedAllocationStrategy({"Stocks": 1.0})

        for gran in ("annual", "monthly"):
            config = SimulationConfig(decumulation_granularity=gran, **base_kw)
            result = Simulator(config).run_stochastic(
                strat, num_trials=1, market_engine=market, track_paths=True
            )
            expected_len = config.end_age - config.starting_age + 1  # + initial snapshot
            assert len(result.withdrawal_paths[0]) == expected_len, (
                f"granularity={gran!r}: expected path length {expected_len}, "
                f"got {len(result.withdrawal_paths[0])}"
            )
            assert len(result.paths[0]) == expected_len

    def test_monthly_social_security_reduces_portfolio_withdrawal(self):
        """
        With social_security_benefit = full annual withdrawal, the portfolio
        withdrawal should be 0 for each month (portfolio never touched).
        """
        withdrawal_rate = 0.04
        initial_salary = 10_000.0
        savings_rate = 0.10
        retirement_savings = (30 - 25) * initial_salary * savings_rate  # 5000.0
        # Set SS to cover the full annual withdrawal
        full_annual_w = retirement_savings * withdrawal_rate

        config = self._monthly_config(
            withdrawal_strategy="variable_pct",
            social_security_benefit=full_annual_w * 12,  # generous: covers all
            initial_salary=initial_salary,
            savings_rate=savings_rate,
        )
        sim = Simulator(config)
        strategy = FixedAllocationStrategy({"Stocks": 1.0})
        market = ConstantMarket({"Stocks": 0.0}, inflation=None)

        result = sim.run_stochastic(
            strategy, num_trials=1, market_engine=market, track_paths=True
        )

        ret_start = config.retirement_age - config.starting_age
        retirement_withdrawals = result.withdrawal_paths[0][ret_start + 1:]
        for w in retirement_withdrawals:
            assert w == pytest.approx(0.0, abs=1e-9), (
                "With SS >= full withdrawal, portfolio should not be touched"
            )
