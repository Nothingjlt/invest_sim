import pandas as pd
import pytest

from src.assets import DOMESTIC_STOCK, INTERNATIONAL_STOCK
from src.config import SimulationConfig
from src.investor import Investor
from src.market import BootstrapMarket, StationaryBootstrapMarket
from src.simulator import Simulator
from src.strategy import PaperOptimalStrategy, PaperTDFStrategy


BOOTSTRAP_ENGINES = [BootstrapMarket, StationaryBootstrapMarket]
PAPER_STRATEGIES = [PaperOptimalStrategy, PaperTDFStrategy]


@pytest.fixture
def country_row():
    return {
        "Year": 2000,
        "USA": 0.20,
        "GBR": 0.10,
        "JPN": -0.20,
        "FRA": 0.30,
        "DEU": 0.50,
        "Bonds": 0.04,
        "Bills": -0.02,
        "Inflation": 0.03,
    }


def write_country_csv(tmp_path, row):
    path = tmp_path / "country_returns.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return str(path)


def short_country_config(**changes):
    markets = SimulationConfig.get_world_market_configs()
    markets[0].weight = 1.0
    values = dict(
        starting_age=25,
        retirement_age=27,
        end_age=28,
        initial_salary=1000.0,
        salary_growth_rate=0.0,
        savings_rate=1.0,
        withdrawal_rate=0.0,
        markets=markets,
    )
    values.update(changes)
    config = SimulationConfig(**values)
    config.validate()
    return config


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
def test_country_bootstrap_preserves_source_series_without_derived_aggregates(
    tmp_path, country_row, engine_type
):
    market = engine_type(write_country_csv(tmp_path, country_row), seed=1)
    returns = market.get_annual_returns()

    assert DOMESTIC_STOCK not in returns
    assert INTERNATIONAL_STOCK not in returns
    assert {name: returns[name] for name in country_row if name != "Year"} == (
        pytest.approx({name: value for name, value in country_row.items() if name != "Year"})
    )


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
def test_missing_country_observation_is_omitted_and_held_asset_fails(
    tmp_path, country_row, engine_type
):
    country_row["JPN"] = float("nan")
    returns = engine_type(write_country_csv(tmp_path, country_row)).get_annual_returns()

    assert "JPN" not in returns
    assert returns["GBR"] == 0.10
    assert INTERNATIONAL_STOCK not in returns

    investor = Investor(25, 0.0, {"GBR": 30.0, "JPN": 30.0, "FRA": 20.0, "DEU": 20.0})
    before = dict(investor.holdings)
    with pytest.raises(ValueError, match="missing return series for JPN"):
        investor.apply_returns(returns)
    assert investor.holdings == before


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
def test_existing_aggregate_series_are_preserved(
    tmp_path, country_row, engine_type
):
    aggregate_value = 0.75
    country_row[DOMESTIC_STOCK] = aggregate_value
    country_row[INTERNATIONAL_STOCK] = aggregate_value
    returns = engine_type(write_country_csv(tmp_path, country_row)).get_annual_returns()

    for name in [DOMESTIC_STOCK, INTERNATIONAL_STOCK]:
        assert returns[name] == aggregate_value

    investor = Investor(25, 0.0, {DOMESTIC_STOCK: 40.0, INTERNATIONAL_STOCK: 60.0})
    investor.apply_returns(returns)
    assert investor.holdings == pytest.approx(
        {DOMESTIC_STOCK: 70.0, INTERNATIONAL_STOCK: 105.0}
    )


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
@pytest.mark.parametrize("missing_asset", [DOMESTIC_STOCK, INTERNATIONAL_STOCK])
def test_missing_aggregate_is_omitted_without_losing_finite_series(
    tmp_path, country_row, engine_type, missing_asset
):
    country_row.update({DOMESTIC_STOCK: 0.0, INTERNATIONAL_STOCK: -1.0})
    country_row[missing_asset] = None
    returns = engine_type(write_country_csv(tmp_path, country_row)).get_annual_returns()
    assert returns == {
        asset: value for asset, value in country_row.items()
        if asset not in {"Year", missing_asset}
    }
    investor = Investor(25, 0.0, {missing_asset: 100.0})
    with pytest.raises(ValueError, match=f"missing return series for {missing_asset}"):
        investor.apply_returns(returns)
    assert investor.total_portfolio_value == 100.0


@pytest.mark.parametrize("invalid_value", [float("inf"), -float("inf")])
def test_nonfinite_bootstrap_observations_fail_closed(tmp_path, country_row, invalid_value):
    country_row["USA"] = invalid_value
    with pytest.raises(ValueError, match="numeric and finite"):
        BootstrapMarket(write_country_csv(tmp_path, country_row))


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
@pytest.mark.parametrize(
    "strategy_type, accumulation_wealth, growth_weights",
    [
        (PaperOptimalStrategy, 2153.80, (0.34, 0.66, 0.0, 0.0)),
        (PaperTDFStrategy, 2107.55, (0.10, 0.07, 0.73, 0.10)),
    ],
)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_default_paper_strategies_grow_with_country_bootstrap_at_current_age(
    tmp_path, country_row, engine_type, strategy_type,
    accumulation_wealth, growth_weights, granularity
):
    config = short_country_config(decumulation_granularity=granularity)
    strategy = strategy_type(retire_age=config.retirement_age)
    result = Simulator(config).run_stochastic(
        strategy,
        num_trials=1,
        market_engine=engine_type(write_country_csv(tmp_path, country_row), seed=1),
        track_paths=True,
    )

    annual_returns = (0.20, 0.13, 0.04, -0.02)
    if granularity == "annual":
        retirement_growth = 1 + sum(
            weight * ret for weight, ret in zip(growth_weights, annual_returns)
        )
    else:
        country_returns = (0.20, 0.10, -0.20, 0.30, 0.50, 0.04, -0.02)
        domestic, international, bonds, bills = growth_weights
        country_weights = (
            domestic, international * 0.3, international * 0.3,
            international * 0.2, international * 0.2, bonds, bills,
        )
        retirement_growth = sum(
            weight * (1 + ret) ** (1 / 12)
            for weight, ret in zip(country_weights, country_returns)
        ) ** 12
    expected_wealth = accumulation_wealth * retirement_growth

    assert result.paths[0] == pytest.approx(
        [0.0, 1000.0, accumulation_wealth, expected_wealth]
    )
    assert result.terminal_wealths == pytest.approx([expected_wealth])
    assert DOMESTIC_STOCK in strategy.get_allocation(25)
    assert INTERNATIONAL_STOCK in strategy.get_allocation(25)


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
@pytest.mark.parametrize(
    "strategy_type, expected_wealth",
    [(PaperOptimalStrategy, 968.00), (PaperTDFStrategy, 1030.20)],
)
def test_explicit_country_mapping_uses_its_own_domestic_and_international_returns(
    tmp_path, country_row, engine_type, strategy_type, expected_wealth
):
    strategy = strategy_type(
        retire_age=26, dom_label="GBR", intl_assets={"USA": 0.25, "JPN": 0.75}
    )
    result = Simulator(short_country_config(retirement_age=26, end_age=27)).run_stochastic(
        strategy,
        num_trials=1,
        market_engine=engine_type(write_country_csv(tmp_path, country_row), seed=1),
        track_paths=True,
    )

    assert result.paths[0] == pytest.approx([0.0, 1000.0, expected_wealth])


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("missing_country", ["USA", "GBR", "JPN", "FRA", "DEU"])
@pytest.mark.parametrize("missing_kind", ["absent", "nan"])
def test_incomplete_country_basket_fails_before_contribution(
    tmp_path, country_row, engine_type, strategy_type, missing_country, missing_kind,
    monkeypatch
):
    if missing_kind == "absent":
        del country_row[missing_country]
    else:
        country_row[missing_country] = float("nan")
    missing_aggregate = DOMESTIC_STOCK if missing_country == "USA" else INTERNATIONAL_STOCK

    def unexpected_contribution(*args, **kwargs):
        pytest.fail("Missing return validation must happen before contribution")

    monkeypatch.setattr(Investor, "earn_and_save", unexpected_contribution)
    monkeypatch.setattr(Investor, "apply_returns", unexpected_contribution)
    with pytest.raises(ValueError, match=f"{strategy_type.__name__} at age 25:.*{missing_aggregate}"):
        Simulator(short_country_config()).run_stochastic(
            strategy_type(),
            num_trials=1,
            market_engine=engine_type(write_country_csv(tmp_path, country_row), seed=1),
        )


@pytest.mark.parametrize("engine_type", BOOTSTRAP_ENGINES)
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_default_paper_strategies_work_with_bundled_country_csv(
    engine_type, strategy_type, granularity
):
    config = short_country_config(
        retirement_age=26, end_age=28, decumulation_granularity=granularity,
        withdrawal_rate=0.06,
    )
    results = []
    for mapping in [
        {},
        {"dom_label": "USA", "intl_assets": {"GBR": 0.3, "JPN": 0.3, "FRA": 0.2, "DEU": 0.2}},
    ]:
        results.append(Simulator(config).run_stochastic(
            strategy_type(retire_age=26, **mapping),
            num_trials=1,
            market_engine=engine_type("data/global_historical_returns.csv", seed=1),
            track_paths=True,
        ))

    assert results[0].paths[0] == pytest.approx(results[1].paths[0])
    assert results[0].terminal_wealths == pytest.approx(results[1].terminal_wealths)
    assert results[0].withdrawal_paths[0] == pytest.approx(results[1].withdrawal_paths[0])
