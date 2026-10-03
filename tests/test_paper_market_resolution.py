import pandas as pd
import pytest

from src.assets import DOMESTIC_STOCK, INTERNATIONAL_STOCK
from src.config import MarketConfig, SimulationConfig
from src.investor import Investor
from src.market import BootstrapMarket, Market, PerspectiveBootstrapMarket, StationaryBootstrapMarket
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy, PaperOptimalStrategy, PaperTDFStrategy, Strategy


PAPER_STRATEGIES = [PaperOptimalStrategy, PaperTDFStrategy]
ENGINES = ["synthetic", "block", "stationary"]
COUNTRY_MAPPING = {
    "dom_label": "USA",
    "intl_assets": {"GBR": 0.3, "JPN": 0.3, "FRA": 0.2, "DEU": 0.2},
}
AGGREGATE_MAPPING = {
    "dom_label": DOMESTIC_STOCK,
    "intl_assets": {INTERNATIONAL_STOCK: 1.0},
}


@pytest.fixture
def country_returns():
    return {
        "USA": 0.20, "GBR": 0.10, "JPN": -0.20, "FRA": 0.30, "DEU": 0.50,
        "Bonds": 0.04, "Bills": -0.02, "Inflation": 0.03,
    }


def lifecycle_config(returns, engine_name, granularity, **changes):
    markets = SimulationConfig.get_world_market_configs()
    if engine_name == "synthetic":
        markets = [
            MarketConfig(name, value, 0.0, 0.0)
            for name, value in returns.items() if name != "Inflation"
        ]
    markets[0].weight = 1.0
    values = dict(
        starting_age=63, retirement_age=65, end_age=68,
        initial_salary=1000.0, savings_rate=1.0, salary_growth_rate=0.0,
        withdrawal_rate=0.06, withdrawal_strategy="fixed_real",
        withdrawal_inflation_adjusted=True,
        decumulation_granularity=granularity, markets=markets,
    )
    values.update(changes)
    config = SimulationConfig(**values)
    config.validate()
    return config


def simulate(tmp_path, returns, engine_name, config, strategy):
    engine = None
    if engine_name != "synthetic":
        path = tmp_path / "returns.csv"
        pd.DataFrame([{"Year": 2000, **returns}]).to_csv(path, index=False)
        engine_type = BootstrapMarket if engine_name == "block" else StationaryBootstrapMarket
        engine = engine_type(str(path), seed=1)
    return Simulator(config).run_stochastic(
        strategy, num_trials=1, market_engine=engine, track_paths=True
    )


def assert_same_result(actual, expected):
    assert actual.paths[0] == pytest.approx(expected.paths[0])
    assert actual.terminal_wealths == pytest.approx(expected.terminal_wealths)
    assert actual.withdrawal_paths[0] == pytest.approx(expected.withdrawal_paths[0])


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_default_country_basket_matches_explicit_constituents_across_retirement(
    tmp_path, country_returns, engine_name, strategy_type, granularity
):
    config = lifecycle_config(country_returns, engine_name, granularity)
    actual = simulate(tmp_path, country_returns, engine_name, config, strategy_type())
    expected = simulate(
        tmp_path, country_returns, engine_name, config, strategy_type(**COUNTRY_MAPPING)
    )
    assert_same_result(actual, expected)
    assert all(value > 0 for value in actual.withdrawal_paths[0][3:])


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
def test_default_paper_strategies_work_with_validated_synthetic_world_configs(strategy_type):
    markets = SimulationConfig.get_world_market_configs()
    markets[0].weight = 1.0
    for market in markets:
        market.expected_return = 0.05
        market.volatility = 0.0
    config = SimulationConfig(starting_age=64, retirement_age=65, end_age=66, markets=markets)
    config.validate()
    result = Simulator(config).run_stochastic(strategy_type(), num_trials=1, track_paths=True)
    assert result.paths[0] == pytest.approx([0.0, 5000.0, 5040.0])


@pytest.mark.parametrize("engine_name", ["block", "stationary"])
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_nan_country_constituents_match_the_explicit_basket(
    tmp_path, country_returns, engine_name, strategy_type, granularity
):
    country_returns["JPN"] = float("nan")
    config = lifecycle_config(country_returns, engine_name, granularity)
    actual = simulate(tmp_path, country_returns, engine_name, config, strategy_type())
    expected = simulate(
        tmp_path, country_returns, engine_name, config, strategy_type(**COUNTRY_MAPPING)
    )
    assert_same_result(actual, expected)


@pytest.mark.parametrize("engine_name", ["block", "stationary"])
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
@pytest.mark.parametrize("aggregate_returns", [(0.45, -0.10), (float("nan"), float("nan"))])
def test_supplied_aggregates_are_authoritative_even_with_countries_and_nans(
    tmp_path, country_returns, engine_name, strategy_type, granularity, aggregate_returns
):
    country_returns[DOMESTIC_STOCK], country_returns[INTERNATIONAL_STOCK] = aggregate_returns
    config = lifecycle_config(country_returns, engine_name, granularity)
    actual = simulate(tmp_path, country_returns, engine_name, config, strategy_type())
    expected = simulate(
        tmp_path, country_returns, engine_name, config, strategy_type(**AGGREGATE_MAPPING)
    )
    country_basket = simulate(
        tmp_path, country_returns, engine_name, config, strategy_type(**COUNTRY_MAPPING)
    )
    assert_same_result(actual, expected)
    assert actual.terminal_wealths[0] != pytest.approx(country_basket.terminal_wealths[0])


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
@pytest.mark.parametrize("aggregate_asset", [DOMESTIC_STOCK, INTERNATIONAL_STOCK])
def test_each_default_component_resolves_independently_in_mixed_sources(
    tmp_path, country_returns, engine_name, strategy_type, granularity, aggregate_asset
):
    country_returns[aggregate_asset] = 0.45
    mapping = dict(COUNTRY_MAPPING)
    if aggregate_asset == DOMESTIC_STOCK:
        mapping["dom_label"] = DOMESTIC_STOCK
    else:
        mapping["intl_assets"] = {INTERNATIONAL_STOCK: 1.0}
    config = lifecycle_config(country_returns, engine_name, granularity)
    actual = simulate(tmp_path, country_returns, engine_name, config, strategy_type())
    expected = simulate(tmp_path, country_returns, engine_name, config, strategy_type(**mapping))
    assert_same_result(actual, expected)


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_custom_country_mapping_uses_its_constituents_for_growth_and_withdrawals(
    tmp_path, country_returns, engine_name, strategy_type, granularity
):
    strategy = strategy_type(dom_label="GBR", intl_assets={"USA": 0.25, "JPN": 0.75})
    config = lifecycle_config(
        country_returns, engine_name, granularity, starting_age=64, end_age=66,
        withdrawal_inflation_adjusted=False,
    )
    result = simulate(tmp_path, country_returns, engine_name, config, strategy)
    allocation = strategy.get_allocation(65)
    wealth = 1000.0
    if granularity == "annual":
        wealth *= 1 + sum(weight * country_returns[asset] for asset, weight in allocation.items())
        wealth -= 60.0
    else:
        growth = sum(weight * (1 + country_returns[asset]) ** (1 / 12)
                     for asset, weight in allocation.items())
        for _ in range(12):
            wealth = (wealth - 5.0) * growth
    assert result.paths[0] == pytest.approx([0.0, 1000.0, wealth])
    assert result.withdrawal_paths[0] == pytest.approx([0.0, 0.0, 60.0])


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("explicit_component", ["domestic", "international"])
def test_explicit_aggregate_labels_fail_with_country_only_sources_before_contribution(
    tmp_path, country_returns, engine_name, strategy_type, explicit_component, monkeypatch
):
    mapping = (
        {"dom_label": DOMESTIC_STOCK} if explicit_component == "domestic"
        else {"intl_assets": {INTERNATIONAL_STOCK: 1.0}}
    )
    missing_asset = DOMESTIC_STOCK if explicit_component == "domestic" else INTERNATIONAL_STOCK

    def unexpected_contribution(*args, **kwargs):
        pytest.fail("Missing returns must fail before contributions")

    monkeypatch.setattr(Investor, "earn_and_save", unexpected_contribution)
    config = lifecycle_config(country_returns, engine_name, "monthly")
    with pytest.raises(ValueError, match=f"{strategy_type.__name__} at age 63:.*{missing_asset}"):
        simulate(tmp_path, country_returns, engine_name, config, strategy_type(**mapping))


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("missing_country", ["USA", "GBR", "JPN", "FRA", "DEU"])
def test_incomplete_synthetic_country_baskets_fail_before_contribution(
    tmp_path, country_returns, strategy_type, missing_country, monkeypatch
):
    del country_returns[missing_country]
    config = lifecycle_config(country_returns, "synthetic", "monthly")
    missing_asset = DOMESTIC_STOCK if missing_country == "USA" else INTERNATIONAL_STOCK

    def unexpected_contribution(*args, **kwargs):
        pytest.fail("Missing returns must fail before contributions")

    monkeypatch.setattr(Investor, "earn_and_save", unexpected_contribution)
    with pytest.raises(ValueError, match=f"{strategy_type.__name__} at age 63:.*{missing_asset}"):
        simulate(tmp_path, country_returns, "synthetic", config, strategy_type())


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("mapping, domestic, international", [
    ({"intl_assets": {"USA": 0.25, "JPN": 0.75}}, {"USA": 1.0}, {"USA": 0.25, "JPN": 0.75}),
    ({"dom_label": "GBR"}, {"GBR": 1.0}, COUNTRY_MAPPING["intl_assets"]),
])
def test_country_resolution_adds_weights_when_default_and_custom_components_overlap(
    country_returns, strategy_type, mapping, domestic, international
):
    strategy = strategy_type(**mapping)
    for age in [63, 65, 67]:
        raw = strategy.get_allocation(age)
        original_raw = dict(raw)
        domestic_weight = raw[strategy.dom_label]
        international_weight = sum(raw[asset] for asset in strategy.intl_assets)
        expected = {asset: weight for asset, weight in raw.items()
                    if asset not in {strategy.dom_label, *strategy.intl_assets}}
        for assets, weight in [(domestic, domestic_weight), (international, international_weight)]:
            for asset, relative_weight in assets.items():
                expected[asset] = expected.get(asset, 0.0) + weight * relative_weight
        actual = strategy.resolve_allocation_for_market(raw, country_returns)
        assert raw == original_raw
        assert actual == pytest.approx(expected)
        assert sum(actual.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("mapping, explicit_asset", [
    ({"intl_assets": {DOMESTIC_STOCK: 1.0}}, DOMESTIC_STOCK),
    ({"dom_label": INTERNATIONAL_STOCK}, INTERNATIONAL_STOCK),
])
def test_defaults_do_not_remap_keys_used_by_explicit_other_components(
    tmp_path, country_returns, strategy_type, mapping, explicit_asset
):
    config = lifecycle_config(country_returns, "block", "monthly")
    with pytest.raises(ValueError, match=explicit_asset):
        simulate(tmp_path, country_returns, "block", config, strategy_type(**mapping))


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
def test_strategy_can_be_reused_between_country_and_aggregate_engines(
    tmp_path, country_returns, strategy_type
):
    strategy = strategy_type()
    original = strategy.get_allocation(63)
    aggregate_returns = {DOMESTIC_STOCK: 0.45, INTERNATIONAL_STOCK: -0.10,
                         "Bonds": 0.04, "Bills": -0.02}
    for returns, engine_name, mapping in [
        (country_returns, "block", COUNTRY_MAPPING),
        (aggregate_returns, "synthetic", AGGREGATE_MAPPING),
        (country_returns, "stationary", COUNTRY_MAPPING),
    ]:
        config = lifecycle_config(returns, engine_name, "monthly")
        actual = simulate(tmp_path, returns, engine_name, config, strategy)
        expected = simulate(tmp_path, returns, engine_name, config, strategy_type(**mapping))
        assert_same_result(actual, expected)
        assert strategy.get_allocation(63) == original
        assert strategy.dom_label == DOMESTIC_STOCK
        assert strategy.intl_assets == {INTERNATIONAL_STOCK: 1.0}


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("allocation_hook", ["get_allocation", "_get_allocation"])
def test_paper_subclass_allocation_overrides_still_control_the_schedule(
    tmp_path, country_returns, strategy_type, allocation_hook
):
    def allocation(self, age):
        return {"USA": 1.0} if age < 65 else {"JPN": 1.0}

    custom_type = type("CustomPaperStrategy", (strategy_type,), {allocation_hook: allocation})
    reference_type = type("ReferenceStrategy", (Strategy,), {"_get_allocation": allocation})
    config = lifecycle_config(country_returns, "block", "monthly")
    actual = simulate(tmp_path, country_returns, "block", config, custom_type())
    expected = simulate(tmp_path, country_returns, "block", config, reference_type())
    assert_same_result(actual, expected)


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
@pytest.mark.parametrize("target, returns, annual_terminal_wealth", [
    ({"Cash": 1.0}, {"Cash": 0.05}, 5040.0),
    ({DOMESTIC_STOCK: 0.4, INTERNATIONAL_STOCK: 0.6},
     {DOMESTIC_STOCK: 0.20, INTERNATIONAL_STOCK: -0.10}, 4896.0),
])
def test_paper_subclass_can_supply_its_own_constructor_without_calling_super(
    strategy_type, granularity, target, returns, annual_terminal_wealth
):
    class CustomPaperStrategy(strategy_type):
        def __init__(self, target):
            self.target = target

        def _get_allocation(self, age):
            return self.target

    markets = [MarketConfig(asset, ret, 0.0, 0.0) for asset, ret in returns.items()]
    markets[0].weight = 1.0
    config = SimulationConfig(
        starting_age=64, retirement_age=65, end_age=66, markets=markets,
        decumulation_granularity=granularity,
    )
    config.validate()
    strategy = CustomPaperStrategy(target)
    original = dict(target)
    actual = Simulator(config).run_stochastic(strategy, num_trials=1, track_paths=True)
    expected = Simulator(config).run_stochastic(
        FixedAllocationStrategy(target), num_trials=1, track_paths=True
    )
    assert_same_result(actual, expected)
    assert strategy.target == original
    if granularity == "annual":
        assert actual.paths[0] == pytest.approx([0.0, 5000.0, annual_terminal_wealth])


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("asset", [DOMESTIC_STOCK, INTERNATIONAL_STOCK])
def test_paper_subclass_without_default_provenance_does_not_remap_custom_labels(
    tmp_path, country_returns, strategy_type, asset
):
    class CustomPaperStrategy(strategy_type):
        def __init__(self, target):
            self.target = target

        def _get_allocation(self, age):
            return self.target

    strategy = CustomPaperStrategy({asset: 1.0})
    config = lifecycle_config(country_returns, "block", "monthly")
    with pytest.raises(ValueError, match=f"CustomPaperStrategy at age 63:.*{asset}"):
        simulate(tmp_path, country_returns, "block", config, strategy)
    assert strategy.target == {asset: 1.0}


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
def test_allocation_is_computed_once_before_each_market_draw(
    country_returns, strategy_type
):
    events = []

    class RecordingStrategy(strategy_type):
        def _get_allocation(self, age):
            events.append(("allocation", age))
            return super()._get_allocation(age)

    class RecordingMarket(Market):
        def get_annual_returns(self):
            events.append(("returns",))
            return dict(country_returns)

    config = lifecycle_config(country_returns, "block", "monthly", starting_age=64, end_age=66)
    Simulator(config).run_stochastic(RecordingStrategy(), num_trials=1, market_engine=RecordingMarket())
    assert events == [("allocation", 64), ("returns",), ("allocation", 65), ("returns",)]


@pytest.mark.parametrize("strategy_type", PAPER_STRATEGIES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
@pytest.mark.parametrize("stationary", [False, True])
def test_perspective_bootstrap_preserves_genuine_aggregate_compounding(
    tmp_path, country_returns, strategy_type, granularity, stationary
):
    path = tmp_path / "jst.csv"
    rows = []
    for year in [1950, 1951]:
        for country in COUNTRY_MAPPING["intl_assets"] | {"USA": 1.0}:
            rows.append(dict(year=year, iso=country, country=country, cpi=1.0, exrat=1.0,
                             eq_tr=country_returns[country], bond_tr=0.04,
                             bill_rate=-0.02, gdp=1000.0))
    pd.DataFrame(rows).to_csv(path, index=False)
    config = lifecycle_config(
        country_returns, "block", granularity, starting_age=64, end_age=67,
        withdrawal_inflation_adjusted=False,
    )

    def run(strategy):
        market = PerspectiveBootstrapMarket(str(path), stationary_bootstrap=stationary, seed=1)
        result = Simulator(config).run_stochastic(
            strategy, num_trials=1, market_engine=market, track_paths=True
        )
        return result, market.get_annual_returns()

    actual, returns = run(strategy_type())
    expected, _ = run(strategy_type(**AGGREGATE_MAPPING))
    assert_same_result(actual, expected)
    assert returns[INTERNATIONAL_STOCK] == pytest.approx(0.175)
    assert "USA" not in returns
    wealth = 1000.0
    expected_path = [0.0, wealth]
    for age in [65, 66]:
        allocation = strategy_type().get_allocation(age)
        if granularity == "annual":
            wealth *= 1 + sum(weight * returns[asset] for asset, weight in allocation.items())
            wealth -= 60.0
        else:
            growth = sum(weight * (1 + returns[asset]) ** (1 / 12)
                         for asset, weight in allocation.items())
            for _ in range(12):
                wealth = (wealth - 5.0) * growth
        expected_path.append(wealth)
    assert actual.paths[0] == pytest.approx(expected_path)
    assert actual.withdrawal_paths[0] == pytest.approx([0.0, 0.0, 60.0, 60.0])
