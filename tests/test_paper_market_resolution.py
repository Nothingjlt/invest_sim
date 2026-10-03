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

# Independent balances: JPN keeps its 30% share of the international sleeve
# with growth factor 1. Annual growth precedes withdrawals; monthly withdrawals
# precede separately compounded constituent growth and monthly rebalancing.
NAN_COUNTRY_PATHS = {
    (PaperOptimalStrategy, "annual"): [0.0, 1000.0, 2193.3, 2359.77147, 2572.008649249, 2840.985625068],
    (PaperOptimalStrategy, "monthly"): [0.0, 1000.0, 2193.3, 2329.822167282, 2505.266440816, 2728.657321248],
    (PaperTDFStrategy, "annual"): [0.0, 1000.0, 2063.4975, 2064.52924875, 2061.909122799, 2055.304754864],
    (PaperTDFStrategy, "monthly"): [0.0, 1000.0, 2063.4975, 2055.544392157, 2043.299873330, 2026.400076736],
}
NAN_COUNTRY_WITHDRAWALS = {
    PaperOptimalStrategy: [0.0, 0.0, 0.0, 131.598, 135.54594, 139.6123182],
    PaperTDFStrategy: [0.0, 0.0, 0.0, 123.80985, 127.5241455, 131.349869865],
}

# Missing aggregates preserve both equity sleeves; Bonds and Bills still grow.
NAN_AGGREGATE_PATHS = {
    (PaperOptimalStrategy, "annual"): [0.0, 1000.0, 2000.0, 1869.2, 1737.525056, 1604.587474819],
    (PaperOptimalStrategy, "monthly"): [0.0, 1000.0, 2000.0, 1869.480573723, 1738.037216716, 1605.279472750],
    (PaperTDFStrategy, "annual"): [0.0, 1000.0, 2026.62, 1960.146864, 1888.217742701, 1810.574795822],
    (PaperTDFStrategy, "monthly"): [0.0, 1000.0, 2026.62, 1957.946024587, 1883.717680770, 1803.674038931],
}
NAN_AGGREGATE_WITHDRAWALS = {
    PaperOptimalStrategy: [0.0, 0.0, 0.0, 120.0, 123.6, 127.308],
    PaperTDFStrategy: [0.0, 0.0, 0.0, 121.5972, 125.245116, 129.00246948],
}


EXPLICIT_OVERLAP_CASES = [
    pytest.param(PaperOptimalStrategy, {"dom_label": "USA", "intl_assets": {"USA": 1.0}},
                 {"USA": 1.0}, {"USA": 0.73, "Bills": 0.27}, id="optimal-equities"),
    pytest.param(PaperOptimalStrategy, {"dom_label": "USA", "intl_assets": {"USA": 0.25, "JPN": 0.75}},
                 {"USA": 0.4975, "JPN": 0.5025}, {"USA": 0.3775, "JPN": 0.3525, "Bills": 0.27},
                 id="optimal-partial-equities"),
    pytest.param(PaperOptimalStrategy, {"intl_assets": {DOMESTIC_STOCK: 1.0}},
                 {DOMESTIC_STOCK: 1.0}, {DOMESTIC_STOCK: 0.73, "Bills": 0.27},
                 id="optimal-default-domestic"),
    pytest.param(PaperOptimalStrategy, {"dom_label": INTERNATIONAL_STOCK},
                 {INTERNATIONAL_STOCK: 1.0}, {INTERNATIONAL_STOCK: 0.73, "Bills": 0.27},
                 id="optimal-default-international"),
    pytest.param(PaperOptimalStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0}, "bills_label": "USA"},
                 {"USA": 0.33, "JPN": 0.67}, {"USA": 0.53, "JPN": 0.47}, id="optimal-bills-domestic"),
    pytest.param(PaperOptimalStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0}, "bills_label": "JPN"},
                 {"USA": 0.33, "JPN": 0.67}, {"USA": 0.26, "JPN": 0.74}, id="optimal-bills-international"),
    pytest.param(PaperOptimalStrategy, {"dom_label": "USA", "intl_assets": {"USA": 1.0}, "bills_label": "USA"},
                 {"USA": 1.0}, {"USA": 1.0}, id="optimal-all-components"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"USA": 1.0}},
                 {"USA": 0.90, "Bonds": 0.10, "Bills": 0.0},
                 {"USA": 0.17, "Bonds": 0.73, "Bills": 0.10}, id="tdf-equities"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"USA": 0.25, "JPN": 0.75}},
                 {"USA": 0.63, "JPN": 0.27, "Bonds": 0.10, "Bills": 0.0},
                 {"USA": 0.1175, "JPN": 0.0525, "Bonds": 0.73, "Bills": 0.10}, id="tdf-partial-equities"),
    pytest.param(PaperTDFStrategy, {"intl_assets": {DOMESTIC_STOCK: 1.0}},
                 {DOMESTIC_STOCK: 0.90, "Bonds": 0.10, "Bills": 0.0},
                 {DOMESTIC_STOCK: 0.17, "Bonds": 0.73, "Bills": 0.10}, id="tdf-default-domestic"),
    pytest.param(PaperTDFStrategy, {"dom_label": INTERNATIONAL_STOCK},
                 {INTERNATIONAL_STOCK: 0.90, "Bonds": 0.10, "Bills": 0.0},
                 {INTERNATIONAL_STOCK: 0.17, "Bonds": 0.73, "Bills": 0.10}, id="tdf-default-international"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0}, "bills_label": "USA"},
                 {"USA": 0.54, "JPN": 0.36, "Bonds": 0.10},
                 {"USA": 0.20, "JPN": 0.07, "Bonds": 0.73}, id="tdf-bills-domestic"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0}, "bills_label": "JPN"},
                 {"USA": 0.54, "JPN": 0.36, "Bonds": 0.10},
                 {"USA": 0.10, "JPN": 0.17, "Bonds": 0.73}, id="tdf-bills-international"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0}, "bond_label": "USA"},
                 {"USA": 0.64, "JPN": 0.36, "Bills": 0.0},
                 {"USA": 0.83, "JPN": 0.07, "Bills": 0.10}, id="tdf-bonds-domestic"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0}, "bond_label": "JPN"},
                 {"USA": 0.54, "JPN": 0.46, "Bills": 0.0},
                 {"USA": 0.10, "JPN": 0.80, "Bills": 0.10}, id="tdf-bonds-international"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"JPN": 1.0},
                                   "bond_label": "Cash", "bills_label": "Cash"},
                 {"USA": 0.54, "JPN": 0.36, "Cash": 0.10},
                 {"USA": 0.10, "JPN": 0.07, "Cash": 0.83}, id="tdf-bonds-bills"),
    pytest.param(PaperTDFStrategy, {"dom_label": "USA", "intl_assets": {"USA": 1.0},
                                   "bond_label": "USA", "bills_label": "USA"},
                 {"USA": 1.0}, {"USA": 1.0}, id="tdf-all-components"),
]


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
    expected_path = NAN_COUNTRY_PATHS[strategy_type, granularity]
    assert actual.paths[0] == pytest.approx(expected_path)
    assert actual.terminal_wealths == pytest.approx([expected_path[-1]])
    assert actual.withdrawal_paths[0] == pytest.approx(NAN_COUNTRY_WITHDRAWALS[strategy_type])


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
    if pd.isna(aggregate_returns[0]):
        expected_path = NAN_AGGREGATE_PATHS[strategy_type, granularity]
        assert actual.paths[0] == pytest.approx(expected_path)
        assert actual.terminal_wealths == pytest.approx([expected_path[-1]])
        assert actual.withdrawal_paths[0] == pytest.approx(NAN_AGGREGATE_WITHDRAWALS[strategy_type])


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


@pytest.mark.parametrize("strategy_type, mapping, young_allocation, retirement_allocation", EXPLICIT_OVERLAP_CASES)
def test_explicit_component_overlaps_add_their_raw_weights(
    strategy_type, mapping, young_allocation, retirement_allocation
):
    strategy = strategy_type(**mapping)
    assert strategy.get_allocation(25) == pytest.approx(young_allocation)
    assert strategy.get_allocation(65) == pytest.approx(retirement_allocation)
    for age in [25, 63, 64, 65, 66, 67, 70]:
        allocation = strategy.get_allocation(age)
        assert sum(allocation.values()) == pytest.approx(1.0)
        assert all(weight >= 0.0 for weight in allocation.values())


@pytest.mark.parametrize("strategy_type, mapping, young_allocation, retirement_allocation", EXPLICIT_OVERLAP_CASES)
@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_explicit_component_overlaps_conserve_zero_return_wealth(
    tmp_path, country_returns, strategy_type, mapping, young_allocation,
    retirement_allocation, engine_name, granularity
):
    returns = {asset: 0.0 for asset in country_returns}
    returns.update({DOMESTIC_STOCK: 0.0, INTERNATIONAL_STOCK: 0.0, "Cash": 0.0})
    config = lifecycle_config(returns, engine_name, granularity)
    result = simulate(tmp_path, returns, engine_name, config, strategy_type(**mapping))

    # Two 1000 contributions fund three fixed 120 withdrawals. Zero returns
    # leave all other wealth intact, including during monthly rebalancing.
    assert result.paths[0] == pytest.approx([0.0, 1000.0, 2000.0, 1880.0, 1760.0, 1640.0])
    assert result.withdrawal_paths[0] == pytest.approx([0.0, 0.0, 0.0, 120.0, 120.0, 120.0])
    assert result.terminal_wealths == pytest.approx([1640.0])


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
