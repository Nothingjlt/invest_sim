import pytest
import pandas as pd
from unittest.mock import Mock
from src.config import MarketConfig, SimulationConfig
from src.market import BootstrapMarket, PerspectiveBootstrapMarket
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy


def test_bootstrap_block_integrity():
    """
    Verify that the bootstrap sampler retrieves contiguous years.
    Using the global historical data.
    """
    csv_path = "data/global_historical_returns.csv"
    # Use block_size=5 to test block jumps
    market = BootstrapMarket(csv_path, block_size=5, seed=10)
    market.start_new_path()

    # 1. First block
    results = [market.get_annual_returns() for _ in range(5)]

    # Check that they were sequential
    df = pd.read_csv(csv_path)

    # Locate the starting row in the CSV based on the first USA return
    # Use approx comparison to find the row
    first_ret = results[0]["USA"]
    start_idx = df[df["USA"] == first_ret].index[0]

    for i in range(5):
        expected_idx = (start_idx + i) % len(df)
        assert results[i]["USA"] == pytest.approx(df.iloc[expected_idx]["USA"])


def test_bootstrap_wrapping():
    """Verify that if we hit the end of the CSV, it wraps to the beginning."""
    csv_path = "data/global_historical_returns.csv"
    df = pd.read_csv(csv_path)
    last_idx = len(df) - 1

    market = BootstrapMarket(csv_path, block_size=100)
    market.current_index = last_idx  # Last row
    market.remaining_in_block = 10

    r1 = market.get_annual_returns()  # Last row
    r2 = market.get_annual_returns()  # First row (wrapped)

    assert r1["USA"] == pytest.approx(df.iloc[last_idx]["USA"])
    assert r2["USA"] == pytest.approx(df.iloc[0]["USA"])


class CountingBootstrapMarket(BootstrapMarket):
    def __init__(self, csv_path, block_size=1, seed=None):
        super().__init__(csv_path, block_size, seed)
        self.start_calls = 0

    def start_new_path(self):
        self.start_calls += 1
        super().start_new_path()


def test_bootstrap_market_starts_new_path_every_trial():
    csv_path = "data/global_historical_returns.csv"
    config = SimulationConfig(
        starting_age=25,
        retirement_age=65,
        end_age=26,
        markets=[MarketConfig("USA", 0.0, 0.0, 1.0)],
    )
    sim = Simulator(config)
    market = CountingBootstrapMarket(csv_path, block_size=1, seed=42)
    strategy = FixedAllocationStrategy({"USA": 1.0})
    sim.run_stochastic(strategy, num_trials=5, market_engine=market)

    assert market.start_calls == 5


def test_stationary_bootstrap_market():
    """Verify StationaryBootstrapMarket works and has expected interface."""
    from src.market import StationaryBootstrapMarket
    csv_path = "data/global_historical_returns.csv"
    market = StationaryBootstrapMarket(csv_path, block_size=5, seed=42)
    market.start_new_path()
    
    # Draw a few returns
    rets = [market.get_annual_returns() for _ in range(20)]
    assert len(rets) == 20
    assert "USA" in rets[0]
    assert "Bonds" in rets[0]


def test_perspective_bootstrap_market(tmp_path):
    """Verify PerspectiveBootstrapMarket loads, converts, and samples correctly."""
    from src.market import PerspectiveBootstrapMarket
    
    # Create dummy JST CSV
    csv_data = """Year,Country,iso,cpi,exrat,eq_tr,bond_tr,bill_rate,gdp
1950,United States,USA,10.0,1.0,0.10,0.02,0.01,5000.0
1951,United States,USA,11.0,1.0,0.15,0.03,0.01,5500.0
1950,United Kingdom,GBR,20.0,0.5,0.08,0.04,0.02,2000.0
1951,United Kingdom,GBR,22.0,0.4,0.12,0.05,0.02,2400.0
"""
    dummy_csv = tmp_path / "dummy_jst.csv"
    dummy_csv.write_text(csv_data)
    
    # Perspective GBR (BGP / GBP) with fixed bootstrap
    market = PerspectiveBootstrapMarket(
        csv_path=str(dummy_csv),
        perspective_country="USA",
        block_size=5,
        stationary_bootstrap=False,
        seed=10
    )
    market.start_new_path()
    
    rets = market.get_annual_returns()
    assert "Domestic Stock" in rets
    assert "International Stock" in rets
    assert "Bonds" in rets
    assert "Bills" in rets


@pytest.fixture
def perspective_market_factory(tmp_path):
    def make_market(
        stationary_bootstrap, *, perspective_country="CHL", block_size=4,
        missing_return_year=None, missing_source_year=None,
        first_year=1968, last_year=2014,
    ):
        # Include source observations inside Chile's excluded period so that
        # eligibility filtering, rather than source coverage, creates the gap.
        rows = [
            {
                "year": year, "iso": country, "cpi": 100.0, "exrat": 1.0,
                "eq_tr": (
                    float("nan")
                    if country == "CHL" and year == missing_return_year
                    else (year - 1900) / 1000 + offset
                ),
                "bond_tr": (year - 1900) / 10000,
                "bill_rate": 0.01, "gdp": 100.0,
            }
            for year in range(first_year, last_year + 1)
            if year != missing_source_year
            for country, offset in (("CHL", 0.0), ("USA", 0.1))
        ]
        path = tmp_path / "bootstrap_periods.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return PerspectiveBootstrapMarket(
            str(path), perspective_country=perspective_country,
            block_size=block_size, stationary_bootstrap=stationary_bootstrap,
            weight_method="equal",
        )

    return make_market


@pytest.mark.parametrize("stationary_bootstrap", [False, True], ids=["fixed", "stationary"])
@pytest.mark.parametrize(
    "perspective_country,missing_return_year,missing_source_year,before_gap,after_gap",
    [
        ("CHL", None, None, 1970, 2011),
        ("USA", None, None, 1970, 2011),
        ("CHL", 2012, None, 2011, 2013),
        ("CHL", None, 2012, 2011, 2014),
    ],
    ids=["perspective-eligibility", "foreign-eligibility", "missing-return", "missing-source"],
)
def test_perspective_bootstrap_restarts_at_processed_calendar_gap(
    perspective_market_factory, monkeypatch, stationary_bootstrap,
    perspective_country, missing_return_year, missing_source_year, before_gap, after_gap,
):
    market = perspective_market_factory(
        stationary_bootstrap, perspective_country=perspective_country,
        missing_return_year=missing_return_year, missing_source_year=missing_source_year,
    )
    start_index = market.data.index[market.data["Year"].eq(before_gap)][0]
    assert market.data.iloc[start_index + 1]["Year"] == after_gap
    draw_start = Mock(side_effect=[start_index, 0])
    monkeypatch.setattr("src.market.random.randint", draw_start)
    # Suppress natural stationary restarts: only the calendar gap can restart.
    monkeypatch.setattr("src.market.random.random", lambda: 0.99)

    market.start_new_path()
    for expected_index in (start_index, 0, 1):
        expected = market.data.iloc[expected_index].drop("Year").to_dict()
        assert market.get_annual_returns() == pytest.approx(expected)
        if expected_index == 0 and not stationary_bootstrap:
            # The gap starts a full new fixed block rather than carrying over
            # the interrupted block's remaining length.
            assert market.remaining_in_block == market.block_size - 1
    assert draw_start.call_count == 2


@pytest.mark.parametrize("stationary_bootstrap", [False, True], ids=["fixed", "stationary"])
@pytest.mark.parametrize("start_index", [0, 1, 2, 5])
def test_perspective_bootstrap_can_start_anywhere_in_disjoint_sample(
    perspective_market_factory, monkeypatch, stationary_bootstrap, start_index,
):
    market = perspective_market_factory(stationary_bootstrap)
    assert market.data["Year"].tolist() == [1969, 1970, 2011, 2012, 2013, 2014]
    draw_start = Mock(return_value=start_index)
    monkeypatch.setattr("src.market.random.randint", draw_start)

    for _ in range(2):
        market.start_new_path()
        expected = market.data.iloc[start_index].drop("Year").to_dict()
        assert market.get_annual_returns() == pytest.approx(expected)
    assert draw_start.call_count == 2


@pytest.mark.parametrize("stationary_bootstrap", [False, True], ids=["fixed", "stationary"])
def test_perspective_bootstrap_preserves_contiguous_blocks_and_wrapping(
    perspective_market_factory, monkeypatch, stationary_bootstrap,
):
    market = perspective_market_factory(
        stationary_bootstrap, first_year=2010, last_year=2014,
    )
    assert market.data["Year"].tolist() == [2011, 2012, 2013, 2014]
    draw_start = Mock(side_effect=[2, 0])
    monkeypatch.setattr("src.market.random.randint", draw_start)
    # Four observations in the first block, then a natural stationary restart.
    draw_restart = Mock(side_effect=[0.99, 0.99, 0.99, 0.0])
    monkeypatch.setattr("src.market.random.random", draw_restart)

    market.start_new_path()
    for expected_index in (2, 3, 0, 1, 0):
        expected = market.data.iloc[expected_index].drop("Year").to_dict()
        assert market.get_annual_returns() == pytest.approx(expected)
    assert draw_start.call_count == 2
    assert draw_restart.call_count == (4 if stationary_bootstrap else 0)


def test_perspective_stationary_bootstrap_can_restart_before_fixed_block_end(
    perspective_market_factory, monkeypatch,
):
    market = perspective_market_factory(True, first_year=2010, last_year=2014)
    draw_start = Mock(side_effect=[0, 3])
    monkeypatch.setattr("src.market.random.randint", draw_start)
    monkeypatch.setattr("src.market.random.random", lambda: 0.0)

    market.start_new_path()
    for expected_index in (0, 3):
        expected = market.data.iloc[expected_index].drop("Year").to_dict()
        assert market.get_annual_returns() == pytest.approx(expected)
    assert draw_start.call_count == 2

