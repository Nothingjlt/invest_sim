import math

import pandas as pd
import pytest

from src.config import MarketConfig, SimulationConfig
from src.market import (
    BootstrapMarket,
    Market,
    PerspectiveBootstrapMarket,
    StationaryBootstrapMarket,
)
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy


def write_bootstrap_panel(tmp_path, *, usa=0.10):
    path = tmp_path / "returns.csv"
    pd.DataFrame([{"Year": 2000, "USA": usa, "Bonds": 0.02}]).to_csv(
        path, index=False
    )
    return path


def write_jst_panel(tmp_path, *, usa_equity=0.10):
    path = tmp_path / "jst.csv"
    rows = []
    for year in (1950, 1951):
        rows.extend([
            {
                "Year": year,
                "Country": "United States",
                "iso": "USA",
                "cpi": 10.0 if year == 1950 else 11.0,
                "exrat": 1.0,
                "eq_tr": usa_equity,
                "bond_tr": 0.02,
                "bill_rate": 0.01,
                "gdp": 5000.0,
            },
            {
                "Year": year,
                "Country": "United Kingdom",
                "iso": "GBR",
                "cpi": 20.0 if year == 1950 else 22.0,
                "exrat": 0.5,
                "eq_tr": 0.08,
                "bond_tr": 0.04,
                "bill_rate": 0.02,
                "gdp": 2000.0,
            },
        ])
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("engine_type", [BootstrapMarket, StationaryBootstrapMarket])
@pytest.mark.parametrize("block_size", [0, -1, 1.5, True, "10"])
def test_legacy_bootstrap_requires_positive_integer_block_size(
    tmp_path, engine_type, block_size
):
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        engine_type(write_bootstrap_panel(tmp_path), block_size=block_size)


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True, "10"])
def test_perspective_bootstrap_requires_positive_integer_block_size(tmp_path, block_size):
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        PerspectiveBootstrapMarket(write_jst_panel(tmp_path), block_size=block_size)


@pytest.mark.parametrize("engine_type", [BootstrapMarket, StationaryBootstrapMarket])
def test_legacy_bootstrap_rejects_empty_raw_panel(tmp_path, engine_type):
    path = tmp_path / "empty.csv"
    path.write_text("Year,USA\n")

    with pytest.raises(ValueError, match="return panel must not be empty"):
        engine_type(path)


def test_perspective_bootstrap_rejects_empty_raw_panel(tmp_path):
    path = tmp_path / "empty_jst.csv"
    path.write_text("Year,iso,cpi,exrat,eq_tr,bond_tr,bill_rate,gdp\n")

    with pytest.raises(ValueError, match="raw return panel must not be empty"):
        PerspectiveBootstrapMarket(path)


def test_perspective_bootstrap_rejects_empty_processed_panel(tmp_path):
    path = tmp_path / "one_year_jst.csv"
    path.write_text(
        "Year,iso,cpi,exrat,eq_tr,bond_tr,bill_rate,gdp\n"
        "1950,USA,10,1,0.1,0.02,0.01,5000\n"
    )

    with pytest.raises(ValueError, match="processed return panel must not be empty"):
        PerspectiveBootstrapMarket(path)


@pytest.mark.parametrize("engine_type", [BootstrapMarket, StationaryBootstrapMarket])
@pytest.mark.parametrize("bad_value", ["not-a-number", math.inf, -math.inf, -1.01])
def test_legacy_bootstrap_rejects_malformed_returns_before_sampling(
    tmp_path, engine_type, bad_value
):
    with pytest.raises(ValueError, match="Bootstrap return panel"):
        engine_type(write_bootstrap_panel(tmp_path, usa=bad_value))


@pytest.mark.parametrize("bad_value", ["not-a-number", math.inf, -math.inf, -1.01])
def test_perspective_bootstrap_rejects_malformed_source_returns(tmp_path, bad_value):
    with pytest.raises(ValueError, match="Perspective"):
        PerspectiveBootstrapMarket(write_jst_panel(tmp_path, usa_equity=bad_value))


class SequenceMarket(Market):
    def __init__(self, observations):
        self.observations = [dict(observation) for observation in observations]
        self.index = 0

    def start_new_path(self):
        self.index = 0

    def get_annual_returns(self):
        observation = self.observations[self.index]
        self.index += 1
        return dict(observation)


def simulator_config(granularity):
    return SimulationConfig(
        starting_age=25,
        retirement_age=26,
        end_age=27,
        initial_salary=1000.0,
        salary_growth_rate=0.0,
        savings_rate=1.0,
        withdrawal_rate=0.0,
        withdrawal_inflation_adjusted=True,
        decumulation_granularity=granularity,
        markets=[MarketConfig("Stocks", 0.0, 0.0, 1.0)],
    )


@pytest.mark.parametrize("granularity", ["annual", "monthly"])
@pytest.mark.parametrize("bad_value", [None, "not-a-number", math.nan, math.inf, -math.inf, -1.01])
def test_custom_market_returns_fail_consistently_in_annual_and_monthly_paths(
    granularity, bad_value
):
    market = SequenceMarket([
        {"Stocks": 0.0, "Inflation": 0.0},
        {"Stocks": bad_value, "Inflation": 0.0},
    ])

    with pytest.raises(ValueError, match="SequenceMarket returns"):
        Simulator(simulator_config(granularity)).run_stochastic(
            FixedAllocationStrategy({"Stocks": 1.0}),
            num_trials=1,
            market_engine=market,
        )


@pytest.mark.parametrize("granularity", ["annual", "monthly"])
@pytest.mark.parametrize("bad_inflation", ["not-a-number", math.nan, math.inf, -math.inf, -1.01])
def test_invalid_custom_inflation_fails_before_annual_or_monthly_arithmetic(
    granularity, bad_inflation
):
    market = SequenceMarket([
        {"Stocks": 0.0, "Inflation": 0.0},
        {"Stocks": 0.0, "Inflation": bad_inflation},
    ])

    with pytest.raises(ValueError, match="Inflation"):
        Simulator(simulator_config(granularity)).run_stochastic(
            FixedAllocationStrategy({"Stocks": 1.0}),
            num_trials=1,
            market_engine=market,
        )


@pytest.mark.parametrize("inflation_observation", [{"Stocks": 0.0}, {"Stocks": 0.0, "Inflation": None}])
@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_missing_and_none_inflation_use_the_same_zero_fallback(
    granularity, inflation_observation
):
    market = SequenceMarket([
        {"Stocks": 0.0},
        inflation_observation,
    ])

    result = Simulator(simulator_config(granularity)).run_stochastic(
        FixedAllocationStrategy({"Stocks": 1.0}),
        num_trials=1,
        market_engine=market,
        track_paths=True,
    )

    assert result.withdrawal_paths[0][-1] == pytest.approx(0.0)


@pytest.mark.parametrize("granularity", ["annual", "monthly"])
def test_exact_minus_one_return_remains_a_valid_total_loss(granularity):
    market = SequenceMarket([
        {"Stocks": 0.0},
        {"Stocks": -1.0},
    ])

    result = Simulator(simulator_config(granularity)).run_stochastic(
        FixedAllocationStrategy({"Stocks": 1.0}),
        num_trials=1,
        market_engine=market,
    )

    assert result == pytest.approx([0.0])
