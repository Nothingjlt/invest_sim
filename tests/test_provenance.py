"""Data origin must survive sampling, simulation, and result export."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src.config import MarketConfig, SimulationConfig
from src.market import (
    BootstrapMarket, Market, PerspectiveBootstrapMarket,
    StationaryBootstrapMarket, SyntheticMarket,
)
from src.metrics import Metrics
from src.provenance import DataProvenance
from src.simulator import SimulationResult, Simulator, TerminalWealths
from src.strategy import FixedAllocationStrategy


BUNDLED_PANEL = Path(__file__).resolve().parents[1] / "data/global_historical_returns.csv"
CSV_ENGINES = [BootstrapMarket, StationaryBootstrapMarket, PerspectiveBootstrapMarket]


@pytest.fixture
def csv_path(tmp_path):
    # Valid both as a generic return table and as a JST-format panel. Its name
    # deliberately suggests historical data while its actual origin is a fixture.
    path = tmp_path / "jst_dataset.csv"
    path.write_text(
        "Year,iso,cpi,exrat,eq_tr,bond_tr,bill_rate,gdp\n"
        "1950,USA,10,1,0.1,0.02,0.01,5000\n"
        "1951,USA,11,1,0.15,0.03,0.01,5500\n"
        "1950,GBR,20,0.5,0.08,0.04,0.02,2000\n"
        "1951,GBR,22,0.4,0.12,0.05,0.02,2400\n"
    )
    return path


def simulate(market=None, *, track_paths=False, num_trials=2):
    config = SimulationConfig(
        starting_age=25, retirement_age=26, end_age=27,
        initial_salary=1000.0, salary_growth_rate=0.0,
        savings_rate=0.1, withdrawal_rate=0.0,
        markets=[MarketConfig("Stocks", 0.0, 0.0, 1.0)],
    )
    return Simulator(config).run_stochastic(
        FixedAllocationStrategy({"Stocks": 1.0}), num_trials=num_trials,
        market_engine=market, track_paths=track_paths,
    )


@pytest.mark.parametrize("engine_type", CSV_ENGINES)
def test_csv_filename_never_implies_historical_provenance(csv_path, engine_type):
    market = engine_type(csv_path)
    assert market.provenance.kind == "unknown"
    assert market.provenance.source == str(csv_path)
    assert market.provenance.sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert "Unknown/unverified" in market.provenance.label


@pytest.mark.parametrize("engine_type", CSV_ENGINES)
@pytest.mark.parametrize("kind", ["historical", "synthetic"])
def test_explicit_csv_provenance_is_bound_to_input_bytes(csv_path, engine_type, kind):
    declaration = DataProvenance(kind, "Declared research input, release 1")
    market = engine_type(csv_path, provenance=declaration)
    assert market.provenance.kind == kind
    assert market.provenance.source == declaration.source
    assert market.provenance.sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert declaration.sha256 is None  # Caller metadata is not mutated.


@pytest.mark.parametrize("engine_type", CSV_ENGINES)
def test_declared_digest_detects_changed_inputs(csv_path, engine_type):
    expected = DataProvenance(
        "historical", "Declared release",
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )
    assert engine_type(csv_path, provenance=expected).provenance == expected
    with csv_path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        engine_type(csv_path, provenance=expected)


@pytest.mark.parametrize("engine_type", CSV_ENGINES)
@pytest.mark.parametrize("declared", [None, DataProvenance("historical", "Claimed JST")])
def test_missing_inputs_are_not_generated(tmp_path, engine_type, declared):
    path = tmp_path / "missing" / "jst_dataset.csv"
    with pytest.raises(FileNotFoundError):
        engine_type(path, provenance=declared)
    assert not path.exists()
    assert not path.parent.exists()


@pytest.mark.parametrize("engine_type", [BootstrapMarket, StationaryBootstrapMarket])
def test_bundled_generated_data_and_renamed_copies_are_synthetic(tmp_path, engine_type):
    copied = tmp_path / "trusted_historical.csv"
    shutil.copyfile(BUNDLED_PANEL, copied)
    for path in (BUNDLED_PANEL, copied):
        market = engine_type(path)
        assert market.provenance.kind == "synthetic"
        assert "generate_global_data.py" in market.provenance.source
        with pytest.raises(ValueError, match="cannot be labeled historical"):
            engine_type(path, provenance=DataProvenance("historical", "Incorrect claim"))


def test_bundled_filename_alone_does_not_set_provenance(tmp_path):
    path = tmp_path / "global_historical_returns.csv"
    path.write_text("Year,Stocks\n2000,0.0\n")
    assert BootstrapMarket(path).provenance.kind == "unknown"


@pytest.mark.parametrize("track_paths", [False, True])
@pytest.mark.parametrize("explicit_engine", [False, True])
@pytest.mark.parametrize("num_trials", [0, 2])
def test_synthetic_result_metadata_with_and_without_paths(track_paths, explicit_engine, num_trials):
    market = SyntheticMarket([MarketConfig("Stocks", 0.0, 0.0, 1.0)]) if explicit_engine else None
    result = simulate(market, track_paths=track_paths, num_trials=num_trials)
    assert result.provenance.kind == "synthetic"
    assert "Synthetic" in result.provenance.label
    wealths = result.terminal_wealths if track_paths else result
    assert isinstance(wealths, list)
    assert wealths == [100.0] * num_trials
    assert wealths.provenance == result.provenance
    assert Metrics.mean(wealths) == (100.0 if num_trials else 0.0)
    if track_paths:
        # Existing tuple unpacking is preserved; no fourth field is added.
        terminal, paths, withdrawals = result
        assert terminal is wealths
        assert len(paths) == len(withdrawals) == num_trials


@pytest.mark.parametrize("track_paths", [False, True])
@pytest.mark.parametrize("kind", ["synthetic", "historical", "unknown"])
@pytest.mark.parametrize("engine_type", [BootstrapMarket, StationaryBootstrapMarket])
def test_csv_source_survives_simulation_and_json_export(tmp_path, track_paths, kind, engine_type):
    path = tmp_path / "returns.csv"
    path.write_text("Year,Stocks\n2000,0.0\n")
    market = engine_type(path, provenance=DataProvenance(kind, "Test fixture"))
    result = simulate(market, track_paths=track_paths)
    assert result.provenance == market.provenance
    saved = json.loads(json.dumps(result.to_dict()))
    assert saved["provenance"] == {
        "kind": kind, "source": "Test fixture",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert saved["terminal_wealths"] == [100.0, 100.0]
    if track_paths:
        assert saved["paths"] == [[0.0, 100.0, 100.0]] * 2
        assert saved["withdrawal_paths"] == [[0.0, 0.0, 0.0]] * 2
    assert kind in repr(result)


@pytest.mark.parametrize("track_paths", [False, True])
@pytest.mark.parametrize("inherit_market", [False, True])
def test_custom_engines_without_metadata_are_unverified(track_paths, inherit_market):
    class CustomMarket(Market if inherit_market else object):
        def get_annual_returns(self):
            return {"Stocks": 0.25}

        def __bool__(self):
            # Supplied engines must be used even when their truth value is false.
            return False

    result = simulate(CustomMarket(), track_paths=track_paths)
    assert result.provenance.kind == "unknown"
    assert result.to_dict()["terminal_wealths"] == [125.0, 125.0]


@pytest.mark.parametrize("track_paths", [False, True])
@pytest.mark.parametrize("stationary", [False, True])
def test_perspective_transform_preserves_declared_input_source(csv_path, track_paths, stationary):
    market = PerspectiveBootstrapMarket(
        csv_path, stationary_bootstrap=stationary,
        provenance=DataProvenance("synthetic", "Controlled JST fixture"),
    )
    config = SimulationConfig(
        starting_age=25, retirement_age=26, end_age=27,
        initial_salary=1000.0, salary_growth_rate=0.0,
        savings_rate=0.1, withdrawal_rate=0.0,
        markets=[MarketConfig("Domestic Stock", 0.0, 0.0, 1.0)],
    )
    result = Simulator(config).run_stochastic(
        FixedAllocationStrategy({"Domestic Stock": 1.0}), num_trials=1,
        market_engine=market, track_paths=track_paths,
    )
    assert result.provenance == market.provenance
    assert result.provenance.kind == "synthetic"
    assert result.provenance.sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert result.to_dict()["terminal_wealths"] == pytest.approx([100.0 * 1.15 / 1.1])


def test_custom_engine_can_declare_provenance():
    class CustomMarket(Market):
        provenance = DataProvenance("synthetic", "Controlled zero-return scenario")

        def get_annual_returns(self):
            return {"Stocks": 0.0}

    assert simulate(CustomMarket()).provenance == CustomMarket.provenance


def test_legacy_manually_constructed_results_are_unverified():
    result = SimulationResult([1.0], [[0.0, 1.0]], [[0.0, 0.0]])
    assert result.provenance.kind == "unknown"
    assert result.to_dict()["provenance"]["kind"] == "unknown"


def test_source_metadata_cannot_be_accidentally_mutated():
    result = simulate()
    with pytest.raises(FrozenInstanceError):
        result.provenance.kind = "historical"
    with pytest.raises(AttributeError):
        result.provenance = DataProvenance("historical", "A different source")


@pytest.mark.parametrize("kwargs", [
    {"kind": "real"}, {"kind": "historical"},
    {"kind": "synthetic", "source": " "}, {"sha256": "bad digest"},
])
def test_malformed_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DataProvenance(**kwargs)


def test_csv_metadata_requires_explicit_typed_declaration(csv_path):
    with pytest.raises(TypeError, match="DataProvenance"):
        BootstrapMarket(csv_path, provenance="historical")
    with pytest.raises(TypeError, match="DataProvenance"):
        TerminalWealths([1.0], provenance="historical")
