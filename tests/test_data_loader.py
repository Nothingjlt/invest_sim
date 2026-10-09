import pytest
import os
import pandas as pd
import numpy as np
from src.data_loader import JSTDataLoader, CountryMetadataRegistry


def test_missing_jst_input_fails_closed_without_generating_data(tmp_path):
    csv_file = tmp_path / "jst_dataset.csv"

    with pytest.raises(FileNotFoundError, match="JST dataset CSV not found"):
        JSTDataLoader(str(csv_file))

    assert not csv_file.exists()

def test_country_metadata_registry():
    assert CountryMetadataRegistry.get_sample_period("USA") == (1890, 2023)
    assert CountryMetadataRegistry.get_sample_period("GBR") == (1890, 2023)
    assert CountryMetadataRegistry.is_valid_year_for_country("USA", 1950) is True
    assert CountryMetadataRegistry.is_valid_year_for_country("ARG", 1930) is False
    assert CountryMetadataRegistry.is_valid_year_for_country("ARG", 1950) is True


def test_country_metadata_disjoint_periods_and_legacy_bounds():
    assert CountryMetadataRegistry.get_sample_periods("chl") == (
        (1927, 1970), (2010, 2023)
    )
    # The compatibility accessor is a bounding envelope, not eligibility.
    assert CountryMetadataRegistry.get_sample_period("CHL") == (1927, 2023)
    eligible_years = {
        year for year in range(1926, 2025)
        if CountryMetadataRegistry.is_valid_year_for_country("chl", year)
    }
    assert eligible_years == set(range(1927, 1971)) | set(range(2010, 2024))


@pytest.mark.parametrize(
    "country,period",
    [("USA", (1890, 2023)), ("ARG", (1947, 1966)), ("CSK", (1922, 1945))],
)
def test_country_metadata_continuous_period_boundaries(country, period):
    start, end = period
    assert CountryMetadataRegistry.get_sample_periods(country) == (period,)
    assert CountryMetadataRegistry.get_sample_period(country.lower()) == period
    assert not CountryMetadataRegistry.is_valid_year_for_country(country, start - 1)
    assert CountryMetadataRegistry.is_valid_year_for_country(country, start)
    assert CountryMetadataRegistry.is_valid_year_for_country(country, end)
    assert not CountryMetadataRegistry.is_valid_year_for_country(country, end + 1)


def test_country_metadata_unknown_country():
    assert CountryMetadataRegistry.get_sample_periods("ZZZ") == ()
    assert CountryMetadataRegistry.get_sample_period("ZZZ") == (None, None)
    assert not CountryMetadataRegistry.is_valid_year_for_country("ZZZ", 1950)


@pytest.fixture
def reentry_loader(tmp_path):
    # Complete observations even in Chile's unavailable years prove that sample
    # eligibility, rather than missing source values, excludes the gap.
    rows = [
        {
            "year": year,
            "iso": country,
            "cpi": 100.0,
            "exrat": 1.0,
            "eq_tr": equity_return,
            "bond_tr": 0.02,
            "bill_rate": 0.01,
            "gdp": gdp,
        }
        for year in range(1926, 2025)
        for country, equity_return, gdp in (
            ("USA", 0.2, 200.0), ("GBR", 0.1, 100.0), ("CHL", 0.9, 300.0)
        )
    ]
    path = tmp_path / "reentry_jst.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return JSTDataLoader(str(path))


@pytest.mark.parametrize("weight_method,in_sample_return", [("equal", 0.5), ("gdp", 0.7)])
def test_foreign_reentry_gap_excluded_from_international_returns(
    reentry_loader, weight_method, in_sample_return
):
    processed = reentry_loader.get_processed_returns(
        perspective_country="USA", weight_method=weight_method
    ).set_index("Year")
    assert set(processed.index) == set(range(1927, 2024))
    # Entry/re-entry years need an eligible lag: 1927 and 2010 are excluded
    # from the foreign basket even though their source observations exist.
    chile_return_years = set(range(1928, 1971)) | set(range(2011, 2024))
    for year, row in processed.iterrows():
        expected = in_sample_return if year in chile_return_years else 0.1
        assert row["International Stock"] == pytest.approx(expected)

    # Removing unavailable observations must not change returns or coverage.
    unavailable = (
        reentry_loader.raw_data["iso"].eq("CHL")
        & reentry_loader.raw_data["year"].between(1971, 2009)
    )
    reentry_loader.raw_data = reentry_loader.raw_data.loc[~unavailable].copy()
    without_gap_observations = reentry_loader.get_processed_returns(
        perspective_country="USA", weight_method=weight_method
    ).set_index("Year")
    pd.testing.assert_frame_equal(processed, without_gap_observations)


def test_perspective_reentry_requires_eligible_current_and_previous_year(reentry_loader):
    processed = reentry_loader.get_processed_returns(
        perspective_country="chl", weight_method="equal"
    ).set_index("Year")
    assert set(processed.index) == set(range(1928, 1971)) | set(range(2011, 2024))
    assert processed["Domestic Stock"].to_numpy() == pytest.approx(0.9)
    assert processed["International Stock"].to_numpy() == pytest.approx(0.15)


def test_reentry_international_coverage_and_true_missing_observation(reentry_loader):
    # Chile is the only foreign market. Without an eligible observed return,
    # the legacy loader must omit the incomplete vector, never fill with zero.
    reentry_loader.raw_data = reentry_loader.raw_data.loc[
        reentry_loader.raw_data["iso"].ne("GBR")
    ].copy()
    complete = reentry_loader.get_processed_returns(weight_method="equal").set_index("Year")
    eligible_years = set(range(1928, 1971)) | set(range(2011, 2024))
    assert set(complete.index) == eligible_years
    assert complete["International Stock"].to_numpy() == pytest.approx(0.9)

    missing = (
        reentry_loader.raw_data["iso"].eq("CHL")
        & reentry_loader.raw_data["year"].eq(2012)
    )
    reentry_loader.raw_data.loc[missing, "eq_tr"] = np.nan
    assert CountryMetadataRegistry.is_valid_year_for_country("CHL", 2012)
    incomplete = reentry_loader.get_processed_returns(weight_method="equal").set_index("Year")
    assert set(incomplete.index) == eligible_years - {2012}
    assert incomplete.loc[2013, "International Stock"] == pytest.approx(0.9)


def test_perspective_continuous_start_requires_eligible_lag(reentry_loader):
    # Shift the complete fixture to include USA's first sample year and its lag.
    reentry_loader.raw_data["year"] -= 37
    processed = reentry_loader.get_processed_returns(weight_method="equal")
    assert processed["Year"].min() == 1891
    assert 1890 not in set(processed["Year"])


def test_unknown_perspective_retains_observed_panel_fallback(reentry_loader):
    reentry_loader.raw_data.loc[reentry_loader.raw_data["iso"].eq("CHL"), "iso"] = "ZZZ"
    processed = reentry_loader.get_processed_returns(
        perspective_country="zzz", weight_method="equal"
    ).set_index("Year")
    assert set(processed.index) == set(range(1927, 2024))
    assert processed.loc[1980, "Domestic Stock"] == pytest.approx(0.9)

def test_jst_data_loader_processing(tmp_path):
    # Create a small synthetic JST dataset CSV
    csv_data = """Year,Country,iso,cpi,exrat,eq_tr,bond_tr,bill_rate,gdp
1950,United States,USA,10.0,1.0,0.10,0.02,0.01,5000.0
1951,United States,USA,11.0,1.0,0.15,0.03,0.01,5500.0
1950,United Kingdom,GBR,20.0,0.5,0.08,0.04,0.02,2000.0
1951,United Kingdom,GBR,22.0,0.4,0.12,0.05,0.02,2400.0
"""
    csv_file = tmp_path / "dummy_jst.csv"
    csv_file.write_text(csv_data)

    loader = JSTDataLoader(str(csv_file))
    
    # Process from the perspective of USA (USD)
    processed = loader.get_processed_returns(perspective_country="USA", weight_method="equal")
    
    assert len(processed) == 1
    row = processed.iloc[0]
    assert row["Year"] == 1951
    
    # Check domestic real returns for USA (1951)
    # CPI ratio of USA: 11.0 / 10.0 = 1.1
    # USA real equity return = (1 + 0.15) / 1.1 - 1 = 1.15 / 1.1 - 1 = 0.04545
    assert row["Domestic Stock"] == pytest.approx(1.15 / 1.1 - 1)
    assert row["Bonds"] == pytest.approx(1.03 / 1.1 - 1)
    assert row["Bills"] == pytest.approx(1.01 / 1.1 - 1)

    # Check GBR real equity return in USA perspective
    # Formula for GBR nominal in USD: (1 + eq_tr_GBR_t) * (exrat_GBR_prev / exrat_GBR_t) * (exrat_USA_t / exrat_USA_prev) - 1
    # = (1 + 0.12) * (0.5 / 0.4) * (1.0 / 1.0) - 1 = 1.12 * 1.25 - 1 = 1.4 - 1 = 0.40
    # GBR real return in USD deflated by USA inflation:
    # = (1 + 0.40) / 1.1 - 1 = 1.4 / 1.1 - 1 = 0.2727
    assert row["International Stock"] == pytest.approx(1.4 / 1.1 - 1)


def test_jst_data_loader_gdp_weighting(tmp_path):
    # Test GDP weighting of international returns
    # GBR and FRA as international countries
    csv_data = """Year,Country,iso,cpi,exrat,eq_tr,bond_tr,bill_rate,gdp
1950,United States,USA,10.0,1.0,0.10,0.02,0.01,5000.0
1951,United States,USA,11.0,1.0,0.15,0.03,0.01,5500.0
1950,United Kingdom,GBR,20.0,0.5,0.08,0.04,0.02,2000.0
1951,United Kingdom,GBR,22.0,0.5,0.12,0.05,0.02,2500.0
1950,France,FRA,15.0,2.0,0.05,0.03,0.01,3000.0
1951,France,FRA,16.0,2.0,0.10,0.04,0.01,3200.0
"""
    csv_file = tmp_path / "dummy_jst.csv"
    csv_file.write_text(csv_data)

    loader = JSTDataLoader(str(csv_file))
    
    # Process GDP weighted
    processed = loader.get_processed_returns(perspective_country="USA", weight_method="gdp")
    row = processed.iloc[0]

    # USA CPI ratio: 1.1
    # GBR return in USD (no FX change since exrat constant 0.5): 1.12 - 1 = 0.12
    # GBR weight (GDP in USD) = 2500.0 / 0.5 = 5000.0
    # FRA return in USD (no FX change since exrat constant 2.0): 1.10 - 1 = 0.10
    # FRA weight (GDP in USD) = 3200.0 / 2.0 = 1600.0
    
    # Weighted avg intl nominal return = (0.12 * 5000 + 0.10 * 1600) / (5000 + 1600)
    # = (600 + 160) / 6600 = 760 / 6600 = 0.11515
    # Weighted avg intl real return = 1.11515 / 1.1 - 1 = 0.01377
    expected_nominal = (0.12 * 5000.0 + 0.10 * 1600.0) / (5000.0 + 1600.0)
    expected_real = (1 + expected_nominal) / 1.1 - 1
    
    assert row["International Stock"] == pytest.approx(expected_real)
