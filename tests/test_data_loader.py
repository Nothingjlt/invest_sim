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
