import pytest
import pandas as pd
import numpy as np
from generate_global_data import generate_global_returns

def test_calibrated_returns_statistical_tolerance(tmp_path, monkeypatch):
    """
    Verify that the synthetic global returns generator produces outputs
    whose mean and standard deviations are within reasonable statistical tolerance
    of the paper's Table II calibrated parameters.
    """
    # The generator writes relative to cwd. Keep its output in pytest's temporary
    # directory so caller data is untouched even if a calibration assertion fails.
    monkeypatch.chdir(tmp_path)
    csv_file = tmp_path / "data" / "global_historical_returns.csv"

    # Generate a large sample size to reduce statistical variance for testing.
    generate_global_returns(num_years=1000, seed=42)

    df = pd.read_csv(csv_file)
    
    # Calibrated targets:
    # USA: mean = 6.24%, std = 17.36%
    # GBR: mean = 4.92%, std = 14.72%
    # Bonds: mean = 1.44%, std = 6.10%
    # Bills: mean = 0.72%, std = 2.11%
    
    targets = {
        "USA": {"mean": 0.0624, "std": 0.1736},
        "GBR": {"mean": 0.0492, "std": 0.1472},
        "Bonds": {"mean": 0.0144, "std": 0.0610},
        "Bills": {"mean": 0.0072, "std": 0.0211},
    }
    
    for col, target in targets.items():
        assert col in df.columns
        # Drop NaNs before calculation since some earlier years might be masked
        series = df[col].dropna()
        
        sample_mean = series.mean()
        sample_std = series.std()
        
        # Check that they are within statistical tolerance (e.g. +/- 3 standard errors)
        # Standard error of the mean = std / sqrt(N)
        se_mean = target["std"] / np.sqrt(len(series))
        
        # Mean check: within 3 standard errors
        assert sample_mean == pytest.approx(target["mean"], abs=3 * se_mean)
        # Std check: within 15% of target standard deviation
        assert sample_std == pytest.approx(target["std"], rel=0.15)

