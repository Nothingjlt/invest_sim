import pandas as pd
import numpy as np
import os
from src.data_loader import CountryMetadataRegistry


def generate_global_returns(num_years=100, seed=42):
    np.random.seed(seed)

    # Define country lists
    developed_countries = [
        "USA",
        "GBR",
        "FRA",
        "DEU",
        "JPN",
        "CAN",
        "AUS",
        "AUT",
        "BEL",
        "DNK",
        "FIN",
        "GRC",
        "ISL",
        "IRL",
        "ITA",
        "LUX",
        "NLD",
        "NZL",
        "NOR",
        "PRT",
        "ESP",
        "SWE",
        "CHE",
        "TUR",
        "ARG_HIST",
        "CHL_HIST",
        "CSK_HIST",
        "ISR",
        "SGP",
        "CZE",
        "EST",
        "HUN",
        "KOR",
        "LVA",
        "LTU",
        "MEX",
        "POL",
        "SVK",
        "SVN",
    ]

    emerging_countries = [
        "BRA",
        "CHL",
        "CHN",
        "COL",
        "CZE_EM",
        "EGY",
        "GRC_EM",
        "HUN_EM",
        "IND",
        "IDN",
        "KOR_EM",
        "KWT",
        "MYS",
        "MEX_EM",
        "PER",
        "PHL",
        "POL_EM",
        "QAT",
        "SAU",
        "ZAF",
        "TWN",
        "THA",
        "TUR_EM",
        "ARE",
    ]

    all_countries = developed_countries + emerging_countries
    num_countries = len(all_countries)

    # 1. Create Correlation Matrix
    corr_matrix = np.eye(num_countries)
    for i in range(num_countries):
        for j in range(num_countries):
            if i == j:
                continue

            name_i = all_countries[i]
            name_j = all_countries[j]

            is_i_dev = name_i in developed_countries
            is_j_dev = name_j in developed_countries

            if is_i_dev and is_j_dev:
                corr_matrix[i, j] = 0.8
            elif not is_i_dev and not is_j_dev:
                corr_matrix[i, j] = 0.6
            else:
                corr_matrix[i, j] = 0.4

    # 2. Define Means and Volatilities calibrated to paper's Table II stats (monthly annualized)
    # Monthly stats from paper:
    # USA: mean=0.52%, std=5.01% -> annual mean = 6.24%, std = 17.36%
    # GBR: mean=0.41%, std=4.25% -> annual mean = 4.92%, std = 14.72%
    # FRA: mean=0.26%, std=5.40% -> annual mean = 3.12%, std = 18.71%
    # DEU: mean=0.25%, std=8.29% -> annual mean = 3.00%, std = 28.72%
    # JPN: mean=0.31%, std=6.59% -> annual mean = 3.72%, std = 22.83%
    # CAN: mean=0.48%, std=4.26% -> annual mean = 5.76%, std = 14.76%
    # AUS: mean=0.57%, std=3.96% -> annual mean = 6.84%, std = 13.72%
    calibrated_params = {
        "USA": {"mean": 0.0624, "std": 0.1736},
        "GBR": {"mean": 0.0492, "std": 0.1472},
        "FRA": {"mean": 0.0312, "std": 0.1871},
        "DEU": {"mean": 0.0300, "std": 0.2872},
        "JPN": {"mean": 0.0372, "std": 0.2283},
        "CAN": {"mean": 0.0576, "std": 0.1476},
        "AUS": {"mean": 0.0684, "std": 0.1372},
    }

    means = []
    vols = []

    for country in all_countries:
        # Check for historical variants
        base_country = country.replace("_HIST", "")
        if base_country in calibrated_params:
            means.append(calibrated_params[base_country]["mean"])
            vols.append(calibrated_params[base_country]["std"])
        elif country in developed_countries:
            # Default developed: mean ~4.5%, std ~17%
            means.append(np.random.normal(0.045, 0.01))
            vols.append(np.random.normal(0.17, 0.02))
        else:
            # Default emerging: mean ~8.0%, std ~21%
            means.append(np.random.normal(0.08, 0.02))
            vols.append(np.random.normal(0.21, 0.03))

    # Add Fixed Income (Bonds and Bills)
    all_assets = all_countries + ["Bonds", "Bills"]
    num_assets = len(all_assets)

    # Expand correlation matrix for fixed income
    full_corr = np.eye(num_assets)
    full_corr[:num_countries, :num_countries] = corr_matrix
    for i in range(num_countries):
        full_corr[i, num_assets - 2] = 0.2
        full_corr[num_assets - 2, i] = 0.2
        full_corr[i, num_assets - 1] = 0.05
        full_corr[num_assets - 1, i] = 0.05

    # Bond-Bill correlation
    full_corr[num_assets - 2, num_assets - 1] = 0.5
    full_corr[num_assets - 1, num_assets - 2] = 0.5

    # Final Means/Vols (calibrated to paper average monthly stats)
    # Bonds: mean=0.12% monthly -> annual mean = 1.44%, std = 1.76% monthly -> annual std = 6.10%
    # Bills: mean=0.06% monthly -> annual mean = 0.72%, std = 0.61% monthly -> annual std = 2.11%
    means.extend([0.0144, 0.0072])
    vols.extend([0.0610, 0.0211])

    # 3. Generate Returns using Multivariate Normal
    vol_diag = np.diag(vols)
    cov_matrix = vol_diag @ full_corr @ vol_diag

    raw_returns = np.random.multivariate_normal(means, cov_matrix, size=num_years)

    # 4. Format into DataFrame
    df = pd.DataFrame(raw_returns, columns=all_assets)
    df.insert(0, "Year", range(2023 - num_years + 1, 2024))

    # Mask returns to respect country sample start dates from paper Table C.V
    for idx, row in df.iterrows():
        year = int(row["Year"])
        for country in all_countries:
            base_country = country.replace("_HIST", "")
            # Check custom start date mapping
            if not CountryMetadataRegistry.is_valid_year_for_country(base_country, year):
                df.at[idx, country] = np.nan

    # Round to 4 decimal places
    df = df.round(4)

    # Save
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/global_historical_returns.csv", index=False)
    print(
        f"Generated {num_years} years of calibrated data for {num_assets} assets in data/global_historical_returns.csv"
    )


if __name__ == "__main__":
    generate_global_returns()

