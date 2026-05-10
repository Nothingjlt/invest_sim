import pandas as pd
import numpy as np
import os

def generate_global_returns(num_years=100, seed=42):
    np.random.seed(seed)
    
    # Define country lists
    developed_countries = [
        "USA", "GBR", "FRA", "DEU", "JPN", "CAN", "AUS",
        "AUT", "BEL", "DNK", "FIN", "GRC", "ISL", "IRL", "ITA", "LUX", "NLD", "NZL", "NOR", "PRT", "ESP", "SWE", "CHE", "TUR",
        "ARG_HIST", "CHL_HIST", "CSK_HIST", "ISR", "SGP",
        "CZE", "EST", "HUN", "KOR", "LVA", "LTU", "MEX", "POL", "SVK", "SVN"
    ]
    
    emerging_countries = [
        "BRA", "CHL", "CHN", "COL", "CZE_EM", "EGY", "GRC_EM", "HUN_EM", "IND", "IDN", "KOR_EM", "KWT", "MYS", "MEX_EM", "PER", "PHL", "POL_EM", "QAT", "SAU", "ZAF", "TWN", "THA", "TUR_EM", "ARE"
    ]
    
    all_countries = developed_countries + emerging_countries
    num_countries = len(all_countries)
    
    # 1. Create Correlation Matrix
    # We'll use a simplified block correlation structure
    # Developed: 0.8 correlation among each other
    # Emerging: 0.6 correlation among each other
    # Dev-EM: 0.4 correlation
    
    corr_matrix = np.eye(num_countries)
    for i in range(num_countries):
        for j in range(num_countries):
            if i == j: continue
            
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
    
    # 2. Define Means and Volatilities
    means = []
    vols = []
    
    for country in all_countries:
        if country in developed_countries:
            # Paper stats: ~5% return, 17% vol
            # We'll add some noise so they aren't identical
            means.append(np.random.normal(0.05, 0.01))
            vols.append(np.random.normal(0.17, 0.02))
        else:
            # EM stats: ~10% return, 21% vol
            means.append(np.random.normal(0.10, 0.02))
            vols.append(np.random.normal(0.21, 0.03))
            
    # Add Fixed Income (Bonds and Bills)
    all_assets = all_countries + ["Bonds", "Bills"]
    num_assets = len(all_assets)
    
    # Expand correlation matrix for fixed income (low correlation to stocks)
    full_corr = np.eye(num_assets)
    full_corr[:num_countries, :num_countries] = corr_matrix
    # Bonds: 0.2 correlation with stocks
    # Bills: 0.05 correlation
    for i in range(num_countries):
        full_corr[i, num_assets-2] = 0.2
        full_corr[num_assets-2, i] = 0.2
        full_corr[i, num_assets-1] = 0.05
        full_corr[num_assets-1, i] = 0.05
    
    # Bond-Bill correlation
    full_corr[num_assets-2, num_assets-1] = 0.5
    full_corr[num_assets-1, num_assets-2] = 0.5
    
    # Final Means/Vols
    means.extend([0.01, 0.00]) # Bonds: 1%, Bills: 0%
    vols.extend([0.10, 0.02])  # Bonds: 10%, Bills: 2%
    
    # 3. Generate Returns using Multivariate Normal
    # Covariance Matrix = Vol * Corr * Vol
    vol_diag = np.diag(vols)
    cov_matrix = vol_diag @ full_corr @ vol_diag
    
    raw_returns = np.random.multivariate_normal(means, cov_matrix, size=num_years)
    
    # 4. Format into DataFrame
    df = pd.DataFrame(raw_returns, columns=all_assets)
    df.insert(0, 'Year', range(2023 - num_years + 1, 2024))
    
    # Round to 4 decimal places
    df = df.round(4)
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/global_historical_returns.csv', index=False)
    print(f"Generated {num_years} years of data for {num_assets} assets in data/global_historical_returns.csv")

if __name__ == "__main__":
    generate_global_returns()
