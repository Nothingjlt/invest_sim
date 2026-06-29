import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

class CountryMetadataRegistry:
    # ISO country codes mapped to (start_year, end_year) from the paper Table C.V
    COUNTRY_SAMPLE_PERIODS = {
        "GBR": (1890, 2023),
        "NLD": (1914, 2023),
        "BEL": (1897, 2023),
        "FRA": (1890, 2023),
        "NOR": (1914, 2023),
        "DEU": (1890, 2023),
        "DNK": (1890, 2023),
        "CHE": (1914, 2023),
        "USA": (1890, 2023),
        "CAN": (1891, 2023),
        "ARG": (1947, 1966),
        "NZL": (1896, 2023),
        "AUS": (1901, 2023),
        "SWE": (1910, 2023),
        "AUT": (1920, 2023),
        "CHL": (1927, 2023),
        "GRC": (1981, 2023),
        "CSK": (1922, 1945),
        "JPN": (1930, 2023),
        "PRT": (1934, 2023),
        "ITA": (1931, 2023),
        "IRL": (1936, 2023),
        "SGP": (1998, 2023),
        "ISL": (2002, 2023),
        "LUX": (1982, 2023),
        "TUR": (2010, 2023),
        "ESP": (1959, 2023),
        "FIN": (1969, 2023),
        "MEX": (2001, 2023),
        "CZE": (2000, 2023),
        "HUN": (1999, 2023),
        "POL": (1999, 2023),
        "KOR": (2000, 2023),
        "SVK": (2000, 2023),
        "ISR": (2010, 2023),
        "SVN": (2010, 2023),
        "LVA": (2016, 2023),
        "LTU": (2018, 2023),
        "COL": (2020, 2023),
    }

    @classmethod
    def get_sample_period(cls, country_iso: str) -> Tuple[Optional[int], Optional[int]]:
        return cls.COUNTRY_SAMPLE_PERIODS.get(country_iso.upper(), (None, None))

    @classmethod
    def is_valid_year_for_country(cls, country_iso: str, year: int) -> bool:
        start, end = cls.get_sample_period(country_iso)
        if start is None or end is None:
            return False
        return start <= year <= end


class JSTDataLoader:
    """Loads and processes the Jordà-Schularick-Taylor Macrohistory Database."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            if "jst_dataset.csv" in csv_path:
                self._generate_dummy_jst_file(csv_path)
            else:
                raise FileNotFoundError(f"JST dataset CSV not found at {csv_path}")
        self.raw_data = pd.read_csv(csv_path)
        self._normalize_columns()

    def _generate_dummy_jst_file(self, path: str):
        """Generates a mock raw JST dataset to allow immediate simulation run."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        countries_iso = ["USA", "GBR", "FRA", "DEU", "JPN", "CAN", "AUS", "ITA", "NLD", "BEL", "ESP", "SWE", "CHE", "DNK", "NOR", "FIN", "PRT", "IRL"]
        rows = []
        np.random.seed(42)
        for country in countries_iso:
            cpi = 1.0
            exrat = 1.0 if country == "USA" else np.random.uniform(0.5, 2.0)
            gdp = 1000.0
            for year in range(1890, 2024):
                cpi *= np.random.normal(1.025, 0.01)  # 2.5% inflation
                exrat *= np.random.normal(1.0, 0.02)  # random walk exchange rate
                if country == "USA":
                    exrat = 1.0
                gdp *= np.random.normal(1.03, 0.02)  # growth
                eq_tr = np.random.normal(0.07, 0.17)
                bond_tr = np.random.normal(0.03, 0.08)
                bill_rate = np.random.normal(0.025, 0.03)
                rows.append({
                    "year": year,
                    "country": country,
                    "iso": country,
                    "cpi": cpi,
                    "exrat": exrat,
                    "eq_tr": eq_tr,
                    "bond_tr": bond_tr,
                    "bill_rate": bill_rate,
                    "gdp": gdp
                })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)


    def _normalize_columns(self):
        """Ensure standardized case-insensitive column names and country ISO values."""
        # Convert columns to lowercase for case-insensitivity
        self.raw_data.columns = [col.lower() for col in self.raw_data.columns]
        
        # Handle JST-specific exchange rate column alias
        if "xrusd" in self.raw_data.columns and "exrat" not in self.raw_data.columns:
            self.raw_data.rename(columns={"xrusd": "exrat"}, inplace=True)
            
        # Standardize crucial columns
        col_mappings = {
            "year": "year",
            "country": "country",
            "iso": "iso",
            "cpi": "cpi",
            "exrat": "exrat",
            "eq_tr": "eq_tr",
            "bond_tr": "bond_tr",
            "bill_rate": "bill_rate",
            "gdp": "gdp"
        }
        
        for std_col in col_mappings.values():
            if std_col not in self.raw_data.columns:
                # Try to find a match
                matches = [c for c in self.raw_data.columns if std_col in c]
                if matches:
                    self.raw_data.rename(columns={matches[0]: std_col}, inplace=True)

        if "iso" in self.raw_data.columns:
            self.raw_data["iso"] = self.raw_data["iso"].astype(str).str.upper()
        if "year" in self.raw_data.columns:
            self.raw_data["year"] = pd.to_numeric(self.raw_data["year"], errors="coerce")
            self.raw_data.dropna(subset=["year"], inplace=True)
            self.raw_data["year"] = self.raw_data["year"].astype(int)

    def get_processed_returns(
        self,
        perspective_country: str = "USA",
        weight_method: str = "gdp"
    ) -> pd.DataFrame:
        """
        Processes JST data into a panel of real returns for:
          Domestic Stock, International Stock, Bonds, and Bills.
        
        Parameters:
          perspective_country: ISO code of the domestic investor's country (e.g. 'USA')
          weight_method: 'gdp' for GDP-weighting international returns, or 'equal' for equal-weighting.
        
        Returns:
          A DataFrame with columns: ['Year', 'Domestic Stock', 'International Stock', 'Bonds', 'Bills']
        """
        p_iso = perspective_country.upper()
        df = self.raw_data.copy()

        # Filter database to years of interest
        start_year, end_year = CountryMetadataRegistry.get_sample_period(p_iso)
        if start_year is None:
            # Fallback to JST overall span
            start_year, end_year = int(df["year"].min()), int(df["year"].max())

        # Pivot data by year and ISO to clean it up and compute lags easily
        pivoted = df.pivot(index="year", columns="iso")
        
        # We need exrat, cpi, eq_tr, bond_tr, bill_rate, gdp (optional)
        cpi_df = pivoted["cpi"]
        exrat_df = pivoted["exrat"]
        
        # Fill in USA exchange rate as 1.0 if missing (it's the anchor currency)
        if "USA" in exrat_df.columns:
            exrat_df["USA"] = exrat_df["USA"].fillna(1.0)
        
        eq_tr_df = pivoted.get("eq_tr", pd.DataFrame())
        bond_tr_df = pivoted.get("bond_tr", pd.DataFrame())
        bill_rate_df = pivoted.get("bill_rate", pd.DataFrame())
        gdp_df = pivoted.get("gdp", pd.DataFrame())

        years = sorted(list(pivoted.index))
        processed_rows = []

        for t in years:
            if t < start_year or t > end_year:
                continue
            
            # We need the previous year to compute CPI and exchange rate changes
            t_prev = t - 1
            if t_prev not in cpi_df.index:
                continue

            # Check if perspective country has valid CPI and exchange rate data
            if p_iso not in cpi_df.columns or p_iso not in exrat_df.columns:
                continue
            
            cpi_p_t = cpi_df.at[t, p_iso]
            cpi_p_prev = cpi_df.at[t_prev, p_iso]
            ex_p_t = exrat_df.at[t, p_iso]
            ex_p_prev = exrat_df.at[t_prev, p_iso]

            if pd.isna(cpi_p_t) or pd.isna(cpi_p_prev) or cpi_p_prev <= 0:
                continue
            if pd.isna(ex_p_t) or pd.isna(ex_p_prev) or ex_p_prev <= 0:
                continue

            # Inflation ratio of perspective country
            cpi_ratio_p = cpi_p_t / cpi_p_prev

            # Domestic real returns
            eq_nom_p = eq_tr_df.at[t, p_iso] if p_iso in eq_tr_df.columns else np.nan
            bond_nom_p = bond_tr_df.at[t, p_iso] if p_iso in bond_tr_df.columns else np.nan
            bill_nom_p = bill_rate_df.at[t, p_iso] if p_iso in bill_rate_df.columns else np.nan

            dom_stock = (1 + eq_nom_p) / cpi_ratio_p - 1 if not pd.isna(eq_nom_p) else np.nan
            dom_bond = (1 + bond_nom_p) / cpi_ratio_p - 1 if not pd.isna(bond_nom_p) else np.nan
            dom_bill = (1 + bill_nom_p) / cpi_ratio_p - 1 if not pd.isna(bill_nom_p) else np.nan

            # Compute International Stock return
            intl_nom_returns = []
            intl_weights = []

            for c_iso in cpi_df.columns:
                if c_iso == p_iso:
                    continue
                
                # Verify country validity and paper-bound validity
                if not CountryMetadataRegistry.is_valid_year_for_country(c_iso, t):
                    continue
                if not CountryMetadataRegistry.is_valid_year_for_country(c_iso, t_prev):
                    continue

                # We need nominal returns, exchange rate for country C
                if c_iso not in eq_tr_df.columns or c_iso not in exrat_df.columns:
                    continue
                
                eq_nom_c = eq_tr_df.at[t, c_iso]
                ex_c_t = exrat_df.at[t, c_iso]
                ex_c_prev = exrat_df.at[t_prev, c_iso]

                if pd.isna(eq_nom_c) or pd.isna(ex_c_t) or pd.isna(ex_c_prev) or ex_c_t <= 0 or ex_c_prev <= 0:
                    continue

                # FX Conversion of country C nominal return to currency of country P
                # Formula: (1 + r_C_t) * (exrat_C_prev / exrat_C_t) * (exrat_P_t / exrat_P_prev) - 1
                r_c_in_p = (1 + eq_nom_c) * (ex_c_prev / ex_c_t) * (ex_p_t / ex_p_prev) - 1

                # Weight
                weight = 1.0
                if weight_method == "gdp" and c_iso in gdp_df.columns:
                    gdp_c = gdp_df.at[t, c_iso]
                    if not pd.isna(gdp_c) and gdp_c > 0:
                        # Convert GDP to USD (standard unit) using exrat
                        weight = gdp_c / ex_c_t

                intl_nom_returns.append(r_c_in_p)
                intl_weights.append(weight)

            # Calculate weighted average of international nominal returns
            if intl_nom_returns:
                total_weight = sum(intl_weights)
                if total_weight > 0:
                    avg_intl_nom = sum(r * w for r, w in zip(intl_nom_returns, intl_weights)) / total_weight
                    # Deflate by perspective country's inflation
                    intl_stock = (1 + avg_intl_nom) / cpi_ratio_p - 1
                else:
                    intl_stock = np.nan
            else:
                intl_stock = np.nan

            processed_rows.append({
                "Year": t,
                "Domestic Stock": dom_stock,
                "International Stock": intl_stock,
                "Bonds": dom_bond,
                "Bills": dom_bill
            })

        result_df = pd.DataFrame(processed_rows)
        # Drop rows where crucial elements are NaN
        result_df.dropna(subset=["Domestic Stock", "International Stock", "Bonds", "Bills"], inplace=True)
        return result_df.reset_index(drop=True)
