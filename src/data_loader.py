import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Mapping

from src.assets import DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS


JST_R6_GDP_SCALE_FACTORS = {
    "AUS": 1e6,
    "BEL": 1e6,
    "CAN": 1e9,
    "DNK": 1e9,
    "FIN": 1e6,
    "FRA": 1e9,
    "DEU": 1e9,
    "IRL": 1e6,
    "ITA": 1e9,
    "JPN": 1e12,
    "NLD": 1e6,
    "NOR": 1e6,
    "PRT": 1e6,
    "ESP": 1e6,
    "SWE": 1e6,
    "CHE": 1e6,
    "GBR": 1e9,
    "USA": 1e9,
}


class CountryMetadataRegistry:
    # ISO country codes mapped to inclusive sample intervals from the paper.
    # These intervals, rather than their outer bounds, define eligibility.
    COUNTRY_SAMPLE_PERIODS = {
        "GBR": ((1890, 2023),),
        "NLD": ((1914, 2023),),
        "BEL": ((1897, 2023),),
        "FRA": ((1890, 2023),),
        "NOR": ((1914, 2023),),
        "DEU": ((1890, 2023),),
        "DNK": ((1890, 2023),),
        "CHE": ((1914, 2023),),
        "USA": ((1890, 2023),),
        "CAN": ((1891, 2023),),
        "ARG": ((1947, 1966),),
        "NZL": ((1896, 2023),),
        "AUS": ((1901, 2023),),
        "SWE": ((1910, 2023),),
        "AUT": ((1920, 2023),),
        "CHL": ((1927, 1970), (2010, 2023)),
        "GRC": ((1981, 2023),),
        "CSK": ((1922, 1945),),
        "JPN": ((1930, 2023),),
        "PRT": ((1934, 2023),),
        "ITA": ((1931, 2023),),
        "IRL": ((1936, 2023),),
        "SGP": ((1998, 2023),),
        "ISL": ((2002, 2023),),
        "LUX": ((1982, 2023),),
        "TUR": ((2010, 2023),),
        "ESP": ((1959, 2023),),
        "FIN": ((1969, 2023),),
        "MEX": ((2001, 2023),),
        "CZE": ((2000, 2023),),
        "HUN": ((1999, 2023),),
        "POL": ((1999, 2023),),
        "KOR": ((2000, 2023),),
        "SVK": ((2000, 2023),),
        "ISR": ((2010, 2023),),
        "SVN": ((2010, 2023),),
        "LVA": ((2016, 2023),),
        "LTU": ((2018, 2023),),
        "COL": ((2020, 2023),),
    }

    @classmethod
    def get_sample_periods(cls, country_iso: str) -> Tuple[Tuple[int, int], ...]:
        """Return all inclusive eligible intervals, or () for an unknown ISO."""
        return cls.COUNTRY_SAMPLE_PERIODS.get(country_iso.upper(), ())

    @classmethod
    def get_sample_period(cls, country_iso: str) -> Tuple[Optional[int], Optional[int]]:
        """Return legacy outer bounds; use interval membership for eligibility."""
        periods = cls.get_sample_periods(country_iso)
        if not periods:
            return None, None
        return min(start for start, _ in periods), max(end for _, end in periods)

    @classmethod
    def is_valid_year_for_country(cls, country_iso: str, year: int) -> bool:
        return any(
            start <= year <= end
            for start, end in cls.get_sample_periods(country_iso)
        )


class JSTDataLoader:
    """Loads and processes the Jordà-Schularick-Taylor Macrohistory Database."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"JST dataset CSV not found at {csv_path}")
        self.raw_data = pd.read_csv(csv_path)
        self._normalize_columns()


    def _normalize_columns(self):
        """Ensure standardized case-insensitive column names and country ISO values."""
        # Convert columns to lowercase for case-insensitivity
        self.raw_data.columns = [col.lower() for col in self.raw_data.columns]
        
        # Handle JST-specific exchange rate column alias
        if "xrusd" in self.raw_data.columns and "exrat" not in self.raw_data.columns:
            self.raw_data.rename(columns={"xrusd": "exrat"}, inplace=True)
            
        # Standardize crucial columns. GDP and its scale metadata require exact
        # names after case normalization: gdp_scale, real GDP, and GDP per capita
        # must never be treated as nominal gdp by substring matching.
        col_mappings = {
            "year": "year",
            "country": "country",
            "iso": "iso",
            "cpi": "cpi",
            "exrat": "exrat",
            "eq_tr": "eq_tr",
            "bond_tr": "bond_tr",
            "bill_rate": "bill_rate",
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
            years = pd.to_numeric(self.raw_data["year"], errors="coerce")
            invalid = years.isna() | (years % 1 != 0)
            self._has_invalid_year_values = bool(invalid.any())
            self.raw_data["year"] = years
            self.raw_data.dropna(subset=["year"], inplace=True)
            self.raw_data["year"] = self.raw_data["year"].astype(int)
        else:
            self._has_invalid_year_values = False

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

        Registered perspectives and foreign markets must be in sample at both
        t and t-1. Out-of-sample periods are unavailable, even if the CSV has
        observations there; missing in-sample observations remain missing.
        Unknown perspectives retain the legacy observed-panel fallback, while
        unknown foreign markets are excluded.
        
        Returns:
          A DataFrame with columns: ['Year', 'Domestic Stock', 'International Stock', 'Bonds', 'Bills']
        """
        p_iso = perspective_country.upper()
        df = self.raw_data.copy()

        perspective_periods = CountryMetadataRegistry.get_sample_periods(p_iso)

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
            # We need the previous year to compute CPI and exchange rate changes
            t_prev = t - 1
            if perspective_periods and (
                not CountryMetadataRegistry.is_valid_year_for_country(p_iso, t)
                or not CountryMetadataRegistry.is_valid_year_for_country(p_iso, t_prev)
            ):
                continue
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
                DOMESTIC_STOCK: dom_stock,
                INTERNATIONAL_STOCK: intl_stock,
                BONDS: dom_bond,
                BILLS: dom_bill,
                "Inflation": cpi_ratio_p - 1,  # Annual inflation rate as a decimal
            })

        result_df = pd.DataFrame(
            processed_rows,
            columns=[
                "Year", DOMESTIC_STOCK, INTERNATIONAL_STOCK,
                BONDS, BILLS, "Inflation",
            ],
        )
        # Drop rows where crucial elements are NaN
        result_df.dropna(subset=["Domestic Stock", "International Stock", "Bonds", "Bills"], inplace=True)
        return result_df.reset_index(drop=True)

    def get_pooled_processed_returns(
        self,
        perspective_countries: Optional[List[str]] = None,
        foreign_countries: Optional[List[str]] = None,
        weight_method: str = "gdp_lagged",
        gdp_lag: int = 2,
        gdp_scale_factors: Optional[Mapping[str, float]] = None,
        min_weight_coverage: float = 1.0,
    ) -> pd.DataFrame:
        """Build complete annual return vectors for every country perspective.

        Unlike :meth:`get_processed_returns`, this method uses the observed
        JST panel as its eligibility calendar rather than the paper's 39-country
        sample-bound registry. It requires two consecutive CPI/FX observations
        for each return and does not bridge calendar gaps.

        ``gdp_lagged`` weights foreign equity markets by nominal GDP scaled to
        local-currency units and divided by JST's ``xrusd`` field at
        ``gdp_lag`` years before the return. This is labeled GDP converted
        using JST FX because its historical FX timing/source conventions are
        not uniform. GDP's stored magnitude must be converted through either a
        ``gdp_scale`` column or ``gdp_scale_factors``; built-in scale factors
        assume JST R6. Nominal GDP requires the exact ``gdp`` column name
        (case-insensitive); missing nominal GDP or scale metadata fails closed.
        The default foreign universe includes markets with an observed
        equity-return series. Within that universe, every market must have lagged
        weight inputs and a year-``t`` return. A lower ``min_weight_coverage`` permits
        partial return baskets to be renormalized; the achieved share is
        retained in the result and in :meth:`get_pooled_coverage_report`.

        The returned rows contain only complete four-asset vectors. Inflation
        and weighting coverage are metadata for diagnostics; callers should
        not feed the ``Inflation`` metadata back into the simulator alongside
        these already-real returns.
        """
        if weight_method not in {"gdp_lagged", "equal"}:
            raise ValueError("weight_method must be 'gdp_lagged' or 'equal'.")
        if isinstance(gdp_lag, bool) or not isinstance(gdp_lag, int) or gdp_lag < 1:
            raise ValueError("gdp_lag must be a positive integer.")
        try:
            min_weight_coverage = float(min_weight_coverage)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("min_weight_coverage must be between 0 and 1.") from exc
        if not np.isfinite(min_weight_coverage) or not 0 < min_weight_coverage <= 1:
            raise ValueError("min_weight_coverage must be greater than 0 and at most 1.")

        df = self.raw_data.copy()
        if "iso" not in df.columns or "year" not in df.columns:
            raise ValueError("JST input must contain 'iso' and 'year' columns.")
        if self._has_invalid_year_values:
            raise ValueError("Pooled JST input years must be present whole numbers.")
        df["iso"] = df["iso"].astype(str).str.strip().str.upper()
        if df["iso"].isin({"", "NAN", "NONE"}).any():
            raise ValueError("JST input country identifiers must be present.")
        numeric_years = pd.to_numeric(df["year"], errors="coerce")
        if numeric_years.isna().any() or (numeric_years % 1 != 0).any():
            raise ValueError("JST input years must be present whole numbers.")
        df["year"] = numeric_years.astype(int)
        if df.empty:
            raise ValueError("JST input has no rows with valid country and year.")

        required_columns = {"cpi", "exrat", "eq_tr", "bond_tr", "bill_rate"}
        if weight_method == "gdp_lagged":
            required_columns.add("gdp")
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(
                "JST pooled return construction requires columns: "
                + ", ".join(missing_columns)
            )

        numeric_columns = required_columns | ({"gdp_scale"} if "gdp_scale" in df else set())
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        duplicated = df.duplicated(subset=["year", "iso"], keep=False)
        if duplicated.any():
            examples = df.loc[duplicated, ["year", "iso"]].head(3).to_dict("records")
            raise ValueError(
                f"JST input must have at most one row per country-year; examples: {examples}"
            )

        try:
            pivoted = df.pivot(index="year", columns="iso")
        except ValueError as exc:
            raise ValueError("Could not pivot JST data into unique country-years.") from exc
        years = sorted(int(year) for year in pivoted.index)
        countries = sorted(str(country) for country in df["iso"].unique())

        def variable_frame(name: str) -> pd.DataFrame:
            if name in pivoted.columns.get_level_values(0):
                return pivoted[name]
            return pd.DataFrame(index=years, columns=countries, dtype=float)

        cpi_df = variable_frame("cpi")
        exrat_df = variable_frame("exrat")
        eq_df = variable_frame("eq_tr")
        bond_df = variable_frame("bond_tr")
        bill_df = variable_frame("bill_rate")
        gdp_df = variable_frame("gdp")
        scale_df = variable_frame("gdp_scale")
        if "USA" in exrat_df:
            exrat_df["USA"] = exrat_df["USA"].fillna(1.0)

        if perspective_countries is None:
            perspectives = countries
        else:
            perspectives = sorted({str(country).strip().upper() for country in perspective_countries})
            unknown = sorted(set(perspectives) - set(countries))
            if unknown:
                raise ValueError(
                    "Unknown perspective country code(s): " + ", ".join(unknown)
                )
        if not perspectives:
            raise ValueError("At least one perspective country is required.")
        equity_return_counts = {}
        for country in countries:
            if country not in eq_df:
                equity_return_counts[country] = 0
                continue
            country_returns = pd.to_numeric(eq_df[country], errors="coerce").to_numpy(dtype=float)
            equity_return_counts[country] = int(
                np.sum(np.isfinite(country_returns) & (country_returns >= -1.0))
            )

        if foreign_countries is None:
            # A market with no observed equity-return series cannot contribute
            # to the international basket. Exclude it from the default universe
            # and report the exclusion instead of making every strict basket
            # unusable.
            foreign_universe = [
                country for country in countries if equity_return_counts[country] > 0
            ]
        else:
            foreign_universe = sorted(
                {str(country).strip().upper() for country in foreign_countries}
            )
            unknown = sorted(set(foreign_universe) - set(countries))
            if unknown:
                raise ValueError("Unknown foreign country code(s): " + ", ".join(unknown))
        if not foreign_universe:
            raise ValueError("At least one foreign market must be declared.")

        selected_markets = set(foreign_universe)
        self._pooled_market_universe_report = pd.DataFrame([
            {
                "ISO": country,
                "Included": country in selected_markets,
                "EquityReturnObservations": equity_return_counts[country],
                "SelectionReason": (
                    "selected_by_caller"
                    if foreign_countries is not None and country in selected_markets
                    else "has_observed_equity_returns"
                    if country in selected_markets
                    else "no_observed_equity_return_series"
                    if foreign_countries is None
                    else "not_in_caller_declared_universe"
                ),
            }
            for country in countries
        ])

        supplied_scales: Dict[str, float] = dict(JST_R6_GDP_SCALE_FACTORS)
        for country, value in (gdp_scale_factors or {}).items():
            code = str(country).strip().upper()
            try:
                factor = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"GDP scale factor for {code} must be positive and finite.") from exc
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f"GDP scale factor for {code} must be positive and finite.")
            supplied_scales[code] = factor

        rows = []
        diagnostics = []
        for perspective in perspectives:
            for year in years:
                previous_year = year - 1
                weight_year = year - gdp_lag
                report = {
                    "Country": perspective,
                    "Year": year,
                    "WeightYear": weight_year if weight_method == "gdp_lagged" else None,
                    "WeightMethod": weight_method,
                    "Included": False,
                    "ExclusionReason": None,
                    "WeightCoverage": 0.0,
                    "ForeignMarketsUsed": 0,
                    "ForeignMarketsWeightable": 0,
                    "ForeignUniverseSize": max(
                        0,
                        sum(country != perspective for country in foreign_universe),
                    ),
                    "WeightInputMarketCoverage": 0.0,
                }

                if previous_year not in cpi_df.index:
                    report["ExclusionReason"] = "missing_previous_calendar_year"
                    diagnostics.append(report)
                    continue
                if perspective not in cpi_df or perspective not in exrat_df:
                    report["ExclusionReason"] = "missing_perspective_series"
                    diagnostics.append(report)
                    continue

                cpi_now = cpi_df.at[year, perspective]
                cpi_previous = cpi_df.at[previous_year, perspective]
                fx_now_home = exrat_df.at[year, perspective]
                fx_previous_home = exrat_df.at[previous_year, perspective]
                values = (cpi_now, cpi_previous, fx_now_home, fx_previous_home)
                if any(pd.isna(value) or not np.isfinite(value) or value <= 0 for value in values):
                    report["ExclusionReason"] = "invalid_perspective_cpi_or_fx"
                    diagnostics.append(report)
                    continue

                cpi_ratio = float(cpi_now) / float(cpi_previous)
                if not np.isfinite(cpi_ratio) or cpi_ratio <= 0:
                    report["ExclusionReason"] = "invalid_perspective_cpi_or_fx"
                    diagnostics.append(report)
                    continue
                domestic_nominal = []
                for frame in (eq_df, bond_df, bill_df):
                    value = frame.at[year, perspective] if perspective in frame else np.nan
                    if pd.isna(value) or not np.isfinite(value) or value < -1:
                        domestic_nominal.append(None)
                    else:
                        domestic_nominal.append(float(value))
                if any(value is None for value in domestic_nominal):
                    report["ExclusionReason"] = "incomplete_domestic_asset_vector"
                    diagnostics.append(report)
                    continue

                candidate_weights: Dict[str, float] = {}
                eligible_foreign = [
                    country for country in foreign_universe if country != perspective
                ]
                if weight_method == "equal":
                    candidate_weights = {
                        country: 1.0 for country in eligible_foreign
                    }
                elif weight_year in gdp_df.index:
                    for country in eligible_foreign:
                        gdp_value = (
                            gdp_df.at[weight_year, country]
                            if country in gdp_df else np.nan
                        )
                        weight_fx = exrat_df.at[weight_year, country] if country in exrat_df else np.nan
                        if pd.isna(gdp_value) or not np.isfinite(gdp_value) or gdp_value <= 0:
                            continue
                        if pd.isna(weight_fx) or not np.isfinite(weight_fx) or weight_fx <= 0:
                            continue
                        scale_value = (
                            scale_df.at[weight_year, country]
                            if country in scale_df and not pd.isna(scale_df.at[weight_year, country])
                            else supplied_scales.get(country)
                        )
                        if scale_value is None or not np.isfinite(scale_value) or scale_value <= 0:
                            raise ValueError(
                                f"Missing positive GDP scale factor for {country} in {weight_year}; "
                                "provide a gdp_scale column or gdp_scale_factors mapping."
                            )
                        candidate_weights[country] = (
                            float(gdp_value) * float(scale_value) / float(weight_fx)
                        )

                total_weight = float(sum(candidate_weights.values()))
                report["ForeignMarketsWeightable"] = len(candidate_weights)
                report["WeightInputMarketCoverage"] = (
                    len(candidate_weights) / len(eligible_foreign)
                    if eligible_foreign else 0.0
                )
                if (
                    weight_method == "gdp_lagged"
                    and len(candidate_weights) != len(eligible_foreign)
                ):
                    report["ExclusionReason"] = "incomplete_foreign_weight_inputs"
                    diagnostics.append(report)
                    continue
                if not candidate_weights or not np.isfinite(total_weight) or total_weight <= 0:
                    report["ExclusionReason"] = "no_weightable_foreign_markets"
                    diagnostics.append(report)
                    continue

                observed = []
                for country, weight in candidate_weights.items():
                    if country not in eq_df or country not in exrat_df:
                        continue
                    equity_nominal = eq_df.at[year, country]
                    foreign_fx_now = exrat_df.at[year, country]
                    foreign_fx_previous = exrat_df.at[previous_year, country]
                    foreign_values = (equity_nominal, foreign_fx_now, foreign_fx_previous)
                    if any(
                        pd.isna(value) or not np.isfinite(value) or value <= 0
                        for value in foreign_values[1:]
                    ):
                        continue
                    if pd.isna(equity_nominal) or not np.isfinite(equity_nominal) or equity_nominal < -1:
                        continue
                    return_home = (
                        (1.0 + float(equity_nominal))
                        * (float(foreign_fx_previous) / float(foreign_fx_now))
                        * (float(fx_now_home) / float(fx_previous_home))
                        - 1.0
                    )
                    if np.isfinite(return_home) and return_home >= -1:
                        observed.append((float(weight), float(return_home)))

                observed_weight = float(sum(weight for weight, _ in observed))
                coverage = observed_weight / total_weight
                report["WeightCoverage"] = coverage
                report["ForeignMarketsUsed"] = len(observed)
                if coverage + 1e-12 < min_weight_coverage or observed_weight <= 0:
                    report["ExclusionReason"] = "insufficient_foreign_weight_coverage"
                    diagnostics.append(report)
                    continue

                nominal_international = sum(
                    weight * return_value for weight, return_value in observed
                ) / observed_weight
                domestic_real = [
                    (1.0 + value) / cpi_ratio - 1.0 for value in domestic_nominal
                ]
                if any(
                    not np.isfinite(value) or value < -1.0
                    for value in domestic_real
                ):
                    report["ExclusionReason"] = "invalid_real_domestic_return"
                    diagnostics.append(report)
                    continue
                international_real = (1.0 + nominal_international) / cpi_ratio - 1.0
                if not np.isfinite(international_real) or international_real < -1:
                    report["ExclusionReason"] = "invalid_real_international_return"
                    diagnostics.append(report)
                    continue

                rows.append({
                    "Country": perspective,
                    "Year": year,
                    DOMESTIC_STOCK: domestic_real[0],
                    INTERNATIONAL_STOCK: international_real,
                    BONDS: domestic_real[1],
                    BILLS: domestic_real[2],
                    "Inflation": cpi_ratio - 1.0,
                    "WeightCoverage": coverage,
                    "ForeignMarketsUsed": len(observed),
                    "ForeignMarketsWeightable": len(candidate_weights),
                    "WeightInputMarketCoverage": report["WeightInputMarketCoverage"],
                    "WeightYear": weight_year if weight_method == "gdp_lagged" else None,
                })
                report["Included"] = True
                diagnostics.append(report)

        result_columns = [
            "Country", "Year", DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS,
            BILLS, "Inflation", "WeightCoverage", "ForeignMarketsUsed",
            "ForeignMarketsWeightable", "WeightInputMarketCoverage", "WeightYear",
        ]
        self._pooled_coverage_report = pd.DataFrame(diagnostics)
        result = pd.DataFrame(rows, columns=result_columns)
        if not result.empty:
            result.sort_values(["Country", "Year"], inplace=True)
            result.reset_index(drop=True, inplace=True)
        return result

    def get_pooled_coverage_report(self) -> pd.DataFrame:
        """Return country-year inclusion and foreign-basket coverage diagnostics."""
        report = getattr(self, "_pooled_coverage_report", None)
        return report.copy() if report is not None else pd.DataFrame()

    def get_pooled_market_universe_report(self) -> pd.DataFrame:
        """Return the selected foreign markets and source-series exclusions."""
        report = getattr(self, "_pooled_market_universe_report", None)
        return report.copy() if report is not None else pd.DataFrame()
