from abc import ABC, abstractmethod
import math
import random
from collections import Counter
import pandas as pd
from typing import List, Dict, Mapping, Optional
from src.config import MarketConfig
from src.provenance import DataProvenance, csv_provenance
from src.assets import DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS


class Market(ABC):
    """Base class for market return generators."""

    # Custom engines may override this with an explicit source declaration.
    provenance = DataProvenance()

    @abstractmethod
    def get_annual_returns(self) -> Dict[str, float]:
        raise NotImplementedError

    def start_new_path(self) -> None:
        """Reset path state when a simulator begins a new trial, if needed."""
        return None


class SyntheticMarket(Market):
    """Generates bounded simple returns with lognormal gross returns."""

    provenance = DataProvenance("synthetic", "SyntheticMarket lognormal model")

    def __init__(self, market_configs: List[MarketConfig], seed: int | None = None):
        self.configs = market_configs
        self._lognormal_params = []
        for config in self.configs:
            try:
                expected_return = float(config.expected_return)
                volatility = float(config.volatility)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Market '{config.name}' expected_return and volatility "
                    "must be finite numbers."
                ) from exc

            if not math.isfinite(expected_return) or expected_return <= -1.0:
                raise ValueError(
                    f"Market '{config.name}' expected_return must be finite "
                    "and greater than -1."
                )
            if not math.isfinite(volatility) or volatility < 0.0:
                raise ValueError(
                    f"Market '{config.name}' volatility must be finite and "
                    "non-negative."
                )

            # Convert arithmetic simple-return moments to the parameters of
            # log gross returns. This keeps the configured mean and standard
            # deviation while ensuring every sampled gross return is positive.
            relative_volatility = volatility / (1.0 + expected_return)
            log_variance = 2.0 * math.log(math.hypot(1.0, relative_volatility))
            log_mean = math.log1p(expected_return) - 0.5 * log_variance
            if not math.isfinite(log_mean) or not math.isfinite(log_variance):
                raise ValueError(
                    f"Market '{config.name}' expected_return and volatility "
                    "are outside the supported numeric range."
                )
            self._lognormal_params.append(
                (config.name, expected_return, log_mean, math.sqrt(log_variance))
            )

        if seed is not None:
            random.seed(seed)

    def get_annual_returns(self) -> Dict[str, float]:
        returns = {}
        for name, expected_return, log_mean, log_volatility in self._lognormal_params:
            if log_volatility == 0.0:
                returns[name] = expected_return
            else:
                returns[name] = math.exp(
                    random.gauss(log_mean, log_volatility)
                ) - 1.0
        return returns


class BootstrapMarket(Market):
    """
    Generates returns by sampling CSV data with contiguous blocks.
    The source is unverified unless explicit provenance is provided or the
    input matches the known bundled synthetic panel.
    """

    _metadata_columns = {"year", "country", "iso"}

    def __init__(
        self, csv_path: str, block_size: int = 10, seed: int | None = None,
        *, provenance: DataProvenance | None = None,
    ):
        self.block_size = self._validate_block_size(block_size)
        self.provenance = csv_provenance(csv_path, provenance)
        try:
            self.data = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("Bootstrap return panel must not be empty.") from exc
        self._validate_panel(self.data, panel_name="Bootstrap return panel")
        if seed is not None:
            random.seed(seed)

        # State for current simulation path
        self.current_index = 0
        self.remaining_in_block = 0

    @staticmethod
    def _validate_block_size(block_size: int) -> int:
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer.")
        return block_size

    @classmethod
    def _validate_panel(cls, data: pd.DataFrame, *, panel_name: str) -> None:
        if data.empty:
            raise ValueError(f"{panel_name} must not be empty.")

        for column in data.columns:
            if column.strip().lower() in cls._metadata_columns:
                continue
            for row_index, raw_value in data[column].items():
                # A missing observation is intentionally omitted later so the
                # simulator can distinguish unavailable coverage from a bad
                # observed value.
                if pd.isna(raw_value):
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{panel_name} value for {column} at row {row_index} "
                        "must be numeric and finite."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"{panel_name} value for {column} at row {row_index} "
                        "must be numeric and finite."
                    )
                if value < -1.0:
                    raise ValueError(
                        f"{panel_name} value for {column} at row {row_index} "
                        "must be at least -1."
                    )

    @classmethod
    def _validate_perspective_source_columns(cls, data: pd.DataFrame) -> None:
        """Fail clearly before JST arithmetic sees malformed numeric inputs."""
        return_columns = {"eq_tr", "bond_tr", "bill_rate"}
        numeric_columns = {
            "cpi", "exrat", "eq_tr", "bond_tr", "bill_rate", "gdp"
        }
        for column in sorted(numeric_columns & set(data.columns)):
            for row_index, raw_value in data[column].items():
                if pd.isna(raw_value):
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"Perspective source column {column} at row {row_index} "
                        "must be numeric and finite."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Perspective source column {column} at row {row_index} "
                        "must be numeric and finite."
                    )
                if column in return_columns and value < -1.0:
                    raise ValueError(
                        f"Perspective source return {column} at row {row_index} "
                        "must be at least -1."
                    )

    def start_new_path(self):
        """Reset state for a new simulation trial."""
        self.current_index = random.randint(0, len(self.data) - 1)
        self.remaining_in_block = self.block_size

    def _observed_returns(self, row: pd.Series) -> Dict[str, float]:
        """Keep observed series without discarding a partially covered year.

        An absent key lets active-asset validation reject unavailable holdings;
        a NaN value would incorrectly present the series as available. Other
        non-finite or non-numeric values are invalid input and fail closed.
        """
        returns = {}
        for col, value in row.items():
            if col.strip().lower() in self._metadata_columns or pd.isna(value):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Bootstrap return for {col} must be numeric and finite."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Bootstrap return for {col} must be numeric and finite."
                )
            if value < -1.0:
                raise ValueError(
                    f"Bootstrap return for {col} must be at least -1."
                )
            returns[col] = value
        return returns

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data from the current block."""
        if self.remaining_in_block <= 0:
            # Pick a new random start for the next block
            self.current_index = random.randint(0, len(self.data) - 1)
            self.remaining_in_block = self.block_size

        # Get data from the current index (wrap around if at end of CSV)
        idx = self.current_index % len(self.data)
        row = self.data.iloc[idx]

        returns = self._observed_returns(row)

        # Advance state
        self.current_index += 1
        self.remaining_in_block -= 1

        return returns


class StationaryBootstrapMarket(BootstrapMarket):
    """
    Generates returns using Politis & Romano (1994) Stationary Block Bootstrap.
    Block lengths are geometrically distributed with expected value equal to block_size.
    """

    def __init__(
        self, csv_path: str, block_size: int = 10, seed: int | None = None,
        *, provenance: DataProvenance | None = None,
    ):
        super().__init__(csv_path, block_size, seed, provenance=provenance)
        self.first_step = True

    def start_new_path(self):
        """Reset state for a new simulation trial."""
        self.first_step = True
        self.current_index = random.randint(0, len(self.data) - 1)

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data, jumping to a new block with probability 1/block_size."""
        if self.first_step:
            self.first_step = False
        else:
            # Jump with probability 1 / block_size
            if random.random() < (1.0 / self.block_size):
                self.current_index = random.randint(0, len(self.data) - 1)
            else:
                self.current_index = (self.current_index + 1) % len(self.data)

        row = self.data.iloc[self.current_index]
        return self._observed_returns(row)


class PooledPerspectiveBootstrapMarket(Market):
    """Annual bootstrap over complete country-perspective return vectors.

    The default start rule gives each country equal probability, then chooses
    a valid country-year uniformly within that country. Geometric blocks remain
    within a contiguous country segment; gaps and endpoints trigger a fresh
    country/year draw. An instance-local RNG avoids changing global random
    state. Returned asset returns are real; diagnostic Inflation metadata from
    the source panel is deliberately not passed to the simulator.
    """

    returns_basis = "real"

    def __init__(
        self,
        csv_path: str,
        block_size: int = 10,
        seed: int | None = None,
        *,
        weight_method: str = "gdp_lagged",
        gdp_lag: int = 2,
        gdp_scale_factors: Optional[Mapping[str, float]] = None,
        min_weight_coverage: float = 1.0,
        country_sampling: str = "equal_country",
        perspective_countries: Optional[List[str]] = None,
        foreign_countries: Optional[List[str]] = None,
        provenance: DataProvenance | None = None,
    ):
        from src.data_loader import JSTDataLoader

        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        if country_sampling not in {"equal_country", "observation_weighted"}:
            raise ValueError(
                "country_sampling must be 'equal_country' or 'observation_weighted'."
            )

        self.provenance = csv_provenance(csv_path, provenance)
        loader = JSTDataLoader(csv_path)
        self.gdp_scale_factor_overrides = {
            str(country).strip().upper(): float(value)
            for country, value in (gdp_scale_factors or {}).items()
        }
        self.gdp_scale_column_present = "gdp_scale" in loader.raw_data.columns
        self.data = loader.get_pooled_processed_returns(
            perspective_countries=perspective_countries,
            foreign_countries=foreign_countries,
            weight_method=weight_method,
            gdp_lag=gdp_lag,
            gdp_scale_factors=gdp_scale_factors,
            min_weight_coverage=min_weight_coverage,
        )
        self.coverage_report = loader.get_pooled_coverage_report()
        self.market_universe_report = loader.get_pooled_market_universe_report()
        if self.data.empty:
            raise ValueError(
                "No complete pooled country-year vectors were available. "
                "Review the JST coverage report and basket-coverage threshold."
            )

        required = {"Country", "Year", DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS}
        missing = sorted(required - set(self.data.columns))
        if missing:
            raise ValueError(
                "Pooled return panel is missing required columns: " + ", ".join(missing)
            )
        if self.data.duplicated(subset=["Country", "Year"]).any():
            raise ValueError("Pooled return panel must have unique country-year rows.")

        self.block_size = block_size
        self.country_sampling = country_sampling
        self.weight_method = weight_method
        self.gdp_lag = gdp_lag
        self.min_weight_coverage = min_weight_coverage
        self._rng = random.Random(seed)
        self._segments_by_country: Dict[str, List[List[dict]]] = {}
        self._country_rows: Dict[str, List[tuple[List[dict], int]]] = {}
        self._segment_end_reasons: Dict[int, str] = {}
        self._prepare_segments()
        if not self._segments_by_country:
            raise ValueError("Pooled panel has no usable country segments.")

        self._country_codes = sorted(self._segments_by_country)
        self._all_starts = [
            start for country in self._country_codes for start in self._country_rows[country]
        ]
        self._reset_diagnostics()
        self.start_new_path()

    def _prepare_segments(self) -> None:
        assets = [DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS]
        ordered = self.data.sort_values(["Country", "Year"])
        for country, group in ordered.groupby("Country", sort=True):
            records = group.to_dict("records")
            segments: List[List[dict]] = []
            current: List[dict] = []
            previous_year = None
            for record in records:
                year = int(record["Year"])
                values = [record.get(asset) for asset in assets]
                if any(
                    value is None or pd.isna(value) or not math.isfinite(float(value))
                    or float(value) < -1.0
                    for value in values
                ):
                    raise ValueError(
                        f"Invalid complete-vector return in {country} {year}."
                    )
                if current and year != previous_year + 1:
                    segments.append(current)
                    self._segment_end_reasons[id(current)] = "gap"
                    current = []
                current.append(record)
                previous_year = year
            if current:
                segments.append(current)
                self._segment_end_reasons[id(current)] = "country_endpoint"
            if not segments:
                continue
            self._segments_by_country[str(country)] = segments
            starts = []
            for segment in segments:
                starts.extend((segment, index) for index in range(len(segment)))
            self._country_rows[str(country)] = starts

    def _reset_diagnostics(self) -> None:
        self._country_starts = Counter()
        self._country_observations = Counter()
        self._year_observations = Counter()
        self._country_year_observations = Counter()
        self._block_lengths = Counter()
        self._boundary_restarts = 0
        self._gap_restarts = 0
        self._endpoint_restarts = 0
        self._natural_block_ends = 0
        self._blocks_started = 0
        self._censored_by_path_end = 0

    def reset_diagnostics(self) -> None:
        """Reset sampler counters and path state before a simulation run."""
        self._reset_diagnostics()
        self._clear_path_state()

    def _clear_path_state(self) -> None:
        self._current_segment = None
        self._current_position = 0
        self._current_country = None
        self._remaining_in_block = 0
        self._current_block_length = 0
        self._path_started = False
        self.last_observation = None

    def _draw_block_start(self) -> None:
        if self.country_sampling == "equal_country":
            country = self._rng.choice(self._country_codes)
            segment, position = self._rng.choice(self._country_rows[country])
        else:
            country = None
            segment, position = self._rng.choice(self._all_starts)
            country = str(segment[position]["Country"])

        self._current_segment = segment
        self._current_position = position
        self._current_country = country
        self._remaining_in_block = 1
        continuation_probability = 1.0 / self.block_size
        while self._rng.random() >= continuation_probability:
            self._remaining_in_block += 1
        self._current_block_length = 0
        self._blocks_started += 1
        self._country_starts[country] += 1

    def start_new_path(self) -> None:
        """Start a fresh simulated path while retaining aggregate diagnostics."""
        if getattr(self, "_current_segment", None) is not None and self._current_block_length:
            self.end_path()
        self._clear_path_state()
        self._path_started = True

    def end_path(self) -> None:
        """Record the final block, which is right-censored by the horizon."""
        if self._current_segment is not None and self._current_block_length:
            if self._remaining_in_block > 0:
                self._censored_by_path_end += 1
            else:
                self._natural_block_ends += 1
            self._block_lengths[self._current_block_length] += 1
        self._current_segment = None
        self._remaining_in_block = 0
        self._current_block_length = 0
        self._path_started = False

    def get_annual_returns(self) -> Dict[str, float]:
        if not getattr(self, "_path_started", False):
            self.start_new_path()
        if (
            self._current_segment is None
            or self._remaining_in_block <= 0
            or self._current_position >= len(self._current_segment)
        ):
            if self._current_segment is not None:
                if self._remaining_in_block > 0:
                    self._boundary_restarts += 1
                    end_reason = self._segment_end_reasons.get(
                        id(self._current_segment), "country_endpoint"
                    )
                    if end_reason == "gap":
                        self._gap_restarts += 1
                    else:
                        self._endpoint_restarts += 1
                else:
                    self._natural_block_ends += 1
                if self._current_block_length:
                    self._block_lengths[self._current_block_length] += 1
            self._draw_block_start()

        record = self._current_segment[self._current_position]
        self._current_position += 1
        self._remaining_in_block -= 1
        self._current_block_length += 1
        country = str(record["Country"])
        year = int(record["Year"])
        self._country_observations[country] += 1
        self._year_observations[year] += 1
        self._country_year_observations[(country, year)] += 1
        self.last_observation = {
            key: record[key]
            for key in (
                "Country", "Year", "Inflation", "WeightCoverage",
                "ForeignMarketsUsed", "ForeignMarketsWeightable", "WeightYear",
            )
            if key in record
        }
        return {
            asset: float(record[asset])
            for asset in (DOMESTIC_STOCK, INTERNATIONAL_STOCK, BONDS, BILLS)
        }

    def get_diagnostics(self) -> dict:
        """Return a snapshot of sampler and source-coverage diagnostics."""
        country_count = len(self._country_codes)
        if self.country_sampling == "equal_country":
            intended = {country: 1.0 / country_count for country in self._country_codes}
        else:
            total_rows = sum(len(rows) for rows in self._country_rows.values())
            intended = {
                country: len(self._country_rows[country]) / total_rows
                for country in self._country_codes
            }

        coverage = self.coverage_report
        included = coverage[coverage["Included"]] if not coverage.empty else coverage
        coverage_summary = {}
        if not included.empty:
            coverage_summary = {
                "included_country_years": int(len(included)),
                "mean_foreign_weight_coverage": float(included["WeightCoverage"].mean()),
                "minimum_foreign_weight_coverage": float(included["WeightCoverage"].min()),
                "excluded_country_years": int((~coverage["Included"]).sum()),
            }
        exclusions = (
            coverage.loc[~coverage["Included"], "ExclusionReason"]
            .value_counts(dropna=False).to_dict()
            if not coverage.empty else {}
        )
        country_year_counts = {
            country: {
                year: count
                for (row_country, year), count in self._country_year_observations.items()
                if row_country == country
            }
            for country in self._country_codes
        }

        return {
            "country_sampling": self.country_sampling,
            "weight_method": self.weight_method,
            "gdp_lag": self.gdp_lag,
            "min_weight_coverage": self.min_weight_coverage,
            "block_size": self.block_size,
            "gdp_scale_factor_overrides": dict(self.gdp_scale_factor_overrides),
            "gdp_scale_column_present": self.gdp_scale_column_present,
            "built_in_gdp_scale_release": "JST R6",
            "foreign_market_universe": self.market_universe_report.to_dict("records"),
            "selected_foreign_markets": self.market_universe_report.loc[
                self.market_universe_report["Included"], "ISO"
            ].tolist(),
            "excluded_foreign_markets": self.market_universe_report.loc[
                ~self.market_universe_report["Included"], "ISO"
            ].tolist(),
            "intended_country_start_probabilities": intended,
            "realized_country_start_counts": dict(self._country_starts),
            "realized_country_observation_counts": dict(self._country_observations),
            "realized_year_observation_counts": dict(self._year_observations),
            "realized_country_year_observation_counts": country_year_counts,
            "observed_block_length_counts": dict(self._block_lengths),
            "blocks_started": self._blocks_started,
            "natural_block_ends": self._natural_block_ends,
            "boundary_restarts": self._boundary_restarts,
            "gap_restarts": self._gap_restarts,
            "country_endpoint_restarts": self._endpoint_restarts,
            "blocks_censored_by_path_end": self._censored_by_path_end,
            "valid_segment_lengths": {
                country: [len(segment) for segment in self._segments_by_country[country]]
                for country in self._country_codes
            },
            "source_coverage": coverage_summary,
            "source_exclusions_by_reason": exclusions,
        }


class PerspectiveBootstrapMarket(BootstrapMarket):
    """
    Generates returns by running a perspective-country-aware bootstrap.
    Uses JSTDataLoader to build a customized panel of FX-adjusted real returns.
    """

    def __init__(
        self,
        csv_path: str,
        perspective_country: str = "USA",
        block_size: int = 10,
        stationary_bootstrap: bool = False,
        weight_method: str = "gdp",
        seed: int | None = None,
        *, provenance: DataProvenance | None = None,
    ):
        from src.data_loader import JSTDataLoader

        self.block_size = self._validate_block_size(block_size)
        self.provenance = csv_provenance(csv_path, provenance)
        try:
            loader = JSTDataLoader(csv_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                "Perspective bootstrap raw return panel must not be empty."
            ) from exc
        if loader.raw_data.empty:
            raise ValueError(
                "Perspective bootstrap raw return panel must not be empty."
            )
        self._validate_perspective_source_columns(loader.raw_data)
        processed_data = loader.get_processed_returns(
            perspective_country=perspective_country,
            weight_method=weight_method
        )
        self._validate_panel(
            processed_data, panel_name="Perspective processed return panel"
        )

        self.data = processed_data
        self.stationary_bootstrap = stationary_bootstrap
        if seed is not None:
            random.seed(seed)

        self.current_index = 0
        self.remaining_in_block = 0
        self.first_step = True

    def start_new_path(self):
        """Reset state for a new simulation trial."""
        self.first_step = True
        self.current_index = random.randint(0, len(self.data) - 1)
        self.remaining_in_block = self.block_size

    def get_annual_returns(self) -> Dict[str, float]:
        """Returns the next year of data using the specified bootstrap method."""
        if self.stationary_bootstrap:
            if self.first_step:
                self.first_step = False
            else:
                if random.random() < (1.0 / self.block_size):
                    self.current_index = random.randint(0, len(self.data) - 1)
                else:
                    self.current_index = (self.current_index + 1) % len(self.data)

            row = self.data.iloc[self.current_index]
            return self._observed_returns(row)
        else:
            return super().get_annual_returns()
