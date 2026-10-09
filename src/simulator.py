import random
from typing import Dict, Iterable, List, NamedTuple, Optional, Union
from src.assets import require_return_series, validate_market_returns
from src.config import SimulationConfig
from src.investor import Investor
from src.market import Market, SyntheticMarket
from src.provenance import DataProvenance
from src.strategy import Strategy, FixedAllocationStrategy


class TerminalWealths(list):
    """List-compatible outcomes with explicit data provenance.

    Iteration, indexing, equality, and existing metrics behave as for a list.
    Use to_dict() when exporting results: plain list/JSON conversion discards
    attached metadata, as it does for any list subclass.
    """

    def __init__(
        self, values: Iterable[float] = (), *,
        provenance: DataProvenance = DataProvenance(),
    ):
        if not isinstance(provenance, DataProvenance):
            raise TypeError("provenance must be a DataProvenance instance.")
        super().__init__(values)
        self._provenance = provenance

    @property
    def provenance(self) -> DataProvenance:
        return self._provenance

    def __repr__(self) -> str:
        return f"TerminalWealths({super().__repr__()}, provenance={self.provenance!r})"

    def to_dict(self) -> dict:
        return {
            "terminal_wealths": list(self),
            "provenance": self.provenance.to_dict(),
        }


class SimulationResult(NamedTuple):
    """Returned by run_stochastic() when track_paths=True.

    Attributes:
        terminal_wealths: Final portfolio value for each trial.
        paths: Year-by-year total portfolio value per trial.
            Each inner list starts at the investor's starting age (before any
            market growth) and appends one value per simulated year, so
            len(paths[i]) == trial_end_age - starting_age + 1.
        withdrawal_paths: Year-by-year actual withdrawal amount from the portfolio
            per trial. Aligned with paths, starting with 0.0 at starting_age.
        provenance: Source metadata shared with terminal_wealths. Exposed as a
            property to preserve the existing three-field tuple/unpacking API.
    """

    terminal_wealths: List[float]
    paths: List[List[float]]
    withdrawal_paths: List[List[float]]

    @property
    def provenance(self) -> DataProvenance:
        # Keep manually constructed legacy results usable and honestly labeled.
        return getattr(self.terminal_wealths, "provenance", DataProvenance())

    def to_dict(self) -> dict:
        """Export outcomes, paths, and source metadata together."""
        return {
            "terminal_wealths": list(self.terminal_wealths),
            "paths": self.paths,
            "withdrawal_paths": self.withdrawal_paths,
            "provenance": self.provenance.to_dict(),
        }


class Simulator:
    """Orchestrates the lifecycle simulation for a single investor."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def _get_trial_end_age(self) -> int:
        """Determines the end age for a simulation trial."""
        if not self.config.enable_mortality:
            return self.config.end_age

        # Simple mortality model: probability of death increases with age
        # Starting from age 50, death probability increases
        for age in range(self.config.starting_age, 120):
            if age < 50:
                prob_death = 0.001
            else:
                # Roughly doubling every 7-8 years (Gompertz-like)
                prob_death = 0.001 * (1.1 ** (age - 50))

            if random.random() < prob_death:
                return age
        return 120

    def run_deterministic(self, annual_return: float = 0.0) -> float:
        """
        Runs a single lifecycle simulation with a fixed annual return.
        Used for mathematical verification (Increment 2).
        Works with the multi-asset Investor by using the first market's name.
        """
        # Create a simple 100% allocation to the first market
        market_name = self.config.markets[0].name
        strategy = FixedAllocationStrategy({market_name: 1.0})
        target_alloc = strategy.get_allocation(self.config.starting_age)

        investor = Investor(
            age=self.config.starting_age,
            current_salary=self.config.initial_salary,
            holdings={market_name: 0.0},
        )

        # Main Lifecycle Loop
        while investor.age < self.config.end_age:
            # Step 1: Market Growth
            investor.apply_returns({market_name: annual_return})

            # Step 2: Income/Savings or Withdrawal
            if investor.age < self.config.retirement_age:
                # Accumulation Phase
                investor.earn_and_save(self.config.savings_rate, target_alloc)
                investor.grow_salary(self.config.salary_growth_rate)
            else:
                # Decumulation Phase
                withdrawal_amount = (
                    investor.total_portfolio_value * self.config.withdrawal_rate
                )
                investor.withdraw(withdrawal_amount, target_alloc)

            # Step 3: Aging
            investor.age += 1

        return investor.total_portfolio_value

    def _get_inflation(self, annual_returns: Dict[str, float]) -> float:
        """Returns the realized annual inflation rate from market returns.

        Returns 0.0 if the market engine does not provide an ``Inflation`` key
        (or explicitly provides ``None``; both are normalized by
        ``validate_market_returns``), so all inflation-sensitive trackers
        simply stay flat that year. Other malformed metadata is rejected when
        the market observation is received, before either decumulation path.
        """
        value = annual_returns.get("Inflation")
        return value if value is not None else 0.0

    # ------------------------------------------------------------------
    # Decumulation helpers
    # ------------------------------------------------------------------

    def _annual_decumulation_step(
        self,
        investor: "Investor",
        annual_returns: Dict[str, float],
        target_alloc: Dict[str, float],
        fixed_withdrawal_amount: float,
        current_cap: Optional[float],
        current_floor: Optional[float],
    ) -> tuple[float, float, float, Optional[float], Optional[float]]:
        """Perform one *annual* decumulation step.

        Returns
        -------
        (actual_withdrawn, new_fixed_withdrawal_amount, inflation,
         new_cap, new_floor)
        """
        # a) Market growth already applied before calling this helper.

        # b) Compute raw withdrawal target (annual)
        if self.config.withdrawal_strategy == "fixed_real":
            raw_withdrawal = fixed_withdrawal_amount
        else:  # variable_pct
            raw_withdrawal = (
                investor.total_portfolio_value * self.config.withdrawal_rate
            )

        # c) Apply cap (before social security offset)
        if current_cap is not None:
            raw_withdrawal = min(raw_withdrawal, current_cap)

        # d) Social security offset
        net_from_portfolio = max(
            0.0, raw_withdrawal - self.config.social_security_benefit
        )

        # e) Apply floor: ensure minimum real expenditure is met from portfolio
        if current_floor is not None:
            floor_from_portfolio = max(
                0.0,
                current_floor - self.config.social_security_benefit,
            )
            net_from_portfolio = max(net_from_portfolio, floor_from_portfolio)

        # f) Execute withdrawal
        actual_withdrawn = investor.withdraw(net_from_portfolio, target_alloc)

        # g) Step inflation trackers at end of year
        inflation = self._get_inflation(annual_returns)
        if self.config.withdrawal_inflation_adjusted:
            fixed_withdrawal_amount *= 1 + inflation
        if self.config.withdrawal_cap_inflation_adjusted and current_cap is not None:
            current_cap *= 1 + inflation
        if self.config.withdrawal_floor_inflation_adjusted and current_floor is not None:
            current_floor *= 1 + inflation

        return (
            actual_withdrawn,
            fixed_withdrawal_amount,
            inflation,
            current_cap,
            current_floor,
        )

    def _monthly_decumulation_step(
        self,
        investor: "Investor",
        annual_returns: Dict[str, float],
        target_alloc: Dict[str, float],
        fixed_withdrawal_amount: float,
        current_cap: Optional[float],
        current_floor: Optional[float],
    ) -> tuple[float, float, float, Optional[float], Optional[float]]:
        """Perform one *annual* decumulation step via 12 monthly sub-steps.

        Follows Anarkulova et al. (2023) timing semantics:
        - Withdrawal is evaluated at the **beginning of each month**.
        - The annual market return is converted to monthly compounded rates:
            r_monthly = (1 + R_annual)^(1/12) - 1
        - Per-month share of annual quantities (withdrawal target, SS, cap,
          floor) is 1/12 of the annual figure.
        - Year-end inflation adjustment is performed once, after the 12th month.
        - An annual return of exactly ``-1`` is a valid total loss. Returns
          below ``-1`` are rejected before this helper is called.

        Returns
        -------
        (total_withdrawn_this_year, new_fixed_withdrawal_amount, inflation,
         new_cap, new_floor)
        """
        # Convert annual returns → monthly compounded rates (per asset)
        monthly_returns: Dict[str, float] = {}
        for asset, r_annual in annual_returns.items():
            base = 1.0 + r_annual
            monthly_returns[asset] = base ** (1.0 / 12.0) - 1.0

        # Monthly split constants (SS, cap, floor are shared equally across months)
        ss_monthly = self.config.social_security_benefit / 12.0
        annual_w_rate = self.config.withdrawal_rate / 12.0  # for variable_pct

        total_withdrawn = 0.0

        for _month in range(12):
            # --- Beginning-of-month withdrawal ---
            if self.config.withdrawal_strategy == "fixed_real":
                raw_w = fixed_withdrawal_amount / 12.0
            else:  # variable_pct
                raw_w = investor.total_portfolio_value * annual_w_rate

            # Apply monthly cap
            if current_cap is not None:
                raw_w = min(raw_w, current_cap / 12.0)

            # Social security offset (monthly)
            net_from_portfolio = max(0.0, raw_w - ss_monthly)

            # Apply monthly floor
            if current_floor is not None:
                floor_monthly = max(0.0, current_floor / 12.0 - ss_monthly)
                net_from_portfolio = max(net_from_portfolio, floor_monthly)

            # Execute monthly withdrawal
            actual_m = investor.withdraw(net_from_portfolio, target_alloc)
            total_withdrawn += actual_m

            # --- End-of-month: apply monthly growth then rebalance ---
            investor.apply_returns(monthly_returns)
            investor.rebalance(target_alloc)

        # Year-end inflation adjustment (applied once, after 12 months)
        inflation = self._get_inflation(annual_returns)
        if self.config.withdrawal_inflation_adjusted:
            fixed_withdrawal_amount *= 1 + inflation
        if self.config.withdrawal_cap_inflation_adjusted and current_cap is not None:
            current_cap *= 1 + inflation
        if self.config.withdrawal_floor_inflation_adjusted and current_floor is not None:
            current_floor *= 1 + inflation

        return (
            total_withdrawn,
            fixed_withdrawal_amount,
            inflation,
            current_cap,
            current_floor,
        )

    def run_stochastic(
        self,
        strategy: Strategy,
        num_trials: int = 1000,
        market_engine: Market | None = None,
        track_paths: bool = False,
    ) -> Union[TerminalWealths, SimulationResult]:
        """
        Runs multiple lifecycle simulations with strategy-based rebalancing.
        If market_engine is not provided, defaults to SyntheticMarket using config.

        When track_paths=True, returns a SimulationResult(terminal_wealths, paths, withdrawal_paths)
        named-tuple. Each entry in ``paths`` is the year-by-year total portfolio
        value for one trial (starting snapshot + one value per simulated year).
        When track_paths=False (default), returns TerminalWealths, a list subclass
        of terminal wealth values for backward compatibility. Both return forms
        expose immutable ``provenance`` metadata and ``to_dict()`` for export.

        The ``config.decumulation_granularity`` field controls the decumulation
        time-step used during the retirement phase:

        - ``"annual"``  (default): one withdrawal per year, backward-compatible.
        - ``"monthly"``: withdrawals at the beginning of each calendar month,
          matching the timing semantics in Anarkulova et al. (2023). Each year's
          annual return is decomposed into 12 monthly compounded returns.
        """
        use_monthly = (
            getattr(self.config, "decumulation_granularity", "annual") == "monthly"
        )

        if market_engine is not None and getattr(market_engine, "returns_basis", None) == "real":
            inflation_flags = (
                "withdrawal_inflation_adjusted",
                "withdrawal_cap_inflation_adjusted",
                "withdrawal_floor_inflation_adjusted",
            )
            enabled_flags = [name for name in inflation_flags if getattr(self.config, name, False)]
            if enabled_flags:
                raise ValueError(
                    "The supplied market returns are already real; disable "
                    "inflation-adjustment flags for withdrawals, caps, and floors."
                )
            reset_diagnostics = getattr(market_engine, "reset_diagnostics", None)
            if callable(reset_diagnostics):
                reset_diagnostics()

        provenance = (
            SyntheticMarket.provenance if market_engine is None
            else getattr(market_engine, "provenance", DataProvenance())
        )
        terminal_wealths = TerminalWealths(provenance=provenance)
        paths: List[List[float]] = []
        withdrawal_paths: List[List[float]] = []

        for _ in range(num_trials):
            # Use provided engine or default to Synthetic
            market = (
                market_engine if market_engine is not None
                else SyntheticMarket(self.config.markets)
            )

            # Give any stateful market engine a fresh path for this trial.
            start_new_path = getattr(market, "start_new_path", None)
            if callable(start_new_path):
                start_new_path()

            investor = Investor(
                age=self.config.starting_age,
                current_salary=self.config.initial_salary,
                # Initialize holdings with zeros for all assets
                holdings={m.name: 0.0 for m in self.config.markets},
            )

            trial_end_age = self._get_trial_end_age()
            trial_path: List[float] = []
            trial_withdrawals: List[float] = []
            if track_paths:
                # Snapshot before any growth: all zeros at t=starting_age
                trial_path.append(investor.total_portfolio_value)
                trial_withdrawals.append(0.0)

            # Per-trial retirement trackers (initialized on first retirement year)
            fixed_withdrawal_amount: float = 0.0
            current_cap: Optional[float] = self.config.withdrawal_cap
            current_floor: Optional[float] = self.config.withdrawal_floor

            while investor.age < trial_end_age:
                # 1. Determine target allocation for current age
                target_alloc = strategy.get_allocation(investor.age)
                annual_returns = validate_market_returns(
                    market.get_annual_returns(),
                    context=f"{type(market).__name__} returns at age {investor.age}",
                )
                target_alloc = strategy.resolve_allocation_for_market(
                    target_alloc, annual_returns.keys()
                )
                require_return_series(
                    (asset for asset, weight in target_alloc.items() if weight != 0.0),
                    annual_returns,
                    context=f"{type(strategy).__name__} at age {investor.age}",
                )
                investor.rebalance(target_alloc)

                withdrawal_amount_this_year = 0.0

                # 3. Income/Savings (accumulation) or Withdrawal (decumulation)
                if investor.age < self.config.retirement_age:
                    # --- Accumulation Phase (unchanged) ---
                    investor.apply_returns(annual_returns)
                    investor.earn_and_save(self.config.savings_rate, target_alloc)
                    investor.grow_salary(self.config.salary_growth_rate)
                    # Annual rebalancing for accumulation phase
                    investor.rebalance(target_alloc)
                else:
                    # --- Decumulation Phase ---

                    # Capture fixed withdrawal base at the start of retirement
                    if investor.age == self.config.retirement_age:
                        fixed_withdrawal_amount = (
                            investor.total_portfolio_value * self.config.withdrawal_rate
                        )


                    if use_monthly:
                        # Monthly sub-stepping: growth and rebalancing happen
                        # inside _monthly_decumulation_step.
                        (
                            withdrawal_amount_this_year,
                            fixed_withdrawal_amount,
                            _inflation,
                            current_cap,
                            current_floor,
                        ) = self._monthly_decumulation_step(
                            investor=investor,
                            annual_returns=annual_returns,
                            target_alloc=target_alloc,
                            fixed_withdrawal_amount=fixed_withdrawal_amount,
                            current_cap=current_cap,
                            current_floor=current_floor,
                        )
                    else:
                        # Annual path: apply market growth first, then withdraw.
                        investor.apply_returns(annual_returns)
                        (
                            withdrawal_amount_this_year,
                            fixed_withdrawal_amount,
                            _inflation,
                            current_cap,
                            current_floor,
                        ) = self._annual_decumulation_step(
                            investor=investor,
                            annual_returns=annual_returns,
                            target_alloc=target_alloc,
                            fixed_withdrawal_amount=fixed_withdrawal_amount,
                            current_cap=current_cap,
                            current_floor=current_floor,
                        )
                        # Annual rebalancing for the annual path
                        investor.rebalance(target_alloc)


                investor.age += 1

                if track_paths:
                    trial_path.append(investor.total_portfolio_value)
                    trial_withdrawals.append(withdrawal_amount_this_year)

            end_path = getattr(market, "end_path", None)
            if callable(end_path):
                end_path()

            terminal_wealths.append(investor.total_portfolio_value)
            if track_paths:
                paths.append(trial_path)
                withdrawal_paths.append(trial_withdrawals)

        if track_paths:
            return SimulationResult(
                terminal_wealths=terminal_wealths,
                paths=paths,
                withdrawal_paths=withdrawal_paths,
            )
        return terminal_wealths
