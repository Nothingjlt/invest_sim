import random
from typing import Dict, List, NamedTuple, Optional, Union
from src.config import SimulationConfig
from src.investor import Investor
from src.market import Market, SyntheticMarket, BootstrapMarket
from src.strategy import Strategy, FixedAllocationStrategy


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
    """

    terminal_wealths: List[float]
    paths: List[List[float]]
    withdrawal_paths: List[List[float]]


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

        Returns 0.0 if the market engine does not provide an 'Inflation' key
        (e.g. SyntheticMarket or a CSV without an Inflation column), so all
        inflation-sensitive trackers simply stay flat that year.
        """
        value = annual_returns.get("Inflation")
        return value if value is not None else 0.0

    def run_stochastic(
        self,
        strategy: Strategy,
        num_trials: int = 1000,
        market_engine: Market | None = None,
        track_paths: bool = False,
    ) -> Union[List[float], SimulationResult]:
        """
        Runs multiple lifecycle simulations with strategy-based rebalancing.
        If market_engine is not provided, defaults to SyntheticMarket using config.

        When track_paths=True, returns a SimulationResult(terminal_wealths, paths, withdrawal_paths)
        named-tuple. Each entry in ``paths`` is the year-by-year total portfolio
        value for one trial (starting snapshot + one value per simulated year).
        When track_paths=False (default), returns a plain List[float] of terminal
        wealth values for backward compatibility.
        """
        terminal_wealths: List[float] = []
        paths: List[List[float]] = []
        withdrawal_paths: List[List[float]] = []

        for _ in range(num_trials):
            # Use provided engine or default to Synthetic
            market = (
                market_engine if market_engine else SyntheticMarket(self.config.markets)
            )

            # If bootstrap, start a new path
            if isinstance(market, BootstrapMarket):
                market.start_new_path()

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

                # 2. Market Growth (applied to existing holdings)
                annual_returns = market.get_annual_returns()
                investor.apply_returns(annual_returns)

                withdrawal_amount_this_year = 0.0

                # 3. Income/Savings or Withdrawal
                if investor.age < self.config.retirement_age:
                    investor.earn_and_save(self.config.savings_rate, target_alloc)
                    investor.grow_salary(self.config.salary_growth_rate)
                else:
                    # --- Decumulation Phase ---

                    # Capture fixed withdrawal base at the start of retirement
                    if investor.age == self.config.retirement_age:
                        fixed_withdrawal_amount = (
                            investor.total_portfolio_value * self.config.withdrawal_rate
                        )

                    # a) Compute raw withdrawal target
                    if self.config.withdrawal_strategy == "fixed_real":
                        raw_withdrawal = fixed_withdrawal_amount
                    else:  # variable_pct
                        raw_withdrawal = (
                            investor.total_portfolio_value * self.config.withdrawal_rate
                        )

                    # b) Apply cap (before social security offset)
                    if current_cap is not None:
                        raw_withdrawal = min(raw_withdrawal, current_cap)

                    # c) Social security offset
                    net_from_portfolio = max(
                        0.0, raw_withdrawal - self.config.social_security_benefit
                    )

                    # d) Apply floor: ensure minimum real expenditure is met from portfolio
                    if current_floor is not None:
                        floor_from_portfolio = max(
                            0.0,
                            current_floor - self.config.social_security_benefit,
                        )
                        net_from_portfolio = max(net_from_portfolio, floor_from_portfolio)

                    # e) Execute withdrawal
                    actual_withdrawn = investor.withdraw(net_from_portfolio, target_alloc)
                    withdrawal_amount_this_year = actual_withdrawn

                    # f) Step inflation trackers at end of year
                    inflation = self._get_inflation(annual_returns)
                    if self.config.withdrawal_inflation_adjusted:
                        fixed_withdrawal_amount *= (1 + inflation)
                    if self.config.withdrawal_cap_inflation_adjusted and current_cap is not None:
                        current_cap *= (1 + inflation)
                    if self.config.withdrawal_floor_inflation_adjusted and current_floor is not None:
                        current_floor *= (1 + inflation)

                # 4. Annual Rebalancing
                investor.rebalance(target_alloc)

                investor.age += 1

                if track_paths:
                    trial_path.append(investor.total_portfolio_value)
                    trial_withdrawals.append(withdrawal_amount_this_year)

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
