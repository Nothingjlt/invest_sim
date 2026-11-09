import math
from collections.abc import Mapping as MappingABC
from numbers import Real
from typing import Iterable, Mapping

DOMESTIC_STOCK = "Domestic Stock"
INTERNATIONAL_STOCK = "International Stock"
BONDS = "Bonds"
BILLS = "Bills"


def finite_number(value: float, *, context: str) -> float:
    """Require a real, finite numeric input (booleans are not amounts/rates)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be a finite number.")
    return number


def validate_allocation(
    allocation: Mapping[str, float], *, context: str = "Allocation"
) -> dict[str, float]:
    """Validate long-only weights and normalize accepted rounding error.

    Components sharing a label must be added before this boundary. Normalizing
    sums within 1e-6 of one prevents rounding error from multiplying wealth on
    repeated contributions or rebalances. The caller's mapping is never changed.
    """
    if not isinstance(allocation, Mapping):
        raise ValueError(f"{context} must be a mapping of asset weights.")
    if not allocation:
        raise ValueError(f"{context} weights must sum to 1.0 (got 0.0).")
    weights = {}
    for asset, weight in allocation.items():
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"{context}: asset labels must be nonempty strings.")
        weight = finite_number(weight, context=f"{context} weight for {asset}")
        if weight < 0:
            raise ValueError(f"{context}: weights must be nonnegative.")
        weights[asset] = weight
    try:
        total = math.fsum(weights.values())
    except OverflowError as exc:
        raise ValueError(f"{context} weights must sum to 1.0.") from exc
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{context} weights must sum to 1.0 (got {total}).")
    return {asset: weight / total for asset, weight in weights.items()}


def require_return_series(
    required_assets: Iterable[str],
    returns: Mapping[str, float],
    *,
    context: str,
) -> None:
    missing = sorted(set(required_assets) - set(returns))
    if missing:
        raise ValueError(
            f"{context}: missing return series for {', '.join(missing)}"
        )


def validate_market_returns(
    returns: Mapping[str, object], *, context: str = "Market returns"
) -> dict[str, float]:
    """Validate and normalize one market observation.

    Market engines return simple returns, optionally accompanied by an
    ``Inflation`` metadata value.  Missing inflation and an explicit
    ``Inflation=None`` both mean zero inflation.  Every other supplied value
    must be a finite number, and no simple return may be below ``-1``.  The
    returned copy contains floats, so annual and monthly simulation paths use
    the same validated values and never perform arithmetic on raw metadata.
    """
    if not isinstance(returns, MappingABC):
        raise ValueError(f"{context} must be a mapping of market returns.")

    validated: dict[str, float] = {}
    for asset, value in returns.items():
        if asset == "Inflation" and value is None:
            validated[asset] = 0.0
            continue

        number = finite_number(value, context=f"{context} value for {asset}")
        if number < -1.0:
            raise ValueError(
                f"{context} value for {asset} must be at least -1."
            )
        validated[asset] = number

    return validated
