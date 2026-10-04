from typing import Iterable, Mapping

DOMESTIC_STOCK = "Domestic Stock"
INTERNATIONAL_STOCK = "International Stock"
BONDS = "Bonds"
BILLS = "Bills"


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
