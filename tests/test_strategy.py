import pytest
from src.strategy import (
    FixedAllocationStrategy,
    GlidePathStrategy,
    PaperOptimalStrategy,
    PaperTDFStrategy,
    BalancedStrategy,
    Strategy,
    WorldEquityStrategy,
)


def test_fixed_allocation_strategy():
    """Verify that a fixed strategy remains constant across all ages."""
    target = {"Stocks": 1.0}
    strategy = FixedAllocationStrategy(target)

    assert strategy.get_allocation(25) == target
    assert strategy.get_allocation(65) == target
    assert strategy.get_allocation(100) == target


def test_glide_path_strategy_linear_shift():
    """
    Verify the linear shift from 90% stocks at age 25 to 30% stocks at age 65.
    (60% drop over 40 years = 1.5% drop per year)
    """
    strategy = GlidePathStrategy(
        start_age=25, retire_age=65, start_equity=0.90, end_equity=0.30
    )

    # Boundary tests
    assert strategy.get_allocation(25)["Stocks"] == pytest.approx(0.90)
    assert strategy.get_allocation(65)["Stocks"] == pytest.approx(0.30)
    assert strategy.get_allocation(20)["Stocks"] == pytest.approx(
        0.90
    )  # Before start age
    assert strategy.get_allocation(70)["Stocks"] == pytest.approx(
        0.30
    )  # After retirement age

    # Midpoint test (Age 45 should be exactly midway between 0.90 and 0.30 = 0.60)
    assert strategy.get_allocation(45)["Stocks"] == pytest.approx(0.60)

    # Check bonds
    assert strategy.get_allocation(45)["Bonds"] == pytest.approx(0.40)


def test_glide_path_granular_assets():
    """Verify that GlidePathStrategy handles multiple equity and bond assets."""
    equity_assets = {"Domestic": 0.5, "International": 0.5}
    bond_assets = {"Gov Bonds": 0.7, "Corp Bonds": 0.3}

    strategy = GlidePathStrategy(
        start_age=25,
        retire_age=65,
        start_equity=1.0,  # 100% equity at start
        end_equity=0.0,  # 0% equity at retirement
        equity_assets=equity_assets,
        bond_assets=bond_assets,
    )

    # Age 25: 100% Equity (50/50 Domestic/Intl)
    alloc_25 = strategy.get_allocation(25)
    assert alloc_25["Domestic"] == pytest.approx(0.5)
    assert alloc_25["International"] == pytest.approx(0.5)
    assert alloc_25.get("Gov Bonds", 0) == 0

    # Age 65: 100% Bonds (70/30 Gov/Corp)
    alloc_65 = strategy.get_allocation(65)
    assert alloc_65["Gov Bonds"] == pytest.approx(0.7)
    assert alloc_65["Corp Bonds"] == pytest.approx(0.3)
    assert alloc_65.get("Domestic", 0) == 0

    # Age 45: 50% Equity, 50% Bonds
    # Equity part: 0.5 * 0.5 = 0.25 Domestic, 0.25 International
    # Bond part: 0.5 * 0.7 = 0.35 Gov, 0.5 * 0.3 = 0.15 Corp
    alloc_45 = strategy.get_allocation(45)
    assert alloc_45["Domestic"] == pytest.approx(0.25)
    assert alloc_45["International"] == pytest.approx(0.25)
    assert alloc_45["Gov Bonds"] == pytest.approx(0.35)
    assert alloc_45["Corp Bonds"] == pytest.approx(0.15)


def test_glide_path_invalid_weights():
    """Confirm ValueError is raised if asset weights do not sum to 1.0."""
    with pytest.raises(ValueError, match="Equity asset weights must sum to 1.0"):
        GlidePathStrategy(25, 65, equity_assets={"A": 0.5})

    with pytest.raises(ValueError, match="Bond asset weights must sum to 1.0"):
        GlidePathStrategy(25, 65, bond_assets={"B": 1.1})


def test_strategy_allocations_sum_to_one_and_non_negative():
    strategies = [
        GlidePathStrategy(
            start_age=25,
            retire_age=65,
            start_equity=0.90,
            end_equity=0.30,
            equity_assets={"Domestic": 0.5, "International": 0.5},
            bond_assets={"Bonds": 1.0},
        ),
        PaperOptimalStrategy(
            retire_age=65,
            dom_label="Domestic Stock",
            intl_assets={"International Stock": 1.0},
            bills_label="Bills",
        ),
        PaperTDFStrategy(
            start_age=25,
            retire_age=65,
            dom_label="Domestic Stock",
            intl_assets={"International Stock": 1.0},
            bond_label="Bonds",
            bills_label="Bills",
        ),
        BalancedStrategy(),
    ]

    ages = [25, 30, 40, 50, 65, 67, 70, 80]
    for strategy in strategies:
        for age in ages:
            allocation = strategy.get_allocation(age)
            assert sum(allocation.values()) == pytest.approx(1.0)
            assert all(value >= 0.0 for value in allocation.values())


def test_strategy_negative_age_raises_error():
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    with pytest.raises(ValueError, match="Age must be positive"):
        strategy.get_allocation(-1.0)


INVALID_ALLOCATIONS = [
    {}, {"A": 2.0}, {"A": 0.4}, {"A": -0.1, "B": 1.1},
    {"A": float("nan")}, {"A": float("inf")}, {"A": -float("inf")},
    {"A": True}, {"A": "1"}, {"": 1.0}, {None: 1.0},
]


@pytest.mark.parametrize("allocation", INVALID_ALLOCATIONS)
def test_fixed_allocation_rejects_malformed_weights(allocation):
    with pytest.raises(ValueError):
        FixedAllocationStrategy(allocation)


@pytest.mark.parametrize("allocation", INVALID_ALLOCATIONS)
def test_custom_strategy_validates_allocation_at_public_boundary(allocation):
    class CustomStrategy(Strategy):
        def _get_allocation(self, age):
            return allocation

    strategy = CustomStrategy()
    with pytest.raises(ValueError):
        strategy.get_allocation(25)
    with pytest.raises(ValueError):
        strategy.resolve_allocation_for_market(allocation, ["A", "B"])


@pytest.mark.parametrize("age", [float("nan"), float("inf"), True, "25"])
def test_strategy_requires_finite_numeric_age(age):
    with pytest.raises(ValueError):
        FixedAllocationStrategy({"A": 1.0}).get_allocation(age)


@pytest.mark.parametrize("factory", [
    lambda weights: GlidePathStrategy(25, 65, equity_assets=weights),
    lambda weights: GlidePathStrategy(25, 65, bond_assets=weights),
    lambda weights: PaperOptimalStrategy(intl_assets=weights),
    lambda weights: PaperTDFStrategy(intl_assets=weights),
    lambda weights: WorldEquityStrategy({"Region": weights}),
])
@pytest.mark.parametrize("weights", [{"A": -0.1, "B": 1.1}, {"A": float("nan")}, {}])
def test_constituent_validation_cannot_hide_bad_weights(factory, weights):
    with pytest.raises(ValueError):
        factory(weights)


@pytest.mark.parametrize("factory", [
    lambda: BalancedStrategy("A", "A"),
    lambda: GlidePathStrategy(25, 65, equity_assets={"A": 1.0}, bond_assets={"A": 1.0}),
    lambda: PaperOptimalStrategy(dom_label="A", intl_assets={"A": 1.0}),
    lambda: PaperTDFStrategy(dom_label="A", intl_assets={"A": 1.0}, bond_label="A", bills_label="A"),
])
def test_shared_component_labels_add_weights(factory):
    for age in [25, 45, 65, 100]:
        assert factory().get_allocation(age) == {"A": pytest.approx(1.0)}


@pytest.mark.parametrize("strategy_type", [GlidePathStrategy, PaperTDFStrategy])
@pytest.mark.parametrize("start_age,retire_age", [(65, 25), (25, 25), (-1, 65), (25, float("nan"))])
def test_glide_path_requires_valid_timeline(strategy_type, start_age, retire_age):
    with pytest.raises(ValueError):
        strategy_type(start_age, retire_age)


@pytest.mark.parametrize("parameter,value", [("start_equity", -0.1), ("end_equity", 1.1), ("start_equity", float("nan"))])
def test_glide_path_rejects_invalid_equity_rates(parameter, value):
    with pytest.raises(ValueError):
        GlidePathStrategy(25, 65, **{parameter: value})


def test_fixed_strategy_copies_input_and_checks_later_mutation():
    source = {"A": 1.0}
    strategy = FixedAllocationStrategy(source)
    source["A"] = 2.0
    returned = strategy.get_allocation(25)
    returned["A"] = 3.0
    assert strategy.get_allocation(25) == {"A": 1.0}
    strategy.target_allocation["A"] = 2.0
    with pytest.raises(ValueError):
        strategy.get_allocation(25)


def test_constituent_mutation_rechecked_even_when_labels_cancel_negative_weight():
    strategy = PaperOptimalStrategy(dom_label="A", intl_assets={"A": 0.5, "B": 0.5})
    strategy.intl_assets.update({"A": -0.1, "B": 1.1})
    with pytest.raises(ValueError):
        strategy.get_allocation(25)


@pytest.mark.parametrize("strategy_type", [PaperOptimalStrategy, PaperTDFStrategy])
def test_paper_resolution_rejects_invalid_input_allocation(strategy_type):
    with pytest.raises(ValueError):
        strategy_type().resolve_allocation_for_market({"A": 2.0}, ["A"])
