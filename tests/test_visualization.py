import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import SimulationConfig, MarketConfig
from src.simulator import Simulator
from src.strategy import FixedAllocationStrategy
from src.visualization import (
    plot_percentile_curve,
    plot_ruin_probability,
    plot_summary_metrics,
    plot_terminal_wealth_histogram,
    plot_simulation_paths,
    plot_withdrawal_paths,
    plot_withdrawal_comparison,
)


def test_terminal_wealth_histogram():
    fig, ax = plot_terminal_wealth_histogram([100.0, 200.0, 300.0, 400.0], strategy_name="Test")
    assert fig is not None
    assert ax.get_title() == "Terminal Wealth Distribution — Test"
    assert ax.get_xlabel() == "Terminal Wealth ($)"
    assert ax.get_ylabel() == "Number of Trials"
    assert ax.get_xscale() in {"log", "symlog"}
    fig.tight_layout()
    plt.close(fig)


def test_percentile_curve():
    values = [100.0, 200.0, 300.0, 400.0]
    fig, ax = plot_percentile_curve(values, strategy_name="Test")
    assert "Percentile" in ax.get_xlabel()
    assert ax.get_ylabel() == "Terminal Wealth ($)"
    assert "Test" in ax.get_title()
    assert ax.get_yscale() in {"log", "symlog"}
    plt.close(fig)


def test_ruin_probability():
    values_optimal = [0.0, 100.0, 200.0]
    values_tdf = [50.0, 100.0, 0.0]
    fig, ax = plot_ruin_probability(
        {
            "Optimal": values_optimal,
            "TDF": values_tdf,
        }
    )
    assert ax.get_ylabel() == "Probability"
    assert len(ax.patches) == 2
    assert [label.get_text() for label in ax.get_xticklabels()] == ["Optimal", "TDF"]
    plt.close(fig)


def test_summary_metrics_bar_chart():
    values_optimal = [100.0, 200.0, 300.0, 400.0]
    values_tdf = [120.0, 220.0, 320.0, 420.0]
    fig, ax = plot_summary_metrics(
        {
            "Optimal": values_optimal,
            "TDF": values_tdf,
        }
    )
    assert ax.get_title() == "Summary Wealth Metrics"
    assert ax.get_ylabel() == "Terminal Wealth ($)"
    assert len(ax.patches) == 8
    assert ax.get_yscale() in {"log", "symlog"}
    plt.close(fig)


def test_plot_save(tmp_path: Path):
    values = [100.0, 200.0, 300.0, 400.0]
    output = tmp_path / "hist.png"
    fig, ax = plot_terminal_wealth_histogram(values, strategy_name="Test", save_path=str(output))
    assert output.exists()
    plt.close(fig)


def test_simulation_paths_plot():
    paths = [
        [100.0, 150.0, 140.0, 180.0],
        [100.0, 120.0, 130.0, 160.0],
    ]
    fig, ax = plot_simulation_paths(paths, strategy_name="Test", num_paths=2, starting_age=30)
    assert "Simulation Path Trajectories" in ax.get_title()
    assert ax.get_xlabel() == "Age"
    lines = ax.get_lines()
    assert list(lines[0].get_xdata()) == [30, 31, 32, 33]
    plt.close(fig)


def test_simulation_paths_from_simulator():
    """Integration: SimulationResult.paths flows correctly into plot_simulation_paths."""
    config = SimulationConfig(
        starting_age=25,
        retirement_age=30,
        end_age=32,
        initial_salary=10_000.0,
        salary_growth_rate=0.0,
        savings_rate=0.10,
        withdrawal_rate=0.04,
        markets=[MarketConfig(name="Stocks", expected_return=0.0, volatility=0.0, weight=1.0)],
    )
    sim = Simulator(config)
    strategy = FixedAllocationStrategy({"Stocks": 1.0})
    result = sim.run_stochastic(strategy, num_trials=10, track_paths=True)

    fig, ax = plot_simulation_paths(
        result.paths,
        strategy_name="Integration Test",
        num_paths=10,
        starting_age=config.starting_age,
    )
    assert "Simulation Path Trajectories" in ax.get_title()
    assert "Integration Test" in ax.get_title()
    assert ax.get_xlabel() == "Age"
    assert ax.get_ylabel() == "Portfolio Value ($)"
    # 10 trials × 1 line each
    assert len(ax.get_lines()) == 10
    assert ax.get_lines()[0].get_xdata()[0] == 25
    plt.close(fig)


def test_plot_withdrawal_paths():
    paths = [
        [0.0, 0.0, 100.0, 110.0],
        [0.0, 0.0, 90.0, 80.0],
    ]
    fig, ax = plot_withdrawal_paths(paths, strategy_name="Test Strategy", num_paths=2, starting_age=25)
    assert "Annual Portfolio Withdrawal Paths" in ax.get_title()
    assert "Test Strategy" in ax.get_title()
    assert ax.get_xlabel() == "Age"
    assert ax.get_ylabel() == "Withdrawal Amount ($)"
    assert len(ax.get_lines()) == 2
    assert list(ax.get_lines()[0].get_xdata()) == [25, 26, 27, 28]
    plt.close(fig)


def test_plot_withdrawal_comparison():
    strategy_withdrawals = {
        "Strategy A": [
            [0.0, 100.0, 100.0],
            [0.0, 120.0, 120.0],
        ],
        "Strategy B": [
            [0.0, 50.0, 50.0],
            [0.0, 60.0, 60.0],
        ],
    }
    fig, ax = plot_withdrawal_comparison(strategy_withdrawals, starting_age=30, percentile=50.0)
    assert "Median Portfolio Withdrawal Over Time" in ax.get_title()
    assert ax.get_xlabel() == "Age"
    assert ax.get_ylabel() == "Withdrawal Amount ($)"
    assert len(ax.get_lines()) == 2
    labels = [line.get_label() for line in ax.get_lines()]
    assert "Strategy A" in labels
    assert "Strategy B" in labels
    line_a = [line for line in ax.get_lines() if line.get_label() == "Strategy A"][0]
    line_b = [line for line in ax.get_lines() if line.get_label() == "Strategy B"][0]
    assert list(line_a.get_ydata()) == [0.0, 110.0, 110.0]
    assert list(line_b.get_ydata()) == [0.0, 55.0, 55.0]
    plt.close(fig)

