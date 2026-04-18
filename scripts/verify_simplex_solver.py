from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.simplex_solution_to_transfer_plan import (  # noqa: E402
    TRANSFER_PLAN_COLUMNS,
    decode_simplex_solution_to_transfer_plan,
    merge_transfer_plans,
)
from src.engine.analyzer import InventoryAnalyzer  # noqa: E402
from src.engine.genetic_algorithm import GeneticAlgorithmOptimizer  # noqa: E402
from src.engine.rule_based import RuleBasedOptimizer  # noqa: E402
from src.simplex.lp_standard_form import transportation_instance_to_standard_form  # noqa: E402
from src.simplex.primal_simplex_solver import PrimalSimplexSolver  # noqa: E402
from src.transport.transportation_instance_builder import build_transportation_instances  # noqa: E402


def _assert_transfer_schema(df: pd.DataFrame, label: str) -> None:
    missing = [col for col in TRANSFER_PLAN_COLUMNS if col not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing schema columns: {missing}")


def run_manual_2x2_case() -> None:
    """Hand-checkable 2x2 transportation case."""
    print("\n[TEST 1] Manual 2x2 case")

    # Supply: source 1=20, source 2=30
    excess_df = pd.DataFrame(
        [
            {"store_id": 1, "product_id": 999, "excess_units": 20},
            {"store_id": 2, "product_id": 999, "excess_units": 30},
        ]
    )

    # Demand: dest 3=25, dest 4=25
    needed_df = pd.DataFrame(
        [
            {"store_id": 3, "product_id": 999, "needed_units": 25},
            {"store_id": 4, "product_id": 999, "needed_units": 25},
        ]
    )

    labels = [1, 2, 3, 4]
    distance_matrix = pd.DataFrame(
        [
            [0, 5, 10, 12],
            [5, 0, 14, 8],
            [10, 14, 0, 6],
            [12, 8, 6, 0],
        ],
        index=labels,
        columns=labels,
    )

    # Cost design with known optimum = 85:
    # x13=20, x23=5, x24=25, x14=0 => 20*2 + 5*4 + 25*1 = 85
    transport_cost_matrix = pd.DataFrame(
        [
            [0, 0, 2, 3],
            [0, 0, 4, 1],
            [2, 4, 0, 0],
            [3, 1, 0, 0],
        ],
        index=labels,
        columns=labels,
    )

    instances = build_transportation_instances(
        excess_inventory=excess_df,
        needed_inventory=needed_df,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
        allow_self_transfer=False,
    )

    assert len(instances) == 1, "Expected exactly one manual product instance"

    instance = instances[0]
    lp = transportation_instance_to_standard_form(instance)

    solver = PrimalSimplexSolver(
        max_iterations=500,
        tolerance=1e-9,
        pivot_rule="dantzig",
        use_bland=True,
        keep_history=False,
    )
    result = solver.solve(lp)
    assert result.status == "OPTIMAL", f"Expected OPTIMAL, got {result.status}"

    plan = decode_simplex_solution_to_transfer_plan(
        simplex_result=result,
        standard_form_lp=lp,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )

    _assert_transfer_schema(plan, "Simplex manual plan")
    total_cost = float(plan["transport_cost"].sum()) if not plan.empty else 0.0

    if result.objective_value is None:
        raise AssertionError("Simplex returned OPTIMAL without objective value")

    assert abs(result.objective_value - 85.0) <= 1e-6, (
        f"Unexpected objective value {result.objective_value}, expected 85.0"
    )
    assert abs(total_cost - 85.0) <= 1e-6, (
        f"Unexpected transport cost {total_cost}, expected 85.0"
    )

    print("Manual 2x2 case passed")
    print(plan.to_string(index=False))


def run_inventory_based_case(data_dir: Path) -> None:
    """Inventory-based integration test using real project data."""
    print("\n[TEST 2] Inventory-based case")

    required_files = [
        "sales_data.csv",
        "inventory_data.csv",
        "stores.csv",
        "products.csv",
        "distance_matrix.csv",
        "transport_cost_matrix.csv",
    ]
    missing = [name for name in required_files if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files for integration test: {missing}")

    analyzer = InventoryAnalyzer()
    analyzer.load_data(
        sales_path=str(data_dir / "sales_data.csv"),
        inventory_path=str(data_dir / "inventory_data.csv"),
        stores_path=str(data_dir / "stores.csv"),
        products_path=str(data_dir / "products.csv"),
    )

    analyzer.analyze_sales_data()
    excess_df, needed_df = analyzer.identify_inventory_imbalances()

    distance_matrix = pd.read_csv(data_dir / "distance_matrix.csv", index_col=0)
    transport_cost_matrix = pd.read_csv(data_dir / "transport_cost_matrix.csv", index_col=0)
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)
    transport_cost_matrix.index = transport_cost_matrix.index.astype(int)
    transport_cost_matrix.columns = transport_cost_matrix.columns.astype(int)

    instances = build_transportation_instances(
        excess_inventory=excess_df,
        needed_inventory=needed_df,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
        allow_self_transfer=False,
    )

    solver = PrimalSimplexSolver(
        max_iterations=2000,
        tolerance=1e-9,
        pivot_rule="dantzig",
        use_bland=True,
        keep_history=False,
    )

    simplex_plans = []
    optimal_count = 0

    for instance in instances:
        lp = transportation_instance_to_standard_form(instance)
        result = solver.solve(lp)
        if result.status != "OPTIMAL":
            continue

        optimal_count += 1
        decoded = decode_simplex_solution_to_transfer_plan(
            simplex_result=result,
            standard_form_lp=lp,
            distance_matrix=distance_matrix,
            transport_cost_matrix=transport_cost_matrix,
        )
        if not decoded.empty:
            simplex_plans.append(decoded)

    simplex_plan = merge_transfer_plans(simplex_plans)
    _assert_transfer_schema(simplex_plan, "Simplex inventory plan")

    print(f"Instances built: {len(instances)}")
    print(f"Optimal simplex instances: {optimal_count}")
    print(f"Merged simplex transfers: {len(simplex_plan)}")

    # Compare schema compatibility with baseline solvers.
    rb = RuleBasedOptimizer(distance_matrix=distance_matrix, transport_cost_matrix=transport_cost_matrix)
    rb_plan = rb.optimize(excess_df, needed_df)
    _assert_transfer_schema(rb_plan if rb_plan is not None else pd.DataFrame(), "Rule-Based plan")

    ga = GeneticAlgorithmOptimizer(
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
        random_seed=2025,
    )
    ga_plan = ga.optimize(
        excess_df,
        needed_df,
        population_size=20,
        num_generations=5,
        crossover_prob=0.6,
        mutation_prob=0.3,
        tournament_size=3,
        verbose=False,
    )
    _assert_transfer_schema(ga_plan if ga_plan is not None else pd.DataFrame(), "GA plan")

    print("Schema compatibility check with Rule-Based and GA passed")


def main() -> None:
    data_dir = PROJECT_ROOT / "data"

    run_manual_2x2_case()
    run_inventory_based_case(data_dir)

    print("\nAll simplex verification checks passed")


if __name__ == "__main__":
    main()
