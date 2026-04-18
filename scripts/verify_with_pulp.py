from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.simplex_solution_to_transfer_plan import decode_simplex_solution_to_transfer_plan  # noqa: E402
from src.simplex.lp_standard_form import transportation_instance_to_standard_form  # noqa: E402
from src.simplex.primal_simplex_solver import PrimalSimplexSolver  # noqa: E402
from src.transport.transportation_instance_builder import build_transportation_instances  # noqa: E402


def run_pulp_verification() -> None:
    try:
        from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum
    except ImportError:
        print("PuLP is not installed. Skipping optional verification.")
        return

    excess_df = pd.DataFrame(
        [
            {"store_id": 1, "product_id": 777, "excess_units": 20},
            {"store_id": 2, "product_id": 777, "excess_units": 30},
        ]
    )
    needed_df = pd.DataFrame(
        [
            {"store_id": 3, "product_id": 777, "needed_units": 25},
            {"store_id": 4, "product_id": 777, "needed_units": 25},
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
    instance = instances[0]

    # Our simplex objective
    lp = transportation_instance_to_standard_form(instance)
    solver = PrimalSimplexSolver(max_iterations=500, tolerance=1e-9, use_bland=True)
    result = solver.solve(lp)
    if result.status != "OPTIMAL":
        raise RuntimeError(f"Custom simplex failed with status {result.status}")

    ours_plan = decode_simplex_solution_to_transfer_plan(
        simplex_result=result,
        standard_form_lp=lp,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )
    ours_obj = float(ours_plan["transport_cost"].sum())

    # PuLP objective
    model = LpProblem("transport_verify", LpMinimize)

    sources = [1, 2]
    demands = [3, 4]
    supply = {1: 20, 2: 30}
    demand = {3: 25, 4: 25}

    x = {
        (i, j): LpVariable(f"x_{i}_{j}", lowBound=0)
        for i in sources
        for j in demands
    }

    model += lpSum(transport_cost_matrix.loc[i, j] * x[(i, j)] for i in sources for j in demands)

    for i in sources:
        model += lpSum(x[(i, j)] for j in demands) == supply[i]

    for j in demands:
        model += lpSum(x[(i, j)] for i in sources) == demand[j]

    model.solve(PULP_CBC_CMD(msg=False))

    if LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"PuLP did not return optimal status: {LpStatus[model.status]}")

    pulp_obj = float(model.objective.value())

    print(f"Our simplex objective: {ours_obj:.6f}")
    print(f"PuLP objective      : {pulp_obj:.6f}")
    print(f"Objective diff      : {abs(ours_obj - pulp_obj):.6f}")

    if abs(ours_obj - pulp_obj) > 1e-6:
        raise AssertionError("Objective mismatch between custom simplex and PuLP")

    print("Optional PuLP verification passed")


if __name__ == "__main__":
    run_pulp_verification()
