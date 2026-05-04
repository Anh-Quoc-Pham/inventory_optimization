"""
MODI (Modified Distribution Method) Optimizer for Transportation Problems.

Drop-in companion to RuleBasedOptimizer / GeneticAlgorithmOptimizer:
  - __init__(distance_matrix, transport_cost_matrix, ...)
  - optimize(excess_inventory, needed_inventory) → pd.DataFrame  (transfer_plan schema)
  - load_matrices(distance_path, cost_path)

Internally uses the same TransportationInstance / build_transportation_instances
pipeline that PrimalSimplexSolver uses, so the two methods are directly comparable
in the benchmark.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.transport.transportation_instance_builder import (
    TransportationInstance,
    build_transportation_instances,
)
from src.utils.logger import get_optimization_logger

_logger = logging.getLogger(__name__)

# Status constants (mirrors feasibility_and_unbounded_checks.py) 
STATUS_OPTIMAL = "OPTIMAL"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_UNBOUNDED = "UNBOUNDED"
STATUS_ITERATION_LIMIT = "ITERATION_LIMIT"
STATUS_NUMERICAL_ISSUE = "NUMERICAL_ISSUE"

# Transfer-plan column contract (must match TRANSFER_PLAN_COLUMNS in adapters)
TRANSFER_PLAN_COLUMNS = [
    "from_store_id",
    "to_store_id",
    "product_id",
    "units",
    "distance_km",
    "transport_cost",
]

Cell = Tuple[int, int]


# Internal BFS dataclass

@dataclass
class _BFS:
    allocation: np.ndarray
    basic_cells: List[Cell]
    cost_matrix: np.ndarray
    supply: np.ndarray
    demand: np.ndarray

    @property
    def m(self) -> int:
        return self.allocation.shape[0]

    @property
    def n(self) -> int:
        return self.allocation.shape[1]

    def objective(self) -> float:
        return float(np.sum(self.allocation * self.cost_matrix))


# SimplexResult-compatible output (used by adapters / benchmark)

@dataclass
class MODIResult:
    """MODI solver output – same public fields as SimplexResult."""

    status: str
    objective_value: Optional[float]
    variable_values: np.ndarray
    variable_values_by_name: Dict[str, float]
    basis_indices: List[int]
    iterations: int
    phase_one_iterations: int = 0
    phase_two_iterations: int = 0
    tableau_history: Optional[List[Dict[str, Any]]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)


# Northwest-corner initial BFS

def _northwest_corner_bfs(
    cost: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    tol: float = 1e-9,
) -> _BFS:
    m, n = cost.shape
    sup = supply.astype(float).copy()
    dem = demand.astype(float).copy()
    allocation = np.zeros((m, n), dtype=float)
    basic_cells: List[Cell] = []

    i, j = 0, 0
    while i < m and j < n:
        amount = min(sup[i], dem[j])
        allocation[i, j] = amount
        basic_cells.append((i, j))
        sup[i] -= amount
        dem[j] -= amount

        row_done = sup[i] <= tol
        col_done = dem[j] <= tol

        if row_done and col_done:
            if i + 1 < m and j + 1 < n:
                basic_cells.append((i + 1, j))
                allocation[i + 1, j] = 0.0
            i += 1
            j += 1
        elif row_done:
            i += 1
        else:
            j += 1

    seen: Set[Cell] = set()
    deduped: List[Cell] = []
    for cell in basic_cells:
        if cell in seen:
            continue
        seen.add(cell)
        deduped.append(cell)

    target = m + n - 1
    if len(deduped) < target:
        for row in range(m):
            for col in range(n):
                if (row, col) not in seen and len(deduped) < target:
                    deduped.append((row, col))
                    seen.add((row, col))
                    allocation[row, col] = 0.0

    return _BFS(
        allocation=allocation,
        basic_cells=deduped[:target],
        cost_matrix=cost,
        supply=supply.astype(float).copy(),
        demand=demand.astype(float).copy(),
    )


# MODI dual variables

def _compute_dual_variables(bfs: _BFS) -> Tuple[np.ndarray, np.ndarray]:
    m, n = bfs.m, bfs.n
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0

    adj_row: Dict[int, List[int]] = {i: [] for i in range(m)}
    adj_col: Dict[int, List[int]] = {j: [] for j in range(n)}
    for (i, j) in bfs.basic_cells:
        adj_row[i].append(j)
        adj_col[j].append(i)

    queue: List[Tuple[str, int]] = [("row", 0)]
    visited_rows: Set[int] = {0}
    visited_cols: Set[int] = set()

    while queue:
        kind, idx = queue.pop(0)
        if kind == "row":
            for j in adj_row[idx]:
                if j not in visited_cols:
                    v[j] = bfs.cost_matrix[idx, j] - u[idx]
                    visited_cols.add(j)
                    queue.append(("col", j))
        else:
            for i in adj_col[idx]:
                if i not in visited_rows:
                    u[i] = bfs.cost_matrix[i, idx] - v[idx]
                    visited_rows.add(i)
                    queue.append(("row", i))

    u = np.where(np.isnan(u), 0.0, u)
    v = np.where(np.isnan(v), 0.0, v)
    return u, v


# Reduced costs

def _reduced_costs(bfs: _BFS, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = bfs.cost_matrix - u[:, np.newaxis] - v[np.newaxis, :]
    for (i, j) in bfs.basic_cells:
        d[i, j] = 0.0
    return d


# Entering cell selection

def _select_entering_cell(
    d: np.ndarray,
    bfs: _BFS,
    tol: float = 1e-9,
) -> Optional[Cell]:
    basic_set: Set[Cell] = set(bfs.basic_cells)
    best_cell: Optional[Cell] = None
    best_val = -tol
    for i in range(bfs.m):
        for j in range(bfs.n):
            if (i, j) in basic_set:
                continue
            if d[i, j] < best_val:
                best_val = d[i, j]
                best_cell = (i, j)
    return best_cell


# Theta-loop detection

def _find_loop(entering: Cell, basic_cells: List[Cell]) -> List[Cell]:
    basic_set: Set[Cell] = set(basic_cells)
    all_cells = basic_cells + [entering]
    row_to_cells: Dict[int, List[Cell]] = {}
    col_to_cells: Dict[int, List[Cell]] = {}
    for cell in all_cells:
        row_to_cells.setdefault(cell[0], []).append(cell)
        col_to_cells.setdefault(cell[1], []).append(cell)

    stack: List[Tuple[Cell, List[Cell], str, Set[Cell]]] = [
        (entering, [entering], "row", {entering})
    ]

    while stack:
        current, path, next_dir, visited = stack.pop()
        neighbors = (
            row_to_cells.get(current[0], [])
            if next_dir == "row"
            else col_to_cells.get(current[1], [])
        )
        for nxt in neighbors:
            if nxt == current:
                continue
            if nxt == entering and len(path) >= 4:
                return path
            if nxt in visited:
                continue
            if nxt != entering and nxt not in basic_set:
                continue
            stack.append((nxt, path + [nxt], "col" if next_dir == "row" else "row", visited | {nxt}))

    raise RuntimeError(
        f"No loop found for entering cell {entering}. "
        "This indicates a degenerate or infeasible transportation basis."
    )


# Loop update

def _apply_loop_update(
    bfs: _BFS,
    loop: List[Cell],
    tol: float = 1e-9,
) -> Tuple[_BFS, Cell]:
    minus_cells = loop[1::2]
    theta = min(bfs.allocation[r, c] for (r, c) in minus_cells)

    new_alloc = bfs.allocation.copy()
    for k, (r, c) in enumerate(loop):
        if k % 2 == 0:
            new_alloc[r, c] += theta
        else:
            new_alloc[r, c] -= theta

    leaving: Cell = min(
        (cell for cell in minus_cells if abs(new_alloc[cell[0], cell[1]]) <= tol),
        default=minus_cells[0],
    )

    entering = loop[0]
    new_basic = [c for c in bfs.basic_cells if c != leaving]
    new_basic.append(entering)

    return (
        _BFS(
            allocation=new_alloc,
            basic_cells=new_basic,
            cost_matrix=bfs.cost_matrix,
            supply=bfs.supply,
            demand=bfs.demand,
        ),
        leaving,
    )


# Core MODI solver (instance-level)

def _solve_instance(
    instance: TransportationInstance,
    max_iterations: int,
    tolerance: float,
    keep_history: bool,
) -> MODIResult:
    source_ids = list(instance.source_store_ids)
    demand_ids = list(instance.demand_store_ids)
    m, n = len(source_ids), len(demand_ids)
    product_id = instance.product_id

    def _empty() -> MODIResult:
        return MODIResult(
            status=STATUS_OPTIMAL,
            objective_value=0.0,
            variable_values=np.zeros(m * n, dtype=float),
            variable_values_by_name={},
            basis_indices=[],
            iterations=0,
            diagnostics={"algorithm": "MODI", "product_id": product_id},
            messages=["Empty problem."],
        )

    if m == 0 or n == 0:
        return _empty()

    cost = np.asarray(instance.cost_matrix, dtype=float)
    supply = np.asarray(instance.supply_vector, dtype=float)
    demand = np.asarray(instance.demand_vector, dtype=float)

    total_supply = float(supply.sum())
    total_demand = float(demand.sum())
    if abs(total_supply - total_demand) > tolerance * max(1.0, total_supply):
        return MODIResult(
            status=STATUS_INFEASIBLE,
            objective_value=None,
            variable_values=np.zeros(m * n, dtype=float),
            variable_values_by_name={},
            basis_indices=[],
            iterations=0,
            diagnostics={"algorithm": "MODI", "product_id": product_id},
            messages=[
                f"Unbalanced: supply={total_supply:.6g}, demand={total_demand:.6g}."
            ],
        )

    bfs = _northwest_corner_bfs(cost, supply, demand, tol=tolerance)
    if not np.isfinite(bfs.allocation).all():
        return MODIResult(
            status=STATUS_NUMERICAL_ISSUE,
            objective_value=None,
            variable_values=np.zeros(m * n, dtype=float),
            variable_values_by_name={},
            basis_indices=[],
            iterations=0,
            diagnostics={"algorithm": "MODI", "product_id": product_id},
            messages=["Non-finite initial allocation."],
        )

    history: Optional[List[Dict[str, Any]]] = [] if keep_history else None
    messages: List[str] = []

    for iteration in range(1, max_iterations + 1):
        u, v = _compute_dual_variables(bfs)
        d = _reduced_costs(bfs, u, v)

        if keep_history and history is not None:
            history.append(
                {
                    "iteration": iteration - 1,
                    "objective_value": bfs.objective(),
                    "basic_cells": list(bfs.basic_cells),
                }
            )

        entering = _select_entering_cell(d, bfs, tol=tolerance)
        if entering is None:
            msg = f"Optimal after {iteration - 1} iteration(s)."
            messages.append(msg)
            variable_values = np.zeros(m * n, dtype=float)
            for i in range(m):
                for j in range(n):
                    variable_values[i * n + j] = bfs.allocation[i, j]
            var_by_name: Dict[str, float] = {}
            for i, src in enumerate(source_ids):
                for j, dem in enumerate(demand_ids):
                    name = f"x_p{product_id}_s{str(src).replace(' ','_')}_d{str(dem).replace(' ','_')}"
                    var_by_name[name] = float(bfs.allocation[i, j])
            basis_indices = sorted(r * n + c for (r, c) in bfs.basic_cells)
            obj = bfs.objective()
            return MODIResult(
                status=STATUS_OPTIMAL,
                objective_value=obj,
                variable_values=variable_values,
                variable_values_by_name=var_by_name,
                basis_indices=basis_indices,
                iterations=iteration - 1,
                phase_two_iterations=iteration - 1,
                tableau_history=history,
                diagnostics={
                    "algorithm": "MODI",
                    "product_id": product_id,
                    "iterations": iteration - 1,
                    "objective_value": obj,
                    "dummy_flows": [],
                    "dummy_flow_count": 0,
                },
                messages=messages,
            )

        try:
            loop = _find_loop(entering, bfs.basic_cells)
        except RuntimeError as exc:
            return MODIResult(
                status=STATUS_NUMERICAL_ISSUE,
                objective_value=None,
                variable_values=np.zeros(m * n, dtype=float),
                variable_values_by_name={},
                basis_indices=[],
                iterations=iteration,
                diagnostics={"algorithm": "MODI", "product_id": product_id},
                messages=[str(exc)],
            )

        bfs, _ = _apply_loop_update(bfs, loop, tol=tolerance)

    messages.append(f"Iteration limit ({max_iterations}) reached.")
    variable_values = np.zeros(m * n, dtype=float)
    for i in range(m):
        for j in range(n):
            variable_values[i * n + j] = bfs.allocation[i, j]
    return MODIResult(
        status=STATUS_ITERATION_LIMIT,
        objective_value=bfs.objective(),
        variable_values=variable_values,
        variable_values_by_name={},
        basis_indices=sorted(r * n + c for (r, c) in bfs.basic_cells),
        iterations=max_iterations,
        diagnostics={"algorithm": "MODI", "product_id": product_id},
        messages=messages,
    )


# Decode MODIResult → transfer_plan rows

def _decode_result(
    result: MODIResult,
    instance: TransportationInstance,
    distance_matrix: pd.DataFrame,
    transport_cost_matrix: pd.DataFrame,
    min_unit_threshold: float = 1e-6,
) -> pd.DataFrame:
    if result.status != STATUS_OPTIMAL:
        return pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)

    source_ids = list(instance.source_store_ids)
    demand_ids = list(instance.demand_store_ids)
    m, n = len(source_ids), len(demand_ids)
    product_id = instance.product_id

    rows: List[Dict[str, Any]] = []
    dummy_flows: List[Dict[str, Any]] = []

    for i, src in enumerate(source_ids):
        for j, dem in enumerate(demand_ids):
            value = float(result.variable_values[i * n + j])
            if value <= min_unit_threshold:
                continue

            src_is_dummy = isinstance(src, str) and src.upper().startswith("DUMMY")
            dem_is_dummy = isinstance(dem, str) and dem.upper().startswith("DUMMY")

            if src_is_dummy or dem_is_dummy:
                dummy_flows.append(
                    {"from_store_id": src, "to_store_id": dem, "product_id": product_id, "units": value}
                )
                continue

            units = int(round(value))
            if units <= 0:
                continue

            def _safe(df: pd.DataFrame, r: Any, c: Any, default: float) -> float:
                if df is None or df.empty:
                    return default
                if r not in df.index or c not in df.columns:
                    return default
                val = pd.to_numeric(df.loc[r, c], errors="coerce")
                return float(val) if not pd.isna(val) else default

            dist = _safe(distance_matrix, src, dem, 0.0)
            unit_cost = _safe(transport_cost_matrix, src, dem, float(instance.cost_matrix[i, j]))
            transport_cost = unit_cost * units

            rows.append(
                {
                    "from_store_id": int(src),
                    "to_store_id": int(dem),
                    "product_id": int(product_id),
                    "units": units,
                    "distance_km": float(dist),
                    "transport_cost": float(transport_cost),
                }
            )

    result.diagnostics["dummy_flows"] = dummy_flows
    result.diagnostics["dummy_flow_count"] = len(dummy_flows)
    return pd.DataFrame(rows, columns=TRANSFER_PLAN_COLUMNS)


# Public optimizer class

class MODIOptimizer:
    """
    MODI (Modified Distribution Method) optimizer.

    Public interface mirrors RuleBasedOptimizer so it plugs directly into
    any pipeline that uses rule_based or genetic_algorithm optimizers.

    Parameters
    ----------
    distance_matrix : pd.DataFrame, optional
        Store-to-store distance matrix.
    transport_cost_matrix : pd.DataFrame, optional
        Store-to-store unit cost matrix.
    max_iterations : int
        Per-instance MODI iteration cap.
    tolerance : float
        Numeric tolerance for optimality / degeneracy checks.
    keep_history : bool
        Whether to capture per-iteration snapshots (debug use).
    dummy_shortage_cost : float, optional
        Penalty cost for dummy-source arcs (inferred if None).
    dummy_excess_cost : float
        Cost for dummy-destination arcs.
    random_seed : int, optional
        Not used by MODI; present for interface parity with GA optimizer.
    """

    def __init__(
        self,
        distance_matrix: Optional[pd.DataFrame] = None,
        transport_cost_matrix: Optional[pd.DataFrame] = None,
        max_iterations: int = 2000,
        tolerance: float = 1e-9,
        keep_history: bool = False,
        dummy_shortage_cost: Optional[float] = None,
        dummy_excess_cost: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> None:
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.keep_history = bool(keep_history)
        self.dummy_shortage_cost = dummy_shortage_cost
        self.dummy_excess_cost = float(dummy_excess_cost)
        self.transfer_plan: Optional[pd.DataFrame] = None
        self.logger_system = get_optimization_logger()

    def load_matrices(self, distance_path: str, cost_path: str) -> None:
        """Load distance and transport cost matrices from CSV files."""
        self.distance_matrix = pd.read_csv(distance_path, index_col=0)
        self.transport_cost_matrix = pd.read_csv(cost_path, index_col=0)
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)
        self.transport_cost_matrix.index = self.transport_cost_matrix.index.astype(int)
        self.transport_cost_matrix.columns = self.transport_cost_matrix.columns.astype(int)

    def optimize(
        self,
        excess_inventory: pd.DataFrame,
        needed_inventory: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a transfer plan using MODI.

        Parameters
        ----------
        excess_inventory : pd.DataFrame
            Must contain ``store_id``, ``product_id``, ``excess_units``.
        needed_inventory : pd.DataFrame
            Must contain ``store_id``, ``product_id``, ``needed_units``.

        Returns
        -------
        pd.DataFrame
            Transfer plan with columns:
            from_store_id, to_store_id, product_id, units,
            distance_km, transport_cost.
        """
        start_time = time.time()

        parameters = {
            "excess_items": len(excess_inventory) if not excess_inventory.empty else 0,
            "needed_items": len(needed_inventory) if not needed_inventory.empty else 0,
            "algorithm": "MODI Optimization",
        }
        self.logger_system.log_execution_start("modi_optimization", parameters)
        print("Generating MODI transfer plan...")
        self.logger_system.log_progress("modi_optimization", "Starting MODI transfer plan generation...")

        if excess_inventory.empty or needed_inventory.empty:
            msg = "No excess or needed inventory found. No transfers needed."
            print(msg)
            self.logger_system.log_progress("modi_optimization", msg)
            self.transfer_plan = pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)
            elapsed = time.time() - start_time
            self.logger_system.log_execution_end("modi_optimization", elapsed, {"transfers_generated": 0})
            return self.transfer_plan

        instances = build_transportation_instances(
            excess_inventory=excess_inventory,
            needed_inventory=needed_inventory,
            distance_matrix=self.distance_matrix,
            transport_cost_matrix=self.transport_cost_matrix,
            allow_self_transfer=False,
            dummy_shortage_cost=self.dummy_shortage_cost,
            dummy_excess_cost=self.dummy_excess_cost,
        )

        self.logger_system.log_progress(
            "modi_optimization",
            f"Transportation instances built: {len(instances)}",
        )

        plan_frames: List[pd.DataFrame] = []
        optimal_count = 0
        status_counts: Dict[str, int] = {}

        for instance in instances:
            result = _solve_instance(
                instance,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                keep_history=self.keep_history,
            )
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

            if result.status != STATUS_OPTIMAL:
                _logger.warning(
                    "MODI: product %s status=%s", instance.product_id, result.status
                )
                continue

            optimal_count += 1
            decoded = _decode_result(
                result,
                instance,
                self.distance_matrix,
                self.transport_cost_matrix,
            )
            if not decoded.empty:
                plan_frames.append(decoded)

        self.logger_system.log_progress(
            "modi_optimization",
            f"Optimal instances: {optimal_count}/{len(instances)} | statuses: {status_counts}",
        )

        if plan_frames:
            combined = pd.concat(plan_frames, ignore_index=True)
            merged = (
                combined.groupby(
                    ["from_store_id", "to_store_id", "product_id"], as_index=False
                )
                .agg({"units": "sum", "distance_km": "mean", "transport_cost": "sum"})
                .sort_values("transport_cost", ascending=False)
            )
            merged = merged[merged["units"] > 0].copy()
            merged["units"] = merged["units"].astype(int)
            self.transfer_plan = merged[TRANSFER_PLAN_COLUMNS].reset_index(drop=True)
        else:
            self.transfer_plan = pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)

        elapsed = time.time() - start_time

        if not self.transfer_plan.empty:
            total_units = self.transfer_plan["units"].sum()
            total_cost = self.transfer_plan["transport_cost"].sum()
            avg_cost = total_cost / total_units if total_units > 0 else 0.0
            print("MODI Transfer Plan Summary:")
            print(f"  Total transfers: {len(self.transfer_plan)}")
            print(f"  Total units    : {total_units}")
            print(f"  Total cost     : {total_cost:,.0f} VND")
            print(f"  Avg cost/unit  : {avg_cost:,.0f} VND")
            self.logger_system.log_progress("modi_optimization", f"Total transfers: {len(self.transfer_plan)}")
            self.logger_system.log_progress("modi_optimization", f"Total transport cost: {total_cost:,.0f} VND")
        else:
            print("No MODI transfers recommended.")
            self.logger_system.log_progress("modi_optimization", "No transfers recommended.")

        self.logger_system.log_execution_end(
            "modi_optimization",
            elapsed,
            {
                "transfers_generated": len(self.transfer_plan),
                "total_units": int(self.transfer_plan["units"].sum()) if not self.transfer_plan.empty else 0,
                "total_cost": float(self.transfer_plan["transport_cost"].sum()) if not self.transfer_plan.empty else 0.0,
            },
        )

        return self.transfer_plan

    def add_store_product_names(
        self,
        stores_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Add human-readable store/product name columns (mirrors RuleBasedOptimizer)."""
        if self.transfer_plan is None or self.transfer_plan.empty:
            return
        if stores_df is not None:
            store_name_map = stores_df.set_index("store_id")["store_name"].to_dict()
            self.transfer_plan["from_store"] = self.transfer_plan["from_store_id"].map(store_name_map)
            self.transfer_plan["to_store"] = self.transfer_plan["to_store_id"].map(store_name_map)
        if products_df is not None:
            product_name_map = products_df.set_index("product_id")["product_name"].to_dict()
            self.transfer_plan["product"] = self.transfer_plan["product_id"].map(product_name_map)


__all__ = ["MODIOptimizer", "MODIResult", "TRANSFER_PLAN_COLUMNS"]
