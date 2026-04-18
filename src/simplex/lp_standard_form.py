from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from src.transport.transportation_instance_builder import TransportationInstance


@dataclass
class StandardFormLP:
    """Standard-form LP contract for simplex solving."""

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    variable_names: List[str]
    basis_indices: List[int]
    artificial_variable_indices: List[int]
    objective_sense: str = "min"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _arc_variable_name(source_id: Any, demand_id: Any, product_id: int) -> str:
    """Build stable variable names for LP columns."""
    source_text = str(source_id).replace(" ", "_")
    demand_text = str(demand_id).replace(" ", "_")
    return f"x_p{product_id}_s{source_text}_d{demand_text}"


def _northwest_corner_basis_cells(
    supply: np.ndarray,
    demand: np.ndarray,
    tol: float = 1e-9,
) -> List[Tuple[int, int]]:
    """
    Create a transportation BFS cell list using northwest corner logic.

    This provides a deterministic basis candidate set for transportation LPs.
    """
    supply_work = supply.astype(float).copy()
    demand_work = demand.astype(float).copy()

    m = supply_work.shape[0]
    n = demand_work.shape[0]

    if m == 0 or n == 0:
        return []

    basis_cells: List[Tuple[int, int]] = []
    i, j = 0, 0

    while i < m and j < n:
        basis_cells.append((i, j))

        allocation = min(supply_work[i], demand_work[j])
        supply_work[i] -= allocation
        demand_work[j] -= allocation

        source_done = supply_work[i] <= tol
        demand_done = demand_work[j] <= tol

        if source_done and demand_done:
            # Degenerate corner: add one extra zero-allocation basis candidate.
            if i < m - 1 and j < n - 1:
                basis_cells.append((i + 1, j))
            i += 1
            j += 1
        elif source_done:
            i += 1
        else:
            j += 1

    target_size = max(0, m + n - 1)

    # Deduplicate while preserving order.
    deduped: List[Tuple[int, int]] = []
    seen = set()
    for cell in basis_cells:
        if cell[0] >= m or cell[1] >= n:
            continue
        if cell in seen:
            continue
        deduped.append(cell)
        seen.add(cell)

    # Add extra cells if basis is still short.
    if len(deduped) < target_size:
        for row in range(m):
            for col in range(n):
                cell = (row, col)
                if cell in seen:
                    continue
                deduped.append(cell)
                seen.add(cell)
                if len(deduped) >= target_size:
                    break
            if len(deduped) >= target_size:
                break

    return deduped[:target_size]


def _repair_basis_indices(
    A: np.ndarray,
    candidate_basis_indices: List[int],
    required_count: int,
    tol: float = 1e-9,
) -> List[int]:
    """Repair and validate basis indices to be full-rank and size-consistent."""
    if required_count == 0:
        return []

    unique_candidates: List[int] = []
    seen = set()
    for idx in candidate_basis_indices:
        if idx in seen:
            continue
        unique_candidates.append(idx)
        seen.add(idx)

    all_indices = list(range(A.shape[1]))
    ordered_candidates = unique_candidates + [
        idx for idx in all_indices if idx not in seen
    ]

    basis: List[int] = []
    current_rank = 0

    for idx in ordered_candidates:
        trial = basis + [idx]
        trial_rank = np.linalg.matrix_rank(A[:, trial], tol=tol)
        if trial_rank > current_rank:
            basis.append(idx)
            current_rank = trial_rank
        if len(basis) >= required_count:
            break

    if len(basis) < required_count:
        raise ValueError(
            "Unable to construct a full-size basis from transportation LP columns."
        )

    basis = basis[:required_count]
    final_rank = np.linalg.matrix_rank(A[:, basis], tol=tol)
    if final_rank < required_count:
        raise ValueError("Constructed basis is rank-deficient.")

    return basis


def transportation_instance_to_standard_form(
    instance: TransportationInstance,
    drop_redundant_constraint: bool = True,
) -> StandardFormLP:
    """
    Convert one transportation instance to LP standard form.

    Standard objective:
        minimize c^T x
    subject to:
        A x = b
        x >= 0
    """
    source_ids = list(instance.source_store_ids)
    demand_ids = list(instance.demand_store_ids)

    m = len(source_ids)
    n = len(demand_ids)
    num_vars = m * n

    if num_vars == 0:
        return StandardFormLP(
            A=np.zeros((0, 0), dtype=float),
            b=np.zeros((0,), dtype=float),
            c=np.zeros((0,), dtype=float),
            variable_names=[],
            basis_indices=[],
            artificial_variable_indices=[],
            objective_sense="min",
            metadata={"product_id": instance.product_id, "empty_problem": True},
        )

    variable_names: List[str] = []
    variable_to_arc: Dict[int, Tuple[Any, Any, int]] = {}
    arc_to_variable: Dict[Tuple[Any, Any], int] = {}

    c_values: List[float] = []
    var_idx = 0
    for i, source_id in enumerate(source_ids):
        for j, demand_id in enumerate(demand_ids):
            variable_names.append(_arc_variable_name(source_id, demand_id, instance.product_id))
            variable_to_arc[var_idx] = (source_id, demand_id, instance.product_id)
            arc_to_variable[(source_id, demand_id)] = var_idx
            c_values.append(float(instance.cost_matrix[i, j]))
            var_idx += 1

    c = np.asarray(c_values, dtype=float)

    # Build equality constraints: source balance + demand balance.
    source_rows = np.zeros((m, num_vars), dtype=float)
    demand_rows = np.zeros((n, num_vars), dtype=float)

    for i in range(m):
        for j in range(n):
            source_rows[i, i * n + j] = 1.0

    for j in range(n):
        for i in range(m):
            demand_rows[j, i * n + j] = 1.0

    A = np.vstack([source_rows, demand_rows])
    b = np.concatenate(
        [instance.supply_vector.astype(float), instance.demand_vector.astype(float)]
    )

    dropped_constraint_row = None
    if drop_redundant_constraint and A.shape[0] > 1:
        # Transportation equalities have one redundant row when balanced.
        dropped_constraint_row = A.shape[0] - 1
        A = np.delete(A, dropped_constraint_row, axis=0)
        b = np.delete(b, dropped_constraint_row, axis=0)

    # Build basis candidates via northwest-corner support.
    nw_cells = _northwest_corner_basis_cells(
        instance.supply_vector.astype(float),
        instance.demand_vector.astype(float),
    )
    basis_candidates = [row * n + col for row, col in nw_cells]

    required_basis_size = A.shape[0]
    basis_indices = _repair_basis_indices(
        A,
        basis_candidates,
        required_basis_size,
    )

    metadata: Dict[str, Any] = {
        "product_id": instance.product_id,
        "num_sources": m,
        "num_demands": n,
        "num_variables": num_vars,
        "dropped_constraint_row": dropped_constraint_row,
        "variable_to_arc": variable_to_arc,
        "arc_to_variable": arc_to_variable,
        "dummy_added": instance.dummy_added,
        "instance_metadata": dict(instance.metadata),
        "original_variable_count": num_vars,
    }

    return StandardFormLP(
        A=A,
        b=b,
        c=c,
        variable_names=variable_names,
        basis_indices=basis_indices,
        artificial_variable_indices=[],
        objective_sense="min",
        metadata=metadata,
    )


def add_phase_one_artificial_variables(lp: StandardFormLP) -> StandardFormLP:
    """
    Build an auxiliary Phase-I LP with artificial variables.

    This utility is provided for robustness fallback when an initial basis is invalid.
    """
    m, n = lp.A.shape

    if m == 0:
        return lp

    identity = np.eye(m, dtype=float)
    A_augmented = np.hstack([lp.A, identity])

    c_phase_one = np.concatenate([np.zeros((n,), dtype=float), np.ones((m,), dtype=float)])
    artificial_indices = list(range(n, n + m))

    variable_names = list(lp.variable_names) + [f"a_{row}" for row in range(m)]

    metadata = dict(lp.metadata)
    metadata.update(
        {
            "phase": "phase_one",
            "original_variable_count": n,
            "original_objective": lp.c.copy(),
        }
    )

    return StandardFormLP(
        A=A_augmented,
        b=lp.b.copy(),
        c=c_phase_one,
        variable_names=variable_names,
        basis_indices=artificial_indices,
        artificial_variable_indices=artificial_indices,
        objective_sense="min",
        metadata=metadata,
    )
