from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def select_entering_variable(
    reduced_costs: np.ndarray,
    nonbasic_indices: Sequence[int],
    objective_sense: str = "min",
    rule: str = "dantzig",
    tolerance: float = 1e-9,
    use_bland: bool = False,
) -> Optional[int]:
    """
    Select entering variable index from nonbasic columns.

    For minimization, any reduced cost < -tol can improve objective.
    For maximization, any reduced cost > +tol can improve objective.
    """
    if len(nonbasic_indices) == 0:
        return None

    if objective_sense not in {"min", "max"}:
        raise ValueError("objective_sense must be 'min' or 'max'.")

    candidates: List[int] = []
    for idx in nonbasic_indices:
        rc = float(reduced_costs[idx])
        if objective_sense == "min" and rc < -tolerance:
            candidates.append(idx)
        elif objective_sense == "max" and rc > tolerance:
            candidates.append(idx)

    if not candidates:
        return None

    if use_bland or rule.lower() == "bland":
        return min(candidates)

    if rule.lower() != "dantzig":
        raise ValueError("Unsupported pivot rule. Use 'dantzig' or 'bland'.")

    if objective_sense == "min":
        return min(candidates, key=lambda idx: reduced_costs[idx])

    return max(candidates, key=lambda idx: reduced_costs[idx])


def minimum_ratio_test(
    pivot_column: np.ndarray,
    rhs_values: np.ndarray,
    basis_indices: Optional[Sequence[int]] = None,
    tolerance: float = 1e-9,
    use_bland: bool = False,
) -> Optional[int]:
    """
    Perform minimum-ratio test and return leaving row index.

    Rows with pivot coefficient <= tol are ignored.
    """
    if pivot_column.shape[0] != rhs_values.shape[0]:
        raise ValueError("pivot_column and rhs_values must have the same length.")

    candidates: List[Tuple[int, float]] = []

    for row_idx, coeff in enumerate(pivot_column):
        coeff_value = float(coeff)
        if coeff_value <= tolerance:
            continue

        rhs_value = float(rhs_values[row_idx])
        ratio = rhs_value / coeff_value

        if ratio < -tolerance:
            continue

        candidates.append((row_idx, ratio))

    if not candidates:
        return None

    min_ratio = min(ratio for _, ratio in candidates)
    tied_rows = [row for row, ratio in candidates if abs(ratio - min_ratio) <= tolerance]

    if len(tied_rows) == 1:
        return tied_rows[0]

    if use_bland and basis_indices is not None:
        return min(tied_rows, key=lambda row: basis_indices[row])

    return min(tied_rows)


def bland_entering_variable(
    reduced_costs: np.ndarray,
    nonbasic_indices: Sequence[int],
    objective_sense: str = "min",
    tolerance: float = 1e-9,
) -> Optional[int]:
    """Convenience wrapper for Bland's entering-variable rule."""
    return select_entering_variable(
        reduced_costs=reduced_costs,
        nonbasic_indices=nonbasic_indices,
        objective_sense=objective_sense,
        rule="bland",
        tolerance=tolerance,
        use_bland=True,
    )


def bland_leaving_row(
    candidate_rows: Sequence[int],
    basis_indices: Sequence[int],
) -> Optional[int]:
    """Select leaving row by smallest indexed basic variable (Bland tie-break)."""
    if not candidate_rows:
        return None

    return min(candidate_rows, key=lambda row: basis_indices[row])


def detect_basis_cycle(
    basis_history: Sequence[Tuple[int, ...]],
    current_basis: Sequence[int],
) -> bool:
    """Detect whether current basis has appeared before."""
    current = tuple(current_basis)
    return current in set(basis_history)
