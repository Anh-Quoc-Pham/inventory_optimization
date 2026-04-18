from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


STATUS_OPTIMAL = "OPTIMAL"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_UNBOUNDED = "UNBOUNDED"
STATUS_ITERATION_LIMIT = "ITERATION_LIMIT"
STATUS_NUMERICAL_ISSUE = "NUMERICAL_ISSUE"


def check_empty_problem(
    num_constraints: int,
    num_variables: int,
    c: np.ndarray,
    objective_sense: str = "min",
    tolerance: float = 1e-9,
) -> Tuple[bool, str, str]:
    """
    Determine whether LP is a trivial empty-case problem.

    Returns:
        (is_terminal, status, message)
    """
    if num_variables == 0 and num_constraints == 0:
        return True, STATUS_OPTIMAL, "Empty LP with no constraints/variables."

    if num_variables == 0 and num_constraints > 0:
        return True, STATUS_INFEASIBLE, "No variables available to satisfy constraints."

    if num_constraints == 0:
        if objective_sense == "min" and np.any(c < -tolerance):
            return True, STATUS_UNBOUNDED, "Unconstrained minimization with negative costs."
        if objective_sense == "max" and np.any(c > tolerance):
            return True, STATUS_UNBOUNDED, "Unconstrained maximization with positive costs."
        return True, STATUS_OPTIMAL, "Unconstrained LP solved at x=0."

    return False, "", ""


def check_primal_rhs_feasibility(rhs_values: np.ndarray, tolerance: float = 1e-9) -> bool:
    """Check whether all RHS values are nonnegative within tolerance."""
    return bool(np.all(rhs_values >= -tolerance))


def is_optimal(
    reduced_costs: np.ndarray,
    nonbasic_indices: Sequence[int],
    objective_sense: str = "min",
    tolerance: float = 1e-9,
) -> bool:
    """Evaluate simplex optimality conditions on nonbasic reduced costs."""
    if len(nonbasic_indices) == 0:
        return True

    relevant = np.asarray([reduced_costs[idx] for idx in nonbasic_indices], dtype=float)
    if objective_sense == "min":
        return bool(np.all(relevant >= -tolerance))

    return bool(np.all(relevant <= tolerance))


def is_unbounded_pivot_column(
    pivot_column: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    """Detect primal unboundedness for chosen entering variable."""
    return bool(np.all(pivot_column <= tolerance))


def check_numeric_validity(values: np.ndarray) -> bool:
    """Check if array values are finite and not NaN."""
    return bool(np.isfinite(values).all())


def build_diagnostics(
    status: str,
    message: str,
    iterations: int,
    objective_value: float | None,
    basis_indices: Sequence[int],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a consistent diagnostics payload for solver results."""
    diagnostics: Dict[str, Any] = {
        "status": status,
        "message": message,
        "iterations": int(iterations),
        "objective_value": objective_value,
        "basis_indices": list(basis_indices),
    }

    if extra:
        diagnostics.update(extra)

    return diagnostics
