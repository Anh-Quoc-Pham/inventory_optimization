from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.simplex.lp_standard_form import StandardFormLP


@dataclass
class TableauSnapshot:
    """Serializable snapshot of tableau state for debugging."""

    iteration: int
    basis_indices: List[int]
    objective_value: float
    reduced_costs: List[float]
    rhs_values: List[float]


class SimplexTableau:
    """Simplex tableau with Gauss-Jordan pivot operations."""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        basis_indices: List[int],
        objective_sense: str = "min",
        variable_names: Optional[List[str]] = None,
        tolerance: float = 1e-9,
    ):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.c = np.asarray(c, dtype=float)

        self.objective_sense = objective_sense
        self.variable_names = variable_names or [f"x_{idx}" for idx in range(self.A.shape[1])]
        self.tolerance = float(tolerance)

        self.basis_indices = list(basis_indices)
        self._validate_input_shapes()
        self.tableau = self._build_tableau_from_basis()

    @classmethod
    def from_standard_form(
        cls,
        lp: StandardFormLP,
        tolerance: float = 1e-9,
    ) -> "SimplexTableau":
        """Create a tableau from the StandardFormLP contract."""
        return cls(
            A=lp.A,
            b=lp.b,
            c=lp.c,
            basis_indices=lp.basis_indices,
            objective_sense=lp.objective_sense,
            variable_names=lp.variable_names,
            tolerance=tolerance,
        )

    @property
    def num_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def num_variables(self) -> int:
        return self.A.shape[1]

    @property
    def nonbasic_indices(self) -> List[int]:
        basis_set = set(self.basis_indices)
        return [idx for idx in range(self.num_variables) if idx not in basis_set]

    def _validate_input_shapes(self) -> None:
        if self.A.ndim != 2:
            raise ValueError("A must be a 2-dimensional matrix.")

        if self.b.ndim != 1:
            raise ValueError("b must be a 1-dimensional vector.")

        if self.c.ndim != 1:
            raise ValueError("c must be a 1-dimensional vector.")

        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A row count must match b size.")

        if self.A.shape[1] != self.c.shape[0]:
            raise ValueError("A column count must match c size.")

        if len(self.basis_indices) != self.A.shape[0]:
            raise ValueError(
                "basis_indices must contain exactly one variable index per constraint row."
            )

        if len(set(self.basis_indices)) != len(self.basis_indices):
            raise ValueError("basis_indices must be unique.")

        if min(self.basis_indices, default=0) < 0 or max(self.basis_indices, default=0) >= self.A.shape[1]:
            raise ValueError("basis_indices contain out-of-range variable indices.")

    def _build_tableau_from_basis(self) -> np.ndarray:
        """Construct tableau using the current basis."""
        if self.num_constraints == 0 or self.num_variables == 0:
            return np.zeros((1, 1), dtype=float)

        basis_matrix = self.A[:, self.basis_indices]

        transformed_A = np.linalg.solve(basis_matrix, self.A)
        transformed_b = np.linalg.solve(basis_matrix, self.b)

        cost_basic = self.c[self.basis_indices]
        dual_prices = np.linalg.solve(basis_matrix.T, cost_basic)
        reduced_costs = self.c - self.A.T @ dual_prices

        objective_value = float(cost_basic @ transformed_b)

        tableau = np.zeros((self.num_constraints + 1, self.num_variables + 1), dtype=float)
        tableau[: self.num_constraints, : self.num_variables] = transformed_A
        tableau[: self.num_constraints, -1] = transformed_b
        tableau[-1, : self.num_variables] = reduced_costs
        tableau[-1, -1] = objective_value

        tableau[np.abs(tableau) < self.tolerance] = 0.0
        return tableau

    def reduced_costs(self) -> np.ndarray:
        """Return reduced costs for all variable columns."""
        return self.tableau[-1, : self.num_variables].copy()

    def rhs_values(self) -> np.ndarray:
        """Return current right-hand side values for constraint rows."""
        return self.tableau[: self.num_constraints, -1].copy()

    def objective_value(self) -> float:
        """Return current objective value."""
        return float(self.tableau[-1, -1])

    def column_values(self, column_index: int) -> np.ndarray:
        """Return one pivot column from constraint rows."""
        return self.tableau[: self.num_constraints, column_index].copy()

    def current_solution(self) -> np.ndarray:
        """Build full variable vector from basic RHS values."""
        x = np.zeros((self.num_variables,), dtype=float)
        rhs = self.rhs_values()
        for row_idx, var_idx in enumerate(self.basis_indices):
            x[var_idx] = rhs[row_idx]
        return x

    def pivot(self, pivot_row: int, pivot_col: int) -> None:
        """Perform one Gauss-Jordan pivot and update basis."""
        pivot_value = float(self.tableau[pivot_row, pivot_col])
        if abs(pivot_value) <= self.tolerance:
            raise ZeroDivisionError("Pivot value is numerically zero.")

        # Normalize pivot row.
        self.tableau[pivot_row, :] = self.tableau[pivot_row, :] / pivot_value

        # Eliminate pivot column from all other rows.
        for row in range(self.tableau.shape[0]):
            if row == pivot_row:
                continue

            factor = float(self.tableau[row, pivot_col])
            if abs(factor) <= self.tolerance:
                continue

            self.tableau[row, :] = self.tableau[row, :] - factor * self.tableau[pivot_row, :]

        self.tableau[np.abs(self.tableau) < self.tolerance] = 0.0
        self.basis_indices[pivot_row] = pivot_col

    def has_finite_values(self) -> bool:
        """Check whether tableau is numerically valid."""
        return bool(np.isfinite(self.tableau).all())

    def snapshot(self, iteration: int) -> TableauSnapshot:
        """Create a lightweight tableau snapshot for debug history."""
        return TableauSnapshot(
            iteration=iteration,
            basis_indices=list(self.basis_indices),
            objective_value=self.objective_value(),
            reduced_costs=self.reduced_costs().tolist(),
            rhs_values=self.rhs_values().tolist(),
        )

    def rebuild_from_basis(self) -> None:
        """Recompute the full tableau from the current basis (numerical refresh)."""
        self.tableau = self._build_tableau_from_basis()
