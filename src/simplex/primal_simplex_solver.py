from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.simplex.feasibility_and_unbounded_checks import (
    STATUS_INFEASIBLE,
    STATUS_ITERATION_LIMIT,
    STATUS_NUMERICAL_ISSUE,
    STATUS_OPTIMAL,
    STATUS_UNBOUNDED,
    build_diagnostics,
    check_empty_problem,
    check_numeric_validity,
    check_primal_rhs_feasibility,
    is_optimal,
    is_unbounded_pivot_column,
)
from src.simplex.lp_standard_form import StandardFormLP, add_phase_one_artificial_variables
from src.simplex.pivot_rules import (
    detect_basis_cycle,
    minimum_ratio_test,
    select_entering_variable,
)
from src.simplex.simplex_tableau import SimplexTableau
from src.utils.logger import get_optimization_logger


@dataclass
class SimplexResult:
    """Simplex solver output contract."""

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


class PrimalSimplexSolver:
    """Primal simplex implementation with Bland anti-cycling support."""

    def __init__(
        self,
        max_iterations: int = 2000,
        tolerance: float = 1e-9,
        pivot_rule: str = "dantzig",
        use_bland: bool = True,
        keep_history: bool = False,
    ):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.pivot_rule = pivot_rule
        self.use_bland = bool(use_bland)
        self.keep_history = bool(keep_history)
        self.logger_system = get_optimization_logger()

    def solve(
        self,
        lp: StandardFormLP,
        max_iterations: Optional[int] = None,
        keep_history: Optional[bool] = None,
    ) -> SimplexResult:
        """Solve an LP in standard form using primal simplex."""
        limit = int(max_iterations) if max_iterations is not None else self.max_iterations
        capture_history = self.keep_history if keep_history is None else bool(keep_history)

        product_id = lp.metadata.get("product_id", "unknown")
        self.logger_system.log_progress(
            "primal_simplex_solver",
            f"Starting simplex solve for product={product_id}, vars={lp.A.shape[1]}, constraints={lp.A.shape[0]}",
        )

        is_terminal, terminal_status, terminal_message = check_empty_problem(
            num_constraints=lp.A.shape[0],
            num_variables=lp.A.shape[1],
            c=lp.c,
            objective_sense=lp.objective_sense,
            tolerance=self.tolerance,
        )

        if is_terminal:
            values = np.zeros((lp.A.shape[1],), dtype=float)
            return self._build_result(
                lp=lp,
                status=terminal_status,
                message=terminal_message,
                values=values,
                basis_indices=[],
                iterations=0,
                objective_value=0.0 if terminal_status == STATUS_OPTIMAL else None,
                history=[] if capture_history else None,
            )

        messages: List[str] = []

        # Attempt direct solve with provided basis.
        direct_result = self._solve_with_basis(
            lp=lp,
            max_iterations=limit,
            keep_history=capture_history,
            use_phase_name="phase_two",
        )

        if direct_result.status != STATUS_NUMERICAL_ISSUE:
            self.logger_system.log_progress(
                "primal_simplex_solver",
                f"Simplex solve finished for product={product_id} with status={direct_result.status}",
            )
            return direct_result

        # Fallback to Phase-I if initial basis is not usable.
        messages.extend(direct_result.messages)
        messages.append("Direct basis solve failed; attempting Phase-I fallback.")

        phase_one_result = self._solve_via_phase_one(
            lp=lp,
            max_iterations=limit,
            keep_history=capture_history,
            inherited_messages=messages,
        )

        self.logger_system.log_progress(
            "primal_simplex_solver",
            f"Simplex solve finished for product={product_id} with status={phase_one_result.status}",
        )
        return phase_one_result

    def _solve_with_basis(
        self,
        lp: StandardFormLP,
        max_iterations: int,
        keep_history: bool,
        use_phase_name: str,
        inherited_messages: Optional[List[str]] = None,
    ) -> SimplexResult:
        """Run simplex iterations with the basis provided in LP."""
        messages = list(inherited_messages or [])

        try:
            tableau = SimplexTableau.from_standard_form(lp, tolerance=self.tolerance)
        except Exception as exc:
            messages.append(f"Failed to initialize tableau: {exc}")
            return self._build_result(
                lp=lp,
                status=STATUS_NUMERICAL_ISSUE,
                message="Tableau initialization failed.",
                values=np.zeros((lp.A.shape[1],), dtype=float),
                basis_indices=[],
                iterations=0,
                objective_value=None,
                history=[] if keep_history else None,
                messages=messages,
            )

        if not check_primal_rhs_feasibility(tableau.rhs_values(), tolerance=self.tolerance):
            messages.append(
                "Initial basis is not primal feasible (negative RHS detected)."
            )
            return self._build_result(
                lp=lp,
                status=STATUS_NUMERICAL_ISSUE,
                message="Initial basis infeasible for primal simplex.",
                values=np.zeros((lp.A.shape[1],), dtype=float),
                basis_indices=tableau.basis_indices,
                iterations=0,
                objective_value=None,
                history=[] if keep_history else None,
                messages=messages,
            )

        status, message, iterations, history = self._iterate_simplex(
            tableau=tableau,
            objective_sense=lp.objective_sense,
            max_iterations=max_iterations,
            keep_history=keep_history,
        )

        values = tableau.current_solution()

        result = self._build_result(
            lp=lp,
            status=status,
            message=f"{use_phase_name}: {message}",
            values=values,
            basis_indices=tableau.basis_indices,
            iterations=iterations,
            objective_value=(tableau.objective_value() if status == STATUS_OPTIMAL else None),
            history=history,
            messages=messages,
        )

        if use_phase_name == "phase_one":
            result.phase_one_iterations = iterations
        else:
            result.phase_two_iterations = iterations

        return result

    def _solve_via_phase_one(
        self,
        lp: StandardFormLP,
        max_iterations: int,
        keep_history: bool,
        inherited_messages: Optional[List[str]] = None,
    ) -> SimplexResult:
        """Fallback two-phase strategy for difficult basis initialization cases."""
        messages = list(inherited_messages or [])

        phase_one_lp = add_phase_one_artificial_variables(lp)
        phase_one_result = self._solve_with_basis(
            lp=phase_one_lp,
            max_iterations=max_iterations,
            keep_history=keep_history,
            use_phase_name="phase_one",
            inherited_messages=messages,
        )

        if phase_one_result.status != STATUS_OPTIMAL:
            phase_one_result.messages.append("Phase-I failed before reaching optimality.")
            return phase_one_result

        if phase_one_result.objective_value is None:
            phase_one_result.messages.append("Phase-I did not produce an objective value.")
            phase_one_result.status = STATUS_NUMERICAL_ISSUE
            return phase_one_result

        if abs(phase_one_result.objective_value) > self.tolerance:
            return self._build_result(
                lp=lp,
                status=STATUS_INFEASIBLE,
                message="Phase-I objective > 0. Original LP is infeasible.",
                values=np.zeros((lp.A.shape[1],), dtype=float),
                basis_indices=[],
                iterations=phase_one_result.iterations,
                objective_value=None,
                history=phase_one_result.tableau_history,
                messages=phase_one_result.messages,
            )

        basis_for_phase_two = self._derive_feasible_basis(
            A=lp.A,
            b=lp.b,
            x_candidate=phase_one_result.variable_values[: lp.A.shape[1]],
        )

        if basis_for_phase_two is None:
            messages = list(phase_one_result.messages)
            messages.append("Could not derive a feasible basis for Phase-II from Phase-I solution.")
            return self._build_result(
                lp=lp,
                status=STATUS_NUMERICAL_ISSUE,
                message="Phase-I succeeded but basis recovery failed.",
                values=np.zeros((lp.A.shape[1],), dtype=float),
                basis_indices=[],
                iterations=phase_one_result.iterations,
                objective_value=None,
                history=phase_one_result.tableau_history,
                messages=messages,
            )

        phase_two_lp = StandardFormLP(
            A=lp.A,
            b=lp.b,
            c=lp.c,
            variable_names=lp.variable_names,
            basis_indices=basis_for_phase_two,
            artificial_variable_indices=[],
            objective_sense=lp.objective_sense,
            metadata=dict(lp.metadata),
        )

        phase_two_result = self._solve_with_basis(
            lp=phase_two_lp,
            max_iterations=max_iterations,
            keep_history=keep_history,
            use_phase_name="phase_two",
            inherited_messages=phase_one_result.messages,
        )

        phase_two_result.phase_one_iterations = phase_one_result.phase_one_iterations
        phase_two_result.phase_two_iterations = phase_two_result.iterations
        phase_two_result.iterations = (
            phase_one_result.phase_one_iterations + phase_two_result.phase_two_iterations
        )

        if keep_history:
            merged_history: List[Dict[str, Any]] = []
            if phase_one_result.tableau_history:
                merged_history.extend(phase_one_result.tableau_history)
            if phase_two_result.tableau_history:
                merged_history.extend(phase_two_result.tableau_history)
            phase_two_result.tableau_history = merged_history

        return phase_two_result

    def _iterate_simplex(
        self,
        tableau: SimplexTableau,
        objective_sense: str,
        max_iterations: int,
        keep_history: bool,
    ) -> Tuple[str, str, int, Optional[List[Dict[str, Any]]]]:
        """Core simplex iteration loop."""
        history: Optional[List[Dict[str, Any]]] = [] if keep_history else None
        basis_history: List[Tuple[int, ...]] = []

        use_bland_now = self.use_bland

        for iteration in range(1, max_iterations + 1):
            if keep_history and history is not None:
                history.append(tableau.snapshot(iteration - 1).__dict__)

            if not tableau.has_finite_values() or not check_numeric_validity(tableau.tableau):
                return (
                    STATUS_NUMERICAL_ISSUE,
                    "Non-finite values encountered in tableau.",
                    iteration - 1,
                    history,
                )

            reduced_costs = tableau.reduced_costs()
            nonbasic = tableau.nonbasic_indices

            if is_optimal(
                reduced_costs=reduced_costs,
                nonbasic_indices=nonbasic,
                objective_sense=objective_sense,
                tolerance=self.tolerance,
            ):
                return STATUS_OPTIMAL, "Optimality condition reached.", iteration - 1, history

            entering_idx = select_entering_variable(
                reduced_costs=reduced_costs,
                nonbasic_indices=nonbasic,
                objective_sense=objective_sense,
                rule=self.pivot_rule,
                tolerance=self.tolerance,
                use_bland=use_bland_now,
            )

            if entering_idx is None:
                return STATUS_OPTIMAL, "No improving entering variable found.", iteration - 1, history

            pivot_column = tableau.column_values(entering_idx)
            if is_unbounded_pivot_column(pivot_column, tolerance=self.tolerance):
                return STATUS_UNBOUNDED, "Unbounded pivot column detected.", iteration - 1, history

            leaving_row = minimum_ratio_test(
                pivot_column=pivot_column,
                rhs_values=tableau.rhs_values(),
                basis_indices=tableau.basis_indices,
                tolerance=self.tolerance,
                use_bland=use_bland_now,
            )

            if leaving_row is None:
                return STATUS_UNBOUNDED, "No valid leaving row from ratio test.", iteration - 1, history

            try:
                tableau.pivot(leaving_row, entering_idx)
            except ZeroDivisionError:
                return (
                    STATUS_NUMERICAL_ISSUE,
                    "Numerical zero pivot encountered.",
                    iteration - 1,
                    history,
                )

            current_basis = tuple(tableau.basis_indices)
            if detect_basis_cycle(basis_history, current_basis):
                use_bland_now = True
            basis_history.append(current_basis)

        return STATUS_ITERATION_LIMIT, "Maximum simplex iterations reached.", max_iterations, history

    def _derive_feasible_basis(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_candidate: np.ndarray,
    ) -> Optional[List[int]]:
        """Recover a feasible full-rank basis from candidate variable values."""
        m, n = A.shape
        if m == 0:
            return []

        positive_candidates = [
            idx for idx, value in enumerate(x_candidate[:n]) if value > self.tolerance
        ]
        remaining_candidates = [idx for idx in range(n) if idx not in positive_candidates]
        ordered_candidates = positive_candidates + remaining_candidates

        basis: List[int] = []
        current_rank = 0

        for idx in ordered_candidates:
            trial = basis + [idx]
            rank = np.linalg.matrix_rank(A[:, trial], tol=self.tolerance)
            if rank > current_rank:
                basis.append(idx)
                current_rank = rank
            if len(basis) == m:
                break

        if len(basis) < m:
            return None

        try:
            x_basic = np.linalg.solve(A[:, basis], b)
        except np.linalg.LinAlgError:
            return None

        if np.any(x_basic < -self.tolerance):
            return None

        return basis

    def _build_result(
        self,
        lp: StandardFormLP,
        status: str,
        message: str,
        values: np.ndarray,
        basis_indices: Sequence[int],
        iterations: int,
        objective_value: Optional[float],
        history: Optional[List[Dict[str, Any]]],
        messages: Optional[List[str]] = None,
    ) -> SimplexResult:
        """Construct a standardized solver result object."""
        variable_values = np.asarray(values, dtype=float)
        name_mapping = {
            lp.variable_names[idx]: float(variable_values[idx])
            for idx in range(min(len(lp.variable_names), variable_values.shape[0]))
        }

        diagnostics = build_diagnostics(
            status=status,
            message=message,
            iterations=iterations,
            objective_value=objective_value,
            basis_indices=list(basis_indices),
            extra={
                "num_variables": lp.A.shape[1],
                "num_constraints": lp.A.shape[0],
                "objective_sense": lp.objective_sense,
            },
        )

        final_messages = list(messages or [])
        final_messages.append(message)

        return SimplexResult(
            status=status,
            objective_value=objective_value,
            variable_values=variable_values,
            variable_values_by_name=name_mapping,
            basis_indices=list(basis_indices),
            iterations=int(iterations),
            tableau_history=history,
            diagnostics=diagnostics,
            messages=final_messages,
        )
