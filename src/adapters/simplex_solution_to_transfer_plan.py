from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.simplex.lp_standard_form import StandardFormLP
from src.simplex.primal_simplex_solver import SimplexResult

TRANSFER_PLAN_COLUMNS = [
    "from_store_id",
    "to_store_id",
    "product_id",
    "units",
    "distance_km",
    "transport_cost",
]


def _is_dummy_store(store_id: Any) -> bool:
    """Identify synthetic dummy nodes used for balancing transportation LPs."""
    return isinstance(store_id, str) and store_id.upper().startswith("DUMMY")


def _safe_lookup(
    matrix_df: pd.DataFrame,
    row_key: Any,
    col_key: Any,
    default_value: float,
) -> float:
    """Safely lookup matrix values for route metrics."""
    if matrix_df is None or matrix_df.empty:
        return default_value

    if row_key not in matrix_df.index or col_key not in matrix_df.columns:
        return default_value

    value = pd.to_numeric(matrix_df.loc[row_key, col_key], errors="coerce")
    if pd.isna(value):
        return default_value

    return float(value)


def decode_simplex_solution_to_transfer_plan(
    simplex_result: SimplexResult,
    standard_form_lp: StandardFormLP,
    distance_matrix: pd.DataFrame,
    transport_cost_matrix: pd.DataFrame,
    min_unit_threshold: float = 1e-6,
) -> pd.DataFrame:
    """
    Decode simplex variable values into the transfer_plan schema used by legacy pipeline.

    Dummy-node flows are removed from business output but kept in diagnostics metadata.
    """
    if simplex_result is None or simplex_result.status != "OPTIMAL":
        return pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)

    if simplex_result.variable_values.size == 0:
        return pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)

    variable_to_arc: Dict[int, Any] = standard_form_lp.metadata.get("variable_to_arc", {})

    if not variable_to_arc:
        return pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)

    rows: List[Dict[str, Any]] = []
    dummy_flow_rows: List[Dict[str, Any]] = []

    variable_values = simplex_result.variable_values
    original_count = int(
        standard_form_lp.metadata.get("original_variable_count", len(variable_to_arc))
    )

    for var_idx in range(min(original_count, variable_values.shape[0])):
        value = float(variable_values[var_idx])

        if value <= min_unit_threshold:
            continue

        arc = variable_to_arc.get(var_idx)
        if arc is None:
            continue

        from_store_id, to_store_id, product_id = arc

        if _is_dummy_store(from_store_id) or _is_dummy_store(to_store_id):
            dummy_flow_rows.append(
                {
                    "from_store_id": from_store_id,
                    "to_store_id": to_store_id,
                    "product_id": product_id,
                    "units": value,
                }
            )
            continue

        # Inventory quantities are integer by business schema.
        units = int(round(value))
        if units <= 0:
            continue

        distance_km = _safe_lookup(distance_matrix, from_store_id, to_store_id, 0.0)
        unit_cost = _safe_lookup(
            transport_cost_matrix,
            from_store_id,
            to_store_id,
            float(standard_form_lp.c[var_idx]),
        )
        transport_cost = float(unit_cost * units)

        rows.append(
            {
                "from_store_id": int(from_store_id),
                "to_store_id": int(to_store_id),
                "product_id": int(product_id),
                "units": int(units),
                "distance_km": float(distance_km),
                "transport_cost": float(transport_cost),
            }
        )

    result_df = pd.DataFrame(rows, columns=TRANSFER_PLAN_COLUMNS)

    # Keep diagnostics for analysis/reporting without changing existing schema.
    simplex_result.diagnostics["dummy_flows"] = dummy_flow_rows
    simplex_result.diagnostics["dummy_flow_count"] = len(dummy_flow_rows)

    return result_df


def merge_transfer_plans(plans: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge product-level transfer plans into one schema-compatible DataFrame."""
    valid_plans = [
        plan for plan in plans if plan is not None and not plan.empty and set(TRANSFER_PLAN_COLUMNS).issubset(plan.columns)
    ]

    if not valid_plans:
        return pd.DataFrame(columns=TRANSFER_PLAN_COLUMNS)

    combined = pd.concat(valid_plans, ignore_index=True)

    merged = (
        combined.groupby(["from_store_id", "to_store_id", "product_id"], as_index=False)
        .agg(
            {
                "units": "sum",
                "distance_km": "mean",
                "transport_cost": "sum",
            }
        )
        .sort_values("transport_cost", ascending=False)
    )

    merged = merged[merged["units"] > 0].copy()
    merged["units"] = merged["units"].astype(int)

    return merged[TRANSFER_PLAN_COLUMNS]
