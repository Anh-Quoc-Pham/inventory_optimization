from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_optimization_logger

StoreId = Union[int, str]


@dataclass
class TransportationInstance:
    """Represents one transportation LP instance for a single product."""

    product_id: int
    source_store_ids: List[StoreId]
    demand_store_ids: List[StoreId]
    supply_vector: np.ndarray
    demand_vector: np.ndarray
    cost_matrix: np.ndarray
    distance_matrix: np.ndarray
    balanced: bool
    dummy_added: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_store_id(value: Any) -> StoreId:
    """Normalize store IDs to int when possible."""
    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float) and float(value).is_integer():
        return int(value)

    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        return text

    return value


def _prepare_inventory_frame(df: pd.DataFrame, units_col: str) -> pd.DataFrame:
    """Aggregate store-product units into one clean frame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["store_id", "product_id", units_col])

    required_cols = {"store_id", "product_id", units_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for transportation build: {sorted(missing_cols)}"
        )

    cleaned = df[["store_id", "product_id", units_col]].copy()
    cleaned["store_id"] = cleaned["store_id"].apply(_normalize_store_id)
    cleaned["product_id"] = cleaned["product_id"].astype(int)
    cleaned[units_col] = pd.to_numeric(cleaned[units_col], errors="coerce").fillna(0)

    grouped = (
        cleaned.groupby(["store_id", "product_id"], as_index=False)[units_col]
        .sum()
        .sort_values(["product_id", "store_id"])
    )

    return grouped[grouped[units_col] > 0].reset_index(drop=True)


def _infer_shortage_penalty(
    transport_cost_matrix: pd.DataFrame,
    fallback_penalty: float = 1_000_000.0,
) -> float:
    """Infer default shortage penalty from existing transport costs."""
    if transport_cost_matrix is None or transport_cost_matrix.empty:
        return fallback_penalty

    values = pd.to_numeric(
        transport_cost_matrix.values.reshape(-1), errors="coerce"
    ).astype(float)
    finite_positive = values[np.isfinite(values) & (values > 0)]

    if finite_positive.size == 0:
        return fallback_penalty

    return float(np.max(finite_positive) * 2.0)


def _safe_matrix_lookup(
    matrix_df: pd.DataFrame,
    row_key: StoreId,
    col_key: StoreId,
    default_value: float,
) -> float:
    """Read matrix values safely with a fallback for missing links."""
    if matrix_df is None or matrix_df.empty:
        return default_value

    if row_key not in matrix_df.index or col_key not in matrix_df.columns:
        return default_value

    raw_value = matrix_df.loc[row_key, col_key]
    numeric_value = pd.to_numeric(raw_value, errors="coerce")

    if pd.isna(numeric_value):
        return default_value

    return float(numeric_value)


def build_transportation_instances(
    excess_inventory: pd.DataFrame,
    needed_inventory: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    transport_cost_matrix: pd.DataFrame,
    allow_self_transfer: bool = False,
    dummy_shortage_cost: Optional[float] = None,
    dummy_excess_cost: float = 0.0,
) -> List[TransportationInstance]:
    """
    Build transportation instances per product_id from analyzer outputs.

    Args:
        excess_inventory: DataFrame containing `excess_units` and store/product keys.
        needed_inventory: DataFrame containing `needed_units` and store/product keys.
        distance_matrix: Store-to-store distance matrix DataFrame.
        transport_cost_matrix: Store-to-store unit cost matrix DataFrame.
        allow_self_transfer: Whether to allow source==destination arcs.
        dummy_shortage_cost: Penalty for dummy-source arcs when demand > supply.
        dummy_excess_cost: Cost for dummy-destination arcs when supply > demand.

    Returns:
        List of TransportationInstance objects (one per valid product).
    """
    logger_system = get_optimization_logger()

    excess_df = _prepare_inventory_frame(excess_inventory, "excess_units")
    needed_df = _prepare_inventory_frame(needed_inventory, "needed_units")

    if excess_df.empty or needed_df.empty:
        logger_system.log_progress(
            "simplex_transport_builder",
            "No valid excess or needed inventory rows. 0 transportation instances created.",
        )
        return []

    valid_products = sorted(
        set(excess_df["product_id"].unique())
        & set(needed_df["product_id"].unique())
    )

    logger_system.log_progress(
        "simplex_transport_builder",
        f"Valid product count for transportation instances: {len(valid_products)}",
    )

    if dummy_shortage_cost is None:
        dummy_shortage_cost = _infer_shortage_penalty(transport_cost_matrix)

    instances: List[TransportationInstance] = []

    for product_id in valid_products:
        product_excess = excess_df[excess_df["product_id"] == product_id].copy()
        product_needed = needed_df[needed_df["product_id"] == product_id].copy()

        if product_excess.empty or product_needed.empty:
            continue

        product_excess = (
            product_excess.groupby("store_id", as_index=False)["excess_units"]
            .sum()
            .sort_values("store_id")
        )
        product_needed = (
            product_needed.groupby("store_id", as_index=False)["needed_units"]
            .sum()
            .sort_values("store_id")
        )

        source_store_ids = product_excess["store_id"].tolist()
        demand_store_ids = product_needed["store_id"].tolist()

        supply_vector = product_excess["excess_units"].to_numpy(dtype=float)
        demand_vector = product_needed["needed_units"].to_numpy(dtype=float)

        if supply_vector.sum() <= 0 or demand_vector.sum() <= 0:
            continue

        fallback_cost = _infer_shortage_penalty(transport_cost_matrix)
        forbidden_arc_cost = fallback_cost * 10.0

        cost_matrix = np.zeros(
            (len(source_store_ids), len(demand_store_ids)), dtype=float
        )
        distances = np.zeros((len(source_store_ids), len(demand_store_ids)), dtype=float)

        invalid_cost_links = 0
        forbidden_self_transfer_links = 0

        for i, source_id in enumerate(source_store_ids):
            for j, demand_id in enumerate(demand_store_ids):
                if not allow_self_transfer and source_id == demand_id:
                    forbidden_self_transfer_links += 1
                    cost_matrix[i, j] = forbidden_arc_cost
                    distances[i, j] = 0.0
                    continue

                distances[i, j] = _safe_matrix_lookup(
                    distance_matrix, source_id, demand_id, 0.0
                )
                cost_value = _safe_matrix_lookup(
                    transport_cost_matrix, source_id, demand_id, np.nan
                )

                if not np.isfinite(cost_value) or cost_value < 0:
                    invalid_cost_links += 1
                    cost_value = fallback_cost

                cost_matrix[i, j] = float(cost_value)

        total_supply = float(supply_vector.sum())
        total_demand = float(demand_vector.sum())
        balance_gap = total_supply - total_demand

        dummy_added = "none"
        balanced = np.isclose(total_supply, total_demand)

        if balance_gap > 1e-9:
            # Supply > demand: add dummy destination to absorb excess inventory.
            dummy_added = "dummy_destination"
            demand_store_ids = demand_store_ids + ["DUMMY_DEMAND"]
            demand_vector = np.append(demand_vector, balance_gap)
            cost_matrix = np.column_stack(
                [
                    cost_matrix,
                    np.full((cost_matrix.shape[0],), float(dummy_excess_cost)),
                ]
            )
            distances = np.column_stack(
                [distances, np.zeros((distances.shape[0],), dtype=float)]
            )

        elif balance_gap < -1e-9:
            # Demand > supply: add dummy source with shortage penalty.
            dummy_added = "dummy_source"
            source_store_ids = source_store_ids + ["DUMMY_SOURCE"]
            supply_vector = np.append(supply_vector, abs(balance_gap))
            cost_matrix = np.vstack(
                [
                    cost_matrix,
                    np.full((cost_matrix.shape[1],), float(dummy_shortage_cost)),
                ]
            )
            distances = np.vstack(
                [distances, np.zeros((distances.shape[1],), dtype=float)]
            )

        instance = TransportationInstance(
            product_id=int(product_id),
            source_store_ids=source_store_ids,
            demand_store_ids=demand_store_ids,
            supply_vector=supply_vector.astype(float),
            demand_vector=demand_vector.astype(float),
            cost_matrix=cost_matrix.astype(float),
            distance_matrix=distances.astype(float),
            balanced=balanced,
            dummy_added=dummy_added,
            metadata={
                "product_id": int(product_id),
                "original_total_supply": total_supply,
                "original_total_demand": total_demand,
                "post_balance_total_supply": float(supply_vector.sum()),
                "post_balance_total_demand": float(demand_vector.sum()),
                "balance_gap": balance_gap,
                "allow_self_transfer": allow_self_transfer,
                "invalid_cost_links": invalid_cost_links,
                "forbidden_self_transfer_links": forbidden_self_transfer_links,
                "dummy_shortage_cost": float(dummy_shortage_cost),
                "dummy_excess_cost": float(dummy_excess_cost),
                "real_source_store_ids": [
                    store_id
                    for store_id in source_store_ids
                    if store_id != "DUMMY_SOURCE"
                ],
                "real_demand_store_ids": [
                    store_id
                    for store_id in demand_store_ids
                    if store_id != "DUMMY_DEMAND"
                ],
            },
        )

        instances.append(instance)

    logger_system.log_progress(
        "simplex_transport_builder",
        f"Transportation instances created: {len(instances)}",
    )

    return instances
