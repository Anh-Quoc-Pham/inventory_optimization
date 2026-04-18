from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adapters.simplex_solution_to_transfer_plan import (
    decode_simplex_solution_to_transfer_plan,
)
from src.config import (
    GA_CROSSOVER_PROB,
    GA_GENERATIONS,
    GA_MUTATION_PROB,
    GA_POPULATION_SIZE,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    RANDOM_SEED,
    SIMPLEX_DUMMY_EXCESS_COST,
    SIMPLEX_DUMMY_SHORTAGE_COST,
    SIMPLEX_MAX_ITERATIONS,
    SIMPLEX_PIVOT_RULE,
    SIMPLEX_TOLERANCE,
    SIMPLEX_USE_BLAND,
)
from src.create_transfer_network import (
    aggregate_transfers,
    create_network_graph,
    export_graphml,
    generate_network_summary,
)
from src.main import (
    run_analysis,
    run_ga_optimization,
    run_rule_based_optimization,
    run_simplex_optimization,
)
from src.simplex.lp_standard_form import transportation_instance_to_standard_form
from src.simplex.primal_simplex_solver import PrimalSimplexSolver
from src.transport.transportation_instance_builder import build_transportation_instances
from src.visualize_network import load_graphml_and_visualize


def _build_args(
    data_dir: Path,
    results_dir: Path,
    seed: int,
    min_days: int,
    max_days: int,
    ga_population: int,
    ga_generations: int,
    ga_crossover: float,
    ga_mutation: float,
    simplex_max_iterations: int,
    simplex_tolerance: float,
    simplex_pivot_rule: str,
    simplex_use_bland: bool,
    simplex_keep_history: bool,
    simplex_dummy_excess_cost: float,
    simplex_dummy_shortage_cost: float | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(data_dir),
        results_dir=str(results_dir),
        seed=seed,
        min_days=min_days,
        max_days=max_days,
        ga_population=ga_population,
        ga_generations=ga_generations,
        ga_crossover=ga_crossover,
        ga_mutation=ga_mutation,
        simplex_max_iterations=simplex_max_iterations,
        simplex_tolerance=simplex_tolerance,
        simplex_pivot_rule=simplex_pivot_rule,
        simplex_use_bland=simplex_use_bland,
        simplex_keep_history=simplex_keep_history,
        simplex_dummy_excess_cost=simplex_dummy_excess_cost,
        simplex_dummy_shortage_cost=simplex_dummy_shortage_cost,
    )


def _load_matrices(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance_matrix = pd.read_csv(data_dir / "distance_matrix.csv", index_col=0)
    transport_cost_matrix = pd.read_csv(
        data_dir / "transport_cost_matrix.csv", index_col=0
    )

    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)
    transport_cost_matrix.index = transport_cost_matrix.index.astype(int)
    transport_cost_matrix.columns = transport_cost_matrix.columns.astype(int)

    return distance_matrix, transport_cost_matrix


def _compute_coverage_rate(
    transfer_plan: pd.DataFrame,
    needed_inventory: pd.DataFrame,
) -> float:
    if needed_inventory is None or needed_inventory.empty:
        return 1.0

    if transfer_plan is None or transfer_plan.empty:
        return 0.0

    needed = (
        needed_inventory.groupby(["store_id", "product_id"], as_index=False)[
            "needed_units"
        ]
        .sum()
        .rename(columns={"store_id": "to_store_id"})
    )

    received = (
        transfer_plan.groupby(["to_store_id", "product_id"], as_index=False)["units"]
        .sum()
        .rename(columns={"units": "received_units"})
    )

    merged = needed.merge(received, on=["to_store_id", "product_id"], how="left")
    merged["received_units"] = merged["received_units"].fillna(0)
    merged["covered_units"] = np.minimum(
        merged["needed_units"], merged["received_units"]
    )

    total_needed = float(merged["needed_units"].sum())
    total_covered = float(merged["covered_units"].sum())

    if total_needed <= 0:
        return 1.0

    return total_covered / total_needed


def _extract_balanced_improvement(impact_df: pd.DataFrame | None) -> float | None:
    if impact_df is None or impact_df.empty:
        return None

    try:
        value = impact_df.loc["Increase in Balanced Items", "Improvement"]
        return float(value)
    except (KeyError, TypeError, ValueError):
        return None


def _enrich_transfer_plan(
    transfer_plan: pd.DataFrame,
    stores_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    if transfer_plan is None or transfer_plan.empty:
        return pd.DataFrame()

    store_name_map = stores_df.set_index("store_id")["store_name"].to_dict()
    product_name_map = products_df.set_index("product_id")["product_name"].to_dict()

    enriched = transfer_plan.copy()
    enriched["from_store_name"] = enriched["from_store_id"].map(store_name_map)
    enriched["to_store_name"] = enriched["to_store_id"].map(store_name_map)
    enriched["product_name"] = enriched["product_id"].map(product_name_map)

    return enriched


def run_manual_simplex_case() -> Dict[str, Any]:
    labels = [1, 2, 3, 4]

    excess_df = pd.DataFrame(
        [
            {"store_id": 1, "product_id": 999, "excess_units": 20},
            {"store_id": 2, "product_id": 999, "excess_units": 30},
        ]
    )

    needed_df = pd.DataFrame(
        [
            {"store_id": 3, "product_id": 999, "needed_units": 25},
            {"store_id": 4, "product_id": 999, "needed_units": 25},
        ]
    )

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

    if not instances:
        return {
            "status": "FAILED",
            "objective_value": None,
            "iterations": None,
            "expected_objective": 85.0,
            "transfer_plan": pd.DataFrame(),
        }

    instance = instances[0]
    lp = transportation_instance_to_standard_form(instance)
    solver = PrimalSimplexSolver(max_iterations=500, tolerance=1e-9, use_bland=True)
    result = solver.solve(lp)

    decoded = decode_simplex_solution_to_transfer_plan(
        simplex_result=result,
        standard_form_lp=lp,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )

    return {
        "status": result.status,
        "objective_value": result.objective_value,
        "iterations": result.iterations,
        "expected_objective": 85.0,
        "transfer_plan": decoded,
    }


def _save_cost_heatmap(instances: list, output_dir: Path) -> Path | None:
    if not instances:
        return None

    first_instance = instances[0]
    cost_matrix = first_instance.cost_matrix

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cost_matrix, cmap="YlOrRd")

    ax.set_title(
        f"Transportation Cost Matrix Snapshot - Product {first_instance.product_id}",
        fontsize=11,
    )
    ax.set_xlabel("Demand Nodes")
    ax.set_ylabel("Supply Nodes")

    ax.set_xticks(np.arange(len(first_instance.demand_store_ids)))
    ax.set_yticks(np.arange(len(first_instance.source_store_ids)))
    ax.set_xticklabels([str(x) for x in first_instance.demand_store_ids], rotation=45)
    ax.set_yticklabels([str(x) for x in first_instance.source_store_ids])

    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{cost_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, label="Unit Transport Cost")
    fig.tight_layout()

    output_path = output_dir / (
        f"chapter3_cost_heatmap_product_{first_instance.product_id}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _format_simplex_status(status_df: pd.DataFrame | None) -> str:
    if status_df is None or status_df.empty:
        return "NO_STATUS"

    counts = status_df["status"].value_counts().to_dict()
    total = int(len(status_df))

    optimal_count = int(counts.get("OPTIMAL", 0))
    error_count = int(counts.get("ERROR", 0))
    unbounded_count = int(counts.get("UNBOUNDED", 0))
    infeasible_count = int(counts.get("INFEASIBLE", 0))

    parts = [f"OPTIMAL {optimal_count}/{total}"]

    if error_count > 0:
        parts.append(f"ERROR {error_count}")
    if unbounded_count > 0:
        parts.append(f"UNBOUNDED {unbounded_count}")
    if infeasible_count > 0:
        parts.append(f"INFEASIBLE {infeasible_count}")

    return " | ".join(parts)


def run_chapter3_benchmark(
    data_dir: str | Path = "data",
    results_dir: str | Path = "results/chapter3_benchmark",
    vis_dir: str | Path = "visualizations/chapter3_benchmark",
    seed: int = RANDOM_SEED,
    min_days: int = MIN_INVENTORY_DAYS,
    max_days: int = MAX_INVENTORY_DAYS,
    ga_population: int = GA_POPULATION_SIZE,
    ga_generations: int = GA_GENERATIONS,
    ga_crossover: float = GA_CROSSOVER_PROB,
    ga_mutation: float = GA_MUTATION_PROB,
    simplex_max_iterations: int = SIMPLEX_MAX_ITERATIONS,
    simplex_tolerance: float = SIMPLEX_TOLERANCE,
    simplex_pivot_rule: str = SIMPLEX_PIVOT_RULE,
    simplex_use_bland: bool = SIMPLEX_USE_BLAND,
    simplex_keep_history: bool = False,
    simplex_dummy_excess_cost: float = SIMPLEX_DUMMY_EXCESS_COST,
    simplex_dummy_shortage_cost: float | None = SIMPLEX_DUMMY_SHORTAGE_COST,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    vis_dir = Path(vis_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    args = _build_args(
        data_dir=data_dir,
        results_dir=results_dir,
        seed=seed,
        min_days=min_days,
        max_days=max_days,
        ga_population=ga_population,
        ga_generations=ga_generations,
        ga_crossover=ga_crossover,
        ga_mutation=ga_mutation,
        simplex_max_iterations=simplex_max_iterations,
        simplex_tolerance=simplex_tolerance,
        simplex_pivot_rule=simplex_pivot_rule,
        simplex_use_bland=simplex_use_bland,
        simplex_keep_history=simplex_keep_history,
        simplex_dummy_excess_cost=simplex_dummy_excess_cost,
        simplex_dummy_shortage_cost=simplex_dummy_shortage_cost,
    )

    # Data and inventory analysis
    analyzer, analysis_df, excess_df, needed_df = run_analysis(args)

    stores_df = pd.read_csv(data_dir / "stores.csv")
    products_df = pd.read_csv(data_dir / "products.csv")
    sales_df = pd.read_csv(data_dir / "sales_data.csv")
    inventory_df = pd.read_csv(data_dir / "inventory_data.csv")

    distance_matrix, transport_cost_matrix = _load_matrices(data_dir)

    instances = build_transportation_instances(
        excess_inventory=excess_df,
        needed_inventory=needed_df,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
        allow_self_transfer=False,
        dummy_shortage_cost=simplex_dummy_shortage_cost,
        dummy_excess_cost=simplex_dummy_excess_cost,
    )

    valid_products = sorted(
        set(excess_df["product_id"].unique()) & set(needed_df["product_id"].unique())
    )

    valid_excess = excess_df[excess_df["product_id"].isin(valid_products)]
    valid_needed = needed_df[needed_df["product_id"].isin(valid_products)]

    summary_df = pd.DataFrame(
        [
            {
                "total_stores": int(stores_df["store_id"].nunique()),
                "total_products": int(products_df["product_id"].nunique()),
                "sales_records": int(len(sales_df)),
                "inventory_records": int(len(inventory_df)),
                "analysis_rows": int(len(analysis_df)),
                "excess_rows": int(len(excess_df)),
                "needed_rows": int(len(needed_df)),
                "total_excess_units": float(excess_df["excess_units"].sum()),
                "total_needed_units": float(needed_df["needed_units"].sum()),
                "valid_products": int(len(valid_products)),
                "source_points": int(len(valid_excess)),
                "demand_points": int(len(valid_needed)),
                "source_stores": int(valid_excess["store_id"].nunique()),
                "demand_stores": int(valid_needed["store_id"].nunique()),
                "transport_instances": int(len(instances)),
            }
        ]
    )

    summary_csv_path = results_dir / "chapter3_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Manual simplex case for near hand-calculation demonstration
    manual_case = run_manual_simplex_case()
    manual_plan = manual_case["transfer_plan"]
    manual_plan_path = results_dir / "chapter3_manual_case_transfer_plan.csv"
    if manual_plan is not None and not manual_plan.empty:
        manual_plan.to_csv(manual_plan_path, index=False)

    # Run 3 algorithms using exact runtime code paths
    algo_outputs: Dict[str, Dict[str, Any]] = {}

    rb_start = time.perf_counter()
    rb_plan, rb_impact = run_rule_based_optimization(analyzer, excess_df, needed_df, args)
    rb_elapsed = time.perf_counter() - rb_start
    algo_outputs["Rule-Based"] = {
        "transfer_plan": rb_plan,
        "impact": rb_impact,
        "elapsed": rb_elapsed,
        "status_solver": "COMPLETED" if rb_plan is not None else "FAILED",
    }

    ga_start = time.perf_counter()
    ga_plan, ga_impact = run_ga_optimization(analyzer, excess_df, needed_df, args)
    ga_elapsed = time.perf_counter() - ga_start
    algo_outputs["Genetic Algorithm"] = {
        "transfer_plan": ga_plan,
        "impact": ga_impact,
        "elapsed": ga_elapsed,
        "status_solver": "COMPLETED" if ga_plan is not None else "FAILED",
    }

    sp_start = time.perf_counter()
    sp_plan, sp_impact = run_simplex_optimization(analyzer, excess_df, needed_df, args)
    sp_elapsed = time.perf_counter() - sp_start

    simplex_status_path = results_dir / "simplex_product_status.csv"
    simplex_status_df = (
        pd.read_csv(simplex_status_path) if simplex_status_path.exists() else None
    )
    simplex_status_text = _format_simplex_status(simplex_status_df)

    algo_outputs["Primal Simplex"] = {
        "transfer_plan": sp_plan,
        "impact": sp_impact,
        "elapsed": sp_elapsed,
        "status_solver": simplex_status_text,
        "simplex_status_df": simplex_status_df,
    }

    # Build benchmark comparison table
    rows = []
    combined_transfer_plans = []

    for algorithm, payload in algo_outputs.items():
        transfer_plan = payload["transfer_plan"]
        impact_df = payload["impact"]

        if transfer_plan is None or transfer_plan.empty:
            total_cost = np.nan
            number_of_transfers = 0
            total_units = 0
            solver_status = (
                "NO_TRANSFER" if algorithm != "Primal Simplex" else payload["status_solver"]
            )
        else:
            total_cost = (
                float(transfer_plan["transport_cost"].sum())
                if "transport_cost" in transfer_plan.columns
                else np.nan
            )
            number_of_transfers = int(len(transfer_plan))
            total_units = (
                int(transfer_plan["units"].sum())
                if "units" in transfer_plan.columns
                else 0
            )
            solver_status = payload["status_solver"]

            enriched = _enrich_transfer_plan(transfer_plan, stores_df, products_df)
            enriched["algorithm"] = algorithm
            combined_transfer_plans.append(enriched)

        coverage_rate = _compute_coverage_rate(transfer_plan, needed_df)
        balanced_improvement = _extract_balanced_improvement(impact_df)

        rows.append(
            {
                "algorithm": algorithm,
                "valid_products": int(len(valid_products)),
                "source_points": int(len(valid_excess)),
                "demand_points": int(len(valid_needed)),
                "total_transport_cost_vnd": total_cost,
                "number_of_transfers": number_of_transfers,
                "total_units_transferred": total_units,
                "coverage_rate": coverage_rate,
                "execution_time_sec": float(payload["elapsed"]),
                "status_solver": solver_status,
                "balanced_items_improvement": balanced_improvement,
            }
        )

    comparison_df = pd.DataFrame(rows)

    # Ranking for best plan selection: highest coverage, then lowest cost
    ranking_df = comparison_df.copy()
    ranking_df["rank_cost"] = ranking_df["total_transport_cost_vnd"].fillna(np.inf)
    ranking_df = ranking_df.sort_values(
        ["coverage_rate", "rank_cost", "execution_time_sec"],
        ascending=[False, True, True],
    )
    best_algorithm = ranking_df.iloc[0]["algorithm"]

    comparison_csv_path = results_dir / "chapter3_algorithm_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    transfer_comparison_csv = results_dir / "chapter3_transfer_plan_comparison.csv"
    if combined_transfer_plans:
        pd.concat(combined_transfer_plans, ignore_index=True).to_csv(
            transfer_comparison_csv, index=False
        )
    else:
        pd.DataFrame().to_csv(transfer_comparison_csv, index=False)

    # Save best transfer plan with names for reporting and network visuals
    best_plan = algo_outputs[best_algorithm]["transfer_plan"]
    best_plan_enriched = _enrich_transfer_plan(best_plan, stores_df, products_df)
    best_plan_csv = results_dir / "chapter3_best_transfer_plan.csv"
    if best_plan_enriched is not None and not best_plan_enriched.empty:
        best_plan_enriched.to_csv(best_plan_csv, index=False)

    # Network artifacts
    graphml_path = None
    network_summary_path = None

    if best_plan_enriched is not None and not best_plan_enriched.empty:
        edge_data, node_data = aggregate_transfers(best_plan_enriched)
        network_graph = create_network_graph(edge_data, node_data)

        graphml_path = vis_dir / "chapter3_store_transfer_network.graphml"
        export_graphml(network_graph, str(graphml_path))

        generate_network_summary(network_graph, str(vis_dir))
        network_summary_path = vis_dir / "network_analysis_summary.txt"

        load_graphml_and_visualize(str(graphml_path), str(vis_dir))

    heatmap_path = _save_cost_heatmap(instances, vis_dir)

    artifact_paths = {
        "summary_csv": str(summary_csv_path),
        "comparison_csv": str(comparison_csv_path),
        "transfer_comparison_csv": str(transfer_comparison_csv),
        "best_transfer_plan_csv": str(best_plan_csv),
        "manual_case_transfer_plan_csv": str(manual_plan_path),
        "simplex_product_status_csv": str(simplex_status_path),
        "network_graphml": str(graphml_path) if graphml_path else None,
        "network_overview_png": str(vis_dir / "network_overview.png"),
        "hub_analysis_png": str(vis_dir / "hub_analysis.png"),
        "flow_intensity_png": str(vis_dir / "flow_intensity_map.png"),
        "network_summary_txt": str(network_summary_path)
        if network_summary_path
        else None,
        "cost_heatmap_png": str(heatmap_path) if heatmap_path else None,
    }

    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "best_algorithm": best_algorithm,
        "manual_case": {
            "status": manual_case["status"],
            "objective_value": manual_case["objective_value"],
            "expected_objective": manual_case["expected_objective"],
            "iterations": manual_case["iterations"],
        },
        "artifacts": artifact_paths,
    }

    metadata_path = results_dir / "chapter3_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=True)

    artifact_paths["metadata_json"] = str(metadata_path)

    return {
        "analysis_df": analysis_df,
        "excess_df": excess_df,
        "needed_df": needed_df,
        "instances": instances,
        "manual_case": manual_case,
        "comparison_df": comparison_df,
        "summary_df": summary_df,
        "best_algorithm": best_algorithm,
        "artifact_paths": artifact_paths,
        "algo_outputs": algo_outputs,
    }
