import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.adapters.simplex_solution_to_transfer_plan import (
    decode_simplex_solution_to_transfer_plan,
    merge_transfer_plans,
)

# Import configuration
from src.config import (
    DATA_DIR,
    EXCESS_PERCENT,
    GA_CROSSOVER_PROB,
    GA_GENERATIONS,
    GA_MUTATION_PROB,
    GA_POPULATION_SIZE,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    NUM_PRODUCTS,
    RANDOM_SEED,
    REQUIRED_DATA_FILES,
    RESULTS_DIR,
    SALES_DAYS,
    SHORTAGE_PERCENT,
    SIMPLEX_DUMMY_EXCESS_COST,
    SIMPLEX_DUMMY_SHORTAGE_COST,
    SIMPLEX_MAX_ITERATIONS,
    SIMPLEX_PIVOT_RULE,
    SIMPLEX_TOLERANCE,
    SIMPLEX_USE_BLAND,
    VISUALIZATIONS_DIR,
    create_directories,
)

# Import components
from src.data_generator.data_generator_main import generate_all_data
from src.engine.analyzer import InventoryAnalyzer
from src.engine.genetic_algorithm import GeneticAlgorithmOptimizer
from src.engine.results_manager import ResultsManager
from src.engine.rule_based import RuleBasedOptimizer
from src.simplex.lp_standard_form import transportation_instance_to_standard_form
from src.simplex.primal_simplex_solver import PrimalSimplexSolver
from src.transport.transportation_instance_builder import build_transportation_instances
from src.utils.logger import get_optimization_logger
from src.engine.modi_optimizer import MODIOptimizer

def setup_directories():
    """Create necessary directories using config."""
    return create_directories()


def run_data_generation(args):
    """Run data generation."""
    print("\n=== DATA GENERATION ===")
    generate_all_data(
        num_products=args.products,
        days=args.days,
        output_dir=args.data_dir,
        random_seed=args.seed,
        min_days=args.min_days,
        max_days=args.max_days,
        excess_percent=args.excess_percent,
        shortage_percent=args.shortage_percent,
    )


def run_analysis(args):
    """Run inventory analysis."""
    print("\n=== INVENTORY ANALYSIS ===")

    # Create analyzer
    analyzer = InventoryAnalyzer()

    # Load data
    analyzer.load_data(
        sales_path=os.path.join(args.data_dir, "sales_data.csv"),
        inventory_path=os.path.join(args.data_dir, "inventory_data.csv"),
        stores_path=os.path.join(args.data_dir, "stores.csv"),
        products_path=os.path.join(args.data_dir, "products.csv"),
    )

    # Analyze data
    analysis_df = analyzer.analyze_sales_data()

    # Identify imbalances
    excess_df, needed_df = analyzer.identify_inventory_imbalances(
        min_days=args.min_days, max_days=args.max_days
    )

    # Save analysis results
    analysis_df.to_csv(
        os.path.join(args.results_dir, "inventory_analysis.csv"), index=False
    )
    excess_df.to_csv(
        os.path.join(args.results_dir, "excess_inventory.csv"), index=False
    )
    needed_df.to_csv(
        os.path.join(args.results_dir, "needed_inventory.csv"), index=False
    )

    # Calculate total excess and needed units
    excess_units = excess_df["excess_units"].sum()
    needed_units = needed_df["needed_units"].sum()
    excess_to_needed_ratio = excess_units / needed_units if needed_units > 0 else float(
        "inf"
    )

    print(f"\nTotal excess units: {excess_units}")
    print(f"Total needed units: {needed_units}")
    print(f"Excess to needed ratio: {excess_to_needed_ratio:.2f}")

    return analyzer, analysis_df, excess_df, needed_df


def run_rule_based_optimization(analyzer, excess_df, needed_df, args):
    """Run rule-based optimization."""
    print("\n=== RULE-BASED OPTIMIZATION ===")

    # Create optimizer
    optimizer = RuleBasedOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(excess_df, needed_df)

    execution_time = time.time() - start_time
    print(f"Rule-based optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "rule_based_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(
            os.path.join(args.results_dir, "rule_based_impact.csv")
        )

        return transfer_plan, impact_df

    return transfer_plan, None


def run_ga_optimization(analyzer, excess_df, needed_df, args):
    """Run genetic algorithm optimization."""
    print("\n=== GENETIC ALGORITHM OPTIMIZATION ===")

    # Create optimizer
    optimizer = GeneticAlgorithmOptimizer(random_seed=args.seed)

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(
        excess_df,
        needed_df,
        population_size=args.ga_population,
        num_generations=args.ga_generations,
        crossover_prob=args.ga_crossover,
        mutation_prob=args.ga_mutation,
    )

    execution_time = time.time() - start_time
    print(f"Genetic algorithm optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "ga_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(os.path.join(args.results_dir, "ga_impact.csv"))

        return transfer_plan, impact_df

    return transfer_plan, None


def run_simplex_optimization(analyzer, excess_df, needed_df, args):
    """Run transportation LP + primal simplex optimization."""
    print("\n=== PRIMAL SIMPLEX OPTIMIZATION ===")

    logger_system = get_optimization_logger()
    start_time = time.time()

    parameters = {
        "algorithm": "Primal Simplex",
        "max_iterations": args.simplex_max_iterations,
        "tolerance": args.simplex_tolerance,
        "pivot_rule": args.simplex_pivot_rule,
        "use_bland": args.simplex_use_bland,
        "keep_history": args.simplex_keep_history,
        "dummy_excess_cost": args.simplex_dummy_excess_cost,
        "dummy_shortage_cost": args.simplex_dummy_shortage_cost,
        "excess_items": len(excess_df) if excess_df is not None else 0,
        "needed_items": len(needed_df) if needed_df is not None else 0,
    }
    logger_system.log_execution_start("simplex_optimization", parameters)

    if excess_df is None or needed_df is None or excess_df.empty or needed_df.empty:
        message = "No excess or needed inventory found. No simplex optimization required."
        print(message)
        logger_system.log_progress("simplex_optimization", message)
        logger_system.log_execution_end(
            "simplex_optimization",
            time.time() - start_time,
            {
                "instances_created": 0,
                "products_solved": 0,
                "transfers_generated": 0,
            },
        )
        return pd.DataFrame(), None

    # Load matrices with integer store IDs for consistent lookups.
    distance_matrix = pd.read_csv(
        os.path.join(args.data_dir, "distance_matrix.csv"), index_col=0
    )
    transport_cost_matrix = pd.read_csv(
        os.path.join(args.data_dir, "transport_cost_matrix.csv"), index_col=0
    )
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)
    transport_cost_matrix.index = transport_cost_matrix.index.astype(int)
    transport_cost_matrix.columns = transport_cost_matrix.columns.astype(int)

    instances = build_transportation_instances(
        excess_inventory=excess_df,
        needed_inventory=needed_df,
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
        allow_self_transfer=False,
        dummy_shortage_cost=args.simplex_dummy_shortage_cost,
        dummy_excess_cost=args.simplex_dummy_excess_cost,
    )

    logger_system.log_progress(
        "simplex_optimization",
        f"Transportation instances prepared: {len(instances)}",
    )

    if not instances:
        logger_system.log_execution_end(
            "simplex_optimization",
            time.time() - start_time,
            {
                "instances_created": 0,
                "products_solved": 0,
                "transfers_generated": 0,
            },
        )
        return pd.DataFrame(), None

    solver = PrimalSimplexSolver(
        max_iterations=args.simplex_max_iterations,
        tolerance=args.simplex_tolerance,
        pivot_rule=args.simplex_pivot_rule,
        use_bland=args.simplex_use_bland,
        keep_history=args.simplex_keep_history,
    )

    per_product_plans = []
    product_summaries = []

    for instance in instances:
        try:
            lp = transportation_instance_to_standard_form(instance)
            result = solver.solve(lp)
        except Exception as exc:
            logger_system.log_progress(
                "simplex_optimization",
                f"Product {instance.product_id}: failed to build/solve LP ({exc})",
            )
            product_summaries.append(
                {
                    "product_id": instance.product_id,
                    "status": "ERROR",
                    "objective_value": None,
                    "iterations": 0,
                    "dummy_added": instance.dummy_added,
                    "dummy_flow_count": 0,
                    "error": str(exc),
                }
            )
            continue

        product_plan = pd.DataFrame()
        if result.status == "OPTIMAL":
            product_plan = decode_simplex_solution_to_transfer_plan(
                simplex_result=result,
                standard_form_lp=lp,
                distance_matrix=distance_matrix,
                transport_cost_matrix=transport_cost_matrix,
            )
            if not product_plan.empty:
                per_product_plans.append(product_plan)

        product_summary = {
            "product_id": instance.product_id,
            "status": result.status,
            "objective_value": result.objective_value,
            "iterations": result.iterations,
            "dummy_added": instance.dummy_added,
            "dummy_flow_count": result.diagnostics.get("dummy_flow_count", 0),
        }
        product_summaries.append(product_summary)

        logger_system.log_progress(
            "simplex_optimization",
            (
                f"Product {instance.product_id}: status={result.status}, "
                f"objective={result.objective_value}, iterations={result.iterations}, "
                f"dummy_flows={product_summary['dummy_flow_count']}"
            ),
        )

    transfer_plan = merge_transfer_plans(per_product_plans)

    # Save per-product diagnostics regardless of transfer existence.
    diagnostics_path = os.path.join(args.results_dir, "simplex_product_status.csv")
    pd.DataFrame(product_summaries).to_csv(diagnostics_path, index=False)

    impact_df = None
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "simplex_transfers.csv"), index=False
        )

        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)
        pd.DataFrame(impact_df).to_csv(
            os.path.join(args.results_dir, "simplex_impact.csv")
        )

    execution_time = time.time() - start_time
    logger_system.log_execution_end(
        "simplex_optimization",
        execution_time,
        {
            "instances_created": len(instances),
            "products_solved": len(product_summaries),
            "optimal_products": sum(1 for row in product_summaries if row["status"] == "OPTIMAL"),
            "transfers_generated": len(transfer_plan),
            "total_units": transfer_plan["units"].sum()
            if not transfer_plan.empty
            else 0,
            "total_transport_cost": transfer_plan["transport_cost"].sum()
            if not transfer_plan.empty
            else 0,
        },
    )

    print(f"Primal simplex optimization completed in {execution_time:.2f} seconds")
    if transfer_plan.empty:
        print("No simplex transfers were generated.")
    else:
        print(f"Simplex transfers generated: {len(transfer_plan)}")

    return transfer_plan, impact_df


def run_modi_optimization(analyzer, excess_df, needed_df, args):
    """Run MODI optimizer and return (transfer_plan, impact_df)."""
    from src.config import (
        SIMPLEX_DUMMY_EXCESS_COST,
        SIMPLEX_DUMMY_SHORTAGE_COST,
        SIMPLEX_MAX_ITERATIONS,
        SIMPLEX_TOLERANCE,
    )

    distance_matrix = pd.read_csv(
        Path(args.data_dir) / "distance_matrix.csv", index_col=0
    )
    transport_cost_matrix = pd.read_csv(
        Path(args.data_dir) / "transport_cost_matrix.csv", index_col=0
    )
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)
    transport_cost_matrix.index = transport_cost_matrix.index.astype(int)
    transport_cost_matrix.columns = transport_cost_matrix.columns.astype(int)

    optimizer = MODIOptimizer(
        distance_matrix=distance_matrix,
        transport_cost_matrix=transport_cost_matrix,
        max_iterations=getattr(args, "simplex_max_iterations", SIMPLEX_MAX_ITERATIONS),
        tolerance=getattr(args, "simplex_tolerance", SIMPLEX_TOLERANCE),
        dummy_shortage_cost=getattr(args, "simplex_dummy_shortage_cost", SIMPLEX_DUMMY_SHORTAGE_COST),
        dummy_excess_cost=getattr(args, "simplex_dummy_excess_cost", SIMPLEX_DUMMY_EXCESS_COST),
    )

    transfer_plan = optimizer.optimize(excess_df, needed_df)

    impact_df = None
    if transfer_plan is not None and not transfer_plan.empty:
        try:
            impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)
        except Exception:
            impact_df = None

    return transfer_plan, impact_df


def create_results(analysis_df, results_dict, analyzer, args):
    """Create simplified results: summary and best transfer plan."""
    print("\n=== GENERATING RESULTS ===")

    # Load store and product data
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))

    # Create results manager and generate final results
    results_manager = ResultsManager(args.results_dir)
    results_manager.create_final_results(results_dict, stores_df, products_df)


def main():
    parser = argparse.ArgumentParser(
        description="Inventory Transfer Optimization System"
    )

    # General options
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR, help="Results directory"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=VISUALIZATIONS_DIR,
        help="Visualizations directory",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")

    # Data generation options
    parser.add_argument("--generate-data", action="store_true", help="Generate data")
    parser.add_argument(
        "--products", type=int, default=NUM_PRODUCTS, help="Number of products"
    )
    parser.add_argument(
        "--days", type=int, default=SALES_DAYS, help="Number of days of sales data"
    )
    parser.add_argument(
        "--excess-percent",
        type=int,
        default=EXCESS_PERCENT,
        help="Percentage of items with excess inventory",
    )
    parser.add_argument(
        "--shortage-percent",
        type=int,
        default=SHORTAGE_PERCENT,
        help="Percentage of items with shortage",
    )

    # Analysis options
    parser.add_argument(
        "--min-days",
        type=int,
        default=MIN_INVENTORY_DAYS,
        help="Minimum days of inventory",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=MAX_INVENTORY_DAYS,
        help="Maximum days of inventory",
    )

    # Optimization options
    parser.add_argument(
        "--rule-based", action="store_true", help="Run rule-based optimization"
    )
    parser.add_argument(
        "--ga", action="store_true", help="Run genetic algorithm optimization"
    )
    parser.add_argument("--simplex", action="store_true", help="Run primal simplex optimization")
    parser.add_argument(
        "--all", action="store_true", help="Run all optimization methods"
    )

    # Genetic algorithm options
    parser.add_argument(
        "--ga-population",
        type=int,
        default=GA_POPULATION_SIZE,
        help="GA population size",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=GA_GENERATIONS,
        help="GA number of generations",
    )
    parser.add_argument(
        "--ga-crossover",
        type=float,
        default=GA_CROSSOVER_PROB,
        help="GA crossover probability",
    )
    parser.add_argument(
        "--ga-mutation",
        type=float,
        default=GA_MUTATION_PROB,
        help="GA mutation probability",
    )

    # Primal simplex options
    parser.add_argument(
        "--simplex-max-iterations",
        type=int,
        default=SIMPLEX_MAX_ITERATIONS,
        help="Maximum simplex iterations per product LP",
    )
    parser.add_argument(
        "--simplex-tolerance",
        type=float,
        default=SIMPLEX_TOLERANCE,
        help="Numerical tolerance for simplex operations",
    )
    parser.add_argument(
        "--simplex-pivot-rule",
        type=str,
        default=SIMPLEX_PIVOT_RULE,
        choices=["dantzig", "bland"],
        help="Pivot entering-variable rule",
    )
    parser.add_argument(
        "--simplex-use-bland",
        action="store_true",
        default=SIMPLEX_USE_BLAND,
        help="Enable Bland tie-break anti-cycling",
    )
    parser.add_argument(
        "--simplex-no-bland",
        action="store_false",
        dest="simplex_use_bland",
        help="Disable Bland anti-cycling fallback",
    )
    parser.add_argument(
        "--simplex-keep-history",
        action="store_true",
        help="Keep simplex tableau snapshots for diagnostics",
    )
    parser.add_argument(
        "--simplex-dummy-excess-cost",
        type=float,
        default=SIMPLEX_DUMMY_EXCESS_COST,
        help="Dummy destination cost when supply > demand",
    )
    parser.add_argument(
        "--simplex-dummy-shortage-cost",
        type=float,
        default=SIMPLEX_DUMMY_SHORTAGE_COST,
        help="Dummy source penalty when demand > supply (default: inferred)",
    )

    # Display options
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only summary results"
    )

    args = parser.parse_args()

    # Create directories
    directories = setup_directories()
    args.data_dir = str(directories["data"])
    args.vis_dir = str(directories["visualizations"])
    args.results_dir = str(directories["results"])

    # Generate data if needed
    if args.generate_data:
        run_data_generation(args)

    # Check if data exists
    for file in REQUIRED_DATA_FILES:
        file_path = Path(args.data_dir) / file
        if not file_path.exists():
            print(
                f"Required file {file} not found. Please run with --generate-data first."
            )
            return

    # Run analysis
    analyzer, analysis_df, excess_df, needed_df = run_analysis(args)

    # Run optimizations
    results_dict = {}

    if args.rule_based or args.all:
        transfer_plan, impact_df = run_rule_based_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Rule-Based"] = (transfer_plan, impact_df)

    if args.ga or args.all:
        transfer_plan, impact_df = run_ga_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Genetic Algorithm"] = (transfer_plan, impact_df)

    # --all now includes all 3 solvers for direct baseline comparison.
    if args.simplex or args.all:
        transfer_plan, impact_df = run_simplex_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Primal Simplex"] = (transfer_plan, impact_df)

    # Create comprehensive results and reports
    if results_dict:
        create_results(analysis_df, results_dict, analyzer, args)

    print("\n=== INVENTORY TRANSFER OPTIMIZATION COMPLETE ===")
    print(f"Results saved to {args.results_dir} directory:")
    print(f"  • result_summary.txt - Algorithm comparison and recommendations")
    print(f"  • best_transfer_plan.csv - Optimized transfer plan")


if __name__ == "__main__":
    main()
