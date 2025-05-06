"""
Main Application Module
-----------------------
Main entry point for the inventory transfer optimization system.
Coordinates data generation, analysis, optimization, and visualization.
Provides clear output of results for easy comparison.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

# Import components
from src.data_generator.data_generator_main import generate_all_data
from src.engine.analyzer import InventoryAnalyzer
from src.engine.genetic_algorithm import GeneticAlgorithmOptimizer
from src.engine.linear_programming import LinearProgrammingOptimizer
from src.engine.results_presenter import ResultsPresenter
from src.engine.rule_based import RuleBasedOptimizer
from src.engine.visualizer import InventoryVisualizer


def setup_directories():
    """Create necessary directories."""
    directories = ["data", "visualizations", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories


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

    print(f"\nTotal excess units: {excess_units}")
    print(f"Total needed units: {needed_units}")
    print(f"Excess to needed ratio: {excess_units / needed_units:.2f}")

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


def run_lp_optimization(analyzer, excess_df, needed_df, args):
    """Run linear programming optimization."""
    print("\n=== LINEAR PROGRAMMING OPTIMIZATION ===")

    # Create optimizer
    optimizer = LinearProgrammingOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(
        excess_df, needed_df, time_limit=args.lp_time_limit
    )

    execution_time = time.time() - start_time
    print(f"Linear programming optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        print(f"\n SUCESSFULLY SAVED TRANSFER PLAN !")
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "lp_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(os.path.join(args.results_dir, "lp_impact.csv"))

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


def run_visualizations(analysis_df, results_dict, analyzer, args):
    """Create visualizations."""
    print("\n=== CREATING VISUALIZATIONS ===")

    # Load store data
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))

    # Create store name mapping dictionary
    store_name_map = stores_df.set_index("store_id")["store_name"].to_dict()

    # Load product data
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))

    # Create product name mapping dictionary
    product_name_map = products_df.set_index("product_id")["product_name"].to_dict()

    # Create visualizer
    visualizer = InventoryVisualizer(stores_df, args.vis_dir)

    # Visualize inventory status
    visualizer.visualize_inventory_status(analysis_df)

    # Visualize transfer plans
    for method, (transfer_plan, _) in results_dict.items():
        if transfer_plan is not None and not transfer_plan.empty:
            print(f"Visualizing {method} transfer plan...")

            # Replace IDs with names in the transfer plan for better visualization
            transfer_plan_with_names = transfer_plan.copy()
            if "from_store_id" in transfer_plan_with_names.columns:
                transfer_plan_with_names["from_store"] = transfer_plan_with_names[
                    "from_store_id"
                ].map(store_name_map)

            if "to_store_id" in transfer_plan_with_names.columns:
                transfer_plan_with_names["to_store"] = transfer_plan_with_names[
                    "to_store_id"
                ].map(store_name_map)

            if "product_id" in transfer_plan_with_names.columns:
                transfer_plan_with_names["product"] = transfer_plan_with_names[
                    "product_id"
                ].map(product_name_map)

            visualizer.visualize_transfer_plan(transfer_plan_with_names)

    # Visualize impact for each method
    for method, (transfer_plan, impact_df) in results_dict.items():
        if (
            impact_df is not None
            and transfer_plan is not None
            and not transfer_plan.empty
        ):
            print(f"Visualizing {method} impact...")

            # Get post-analysis data from the analyzer's evaluate_plan_impact method
            _, post_analysis = analyzer.evaluate_plan_impact(transfer_plan)

            visualizer.visualize_impact(
                impact_df,
                analysis_df=analysis_df,
                post_analysis=post_analysis,
                store_names=store_name_map,
                product_names=product_name_map,
                algorithm_name=method,
                save_png=True,
                show_plot=False,
            )

    # Create results presenter
    presenter = ResultsPresenter(args.results_dir)

    # Load transfer plans
    transfer_plans = {
        method: results[0]
        for method, results in results_dict.items()
        if results[0] is not None and not results[0].empty
    }

    # Load impact data
    impact_data = {
        method: results[1]
        for method, results in results_dict.items()
        if results[1] is not None
    }

    # Generate comparison report
    if transfer_plans and impact_data:
        presenter.visualize_comparison(transfer_plans, impact_data)
        presenter.generate_report(transfer_plans, impact_data)
        presenter.print_summary_tables()


def main():
    parser = argparse.ArgumentParser(
        description="Inventory Transfer Optimization System"
    )

    # General options
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--vis-dir", type=str, default="visualizations", help="Visualizations directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data generation options
    parser.add_argument("--generate-data", action="store_true", help="Generate data")
    parser.add_argument("--products", type=int, default=100, help="Number of products")
    parser.add_argument(
        "--days", type=int, default=365, help="Number of days of sales data"
    )

    # Analysis options
    parser.add_argument(
        "--min-days", type=int, default=7, help="Minimum days of inventory"
    )
    parser.add_argument(
        "--max-days", type=int, default=30, help="Maximum days of inventory"
    )

    # Optimization options
    parser.add_argument(
        "--rule-based", action="store_true", help="Run rule-based optimization"
    )
    parser.add_argument(
        "--lp", action="store_true", help="Run linear programming optimization"
    )
    parser.add_argument(
        "--ga", action="store_true", help="Run genetic algorithm optimization"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all optimization methods"
    )

    # Linear programming options
    parser.add_argument(
        "--lp-time-limit", type=int, default=300, help="LP solver time limit in seconds"
    )

    # Genetic algorithm options
    parser.add_argument(
        "--ga-population", type=int, default=100, help="GA population size"
    )
    parser.add_argument(
        "--ga-generations", type=int, default=50, help="GA number of generations"
    )
    parser.add_argument(
        "--ga-crossover", type=float, default=0.7, help="GA crossover probability"
    )
    parser.add_argument(
        "--ga-mutation", type=float, default=0.2, help="GA mutation probability"
    )

    # Display options
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only summary results"
    )

    args = parser.parse_args()

    # Create directories
    directories = setup_directories()
    args.data_dir = directories[0]  # data
    args.vis_dir = directories[1]  # visualizations
    args.results_dir = directories[2]  # results

    # Generate data if needed
    if args.generate_data:
        run_data_generation(args)

    # Check if data exists
    required_files = [
        "sales_data.csv",
        "inventory_data.csv",
        "stores.csv",
        "products.csv",
        "distance_matrix.csv",
        "transport_cost_matrix.csv",
    ]

    for file in required_files:
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

    if args.lp or args.all:
        transfer_plan, impact_df = run_lp_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Linear Programming"] = (transfer_plan, impact_df)

    if args.ga or args.all:
        transfer_plan, impact_df = run_ga_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Genetic Algorithm"] = (transfer_plan, impact_df)

    # Create visualizations and reports
    if results_dict:
        run_visualizations(analysis_df, results_dict, analyzer, args)

    # Display summary results
    if not args.summary_only and results_dict:
        # Create results presenter to show summary tables
        presenter = ResultsPresenter(args.results_dir)
        presenter.print_summary_tables()

    print("\n=== INVENTORY TRANSFER OPTIMIZATION COMPLETE ===")
    print(f"Results saved to {args.results_dir} directory")
    print(f"Visualizations saved to {args.vis_dir} directory")


if __name__ == "__main__":
    main()
