"""
Coordinates the generation of all data for the inventory optimization system.
Ensures meaningful inventory imbalances are created for optimization.
"""

import argparse
import os

import pandas as pd

from src.data_generator.distance_calculator import DistanceCalculator
from src.data_generator.inventory_generator import InventoryGenerator
from src.data_generator.product_generator import ProductGenerator
from src.data_generator.sales_generator import SalesGenerator
from src.data_generator.store_generator import StoreGenerator


def generate_all_data(
    num_products=100,
    days=365,
    output_dir="data",
    random_seed=2025,
    min_days=7,
    max_days=30,
):
    """
    Generate all required data for the inventory optimization system.

    Args:
        num_products: Number of products to generate
        days: Number of days of sales history to generate
        output_dir: Directory to save the generated data
        random_seed: Random seed for reproducibility
        min_days: Minimum days of inventory threshold (below this is shortage)
        max_days: Maximum days of inventory threshold (above this is excess)
    """
    print(f"Generating all data with seed {random_seed}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    stores_path = os.path.join(output_dir, "stores.csv")
    products_path = os.path.join(output_dir, "products.csv")
    sales_path = os.path.join(output_dir, "sales_data.csv")
    inventory_path = os.path.join(output_dir, "inventory_data.csv")
    distance_path = os.path.join(output_dir, "distance_matrix.csv")
    cost_path = os.path.join(output_dir, "transport_cost_matrix.csv")

    # Generate store data
    print("\n1. Generating store data...")
    store_gen = StoreGenerator(random_seed=random_seed)
    stores = store_gen.generate_stores(stores_path)

    # Generate product data
    print("\n2. Generating product data...")
    product_gen = ProductGenerator(random_seed=random_seed)
    products = product_gen.generate_products(num_products, products_path)

    # Generate sales data
    print("\n3. Generating sales data...")
    sales_gen = SalesGenerator(stores, products, random_seed=random_seed)
    sales_df = sales_gen.generate_sales_data(days, sales_path)

    # Generate inventory data with meaningful imbalances
    print("\n4. Generating inventory data with meaningful imbalances...")
    inventory_gen = InventoryGenerator(sales_df, random_seed=random_seed)
    inventory_df = inventory_gen.generate_inventory_data(
        inventory_path, min_days=min_days, max_days=max_days
    )

    # Generate distance and transport cost matrices
    print("\n5. Generating distance and cost matrices...")
    distance_calc = DistanceCalculator(stores)
    distance_matrix = distance_calc.generate_distance_matrix(distance_path)
    cost_matrix = distance_calc.generate_transport_cost_matrix(
        distance_matrix, cost_path
    )

    print("\nData generation complete!")
    print(f"All files saved to directory: {output_dir}")

    # Print summary statistics
    print("\nData Summary:")
    print(f"- Stores: {len(stores)}")
    print(f"- Products: {len(products)}")
    print(f"- Sales records: {len(sales_df)}")
    print(f"- Store-product combinations: {len(inventory_df)}")

    # Calculate inventory status
    avg_sales = (
        sales_df.groupby(["store_id", "product_id"])["quantity"].mean().reset_index()
    )
    avg_sales.rename(columns={"quantity": "avg_daily_sales"}, inplace=True)

    # Merge with inventory
    analysis_df = pd.merge(
        inventory_df, avg_sales, on=["store_id", "product_id"], how="left"
    )

    # Fill NaN values
    analysis_df["avg_daily_sales"].fillna(0.01, inplace=True)

    # Calculate days of inventory
    analysis_df["days_of_inventory"] = (
        analysis_df["current_stock"] / analysis_df["avg_daily_sales"]
    )

    # Count items with excess inventory (more than max_days)
    excess_count = len(analysis_df[analysis_df["days_of_inventory"] > max_days])

    # Count items with shortage (less than min_days)
    shortage_count = len(analysis_df[analysis_df["days_of_inventory"] < min_days])

    # Count balanced items
    balanced_count = len(analysis_df) - excess_count - shortage_count

    print(f"- Inventory status:")
    print(
        f"  * Excess items (>{max_days} days): {excess_count} ({excess_count/len(analysis_df)*100:.1f}%)"
    )
    print(
        f"  * Shortage items (<{min_days} days): {shortage_count} ({shortage_count/len(analysis_df)*100:.1f}%)"
    )
    print(
        f"  * Balanced items: {balanced_count} ({balanced_count/len(analysis_df)*100:.1f}%)"
    )

    # Check if we have products with both excess and shortage
    # This is critical for meaningful transfer optimization
    product_status = []
    for product_id in analysis_df["product_id"].unique():
        product_df = analysis_df[analysis_df["product_id"] == product_id]
        has_excess = len(product_df[product_df["days_of_inventory"] > max_days]) > 0
        has_shortage = len(product_df[product_df["days_of_inventory"] < min_days]) > 0

        if has_excess and has_shortage:
            product_status.append(
                {"product_id": product_id, "has_both_excess_and_shortage": True}
            )

    transferable_products = len(product_status)
    print(
        f"- Products with both excess and shortage across stores: {transferable_products}"
    )
    print(
        f"  ({transferable_products/len(analysis_df['product_id'].unique())*100:.1f}% of total products)"
    )

    # Add a check to ensure we have meaningful inventory imbalances
    if transferable_products < 10:
        print(
            "\nWARNING: Few products have both excess and shortage. Optimization results may be limited."
        )
        print(
            "Consider regenerating data with different random seed for more meaningful imbalances."
        )
    else:
        print("\nData generation successful with meaningful inventory imbalances.")
        print("Ready for optimization!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for inventory optimization system"
    )
    parser.add_argument(
        "--products", type=int, default=100, help="Number of products to generate"
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Number of days of sales history"
    )
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min-days", type=int, default=7, help="Minimum days of inventory threshold"
    )
    parser.add_argument(
        "--max-days", type=int, default=30, help="Maximum days of inventory threshold"
    )

    args = parser.parse_args()

    generate_all_data(
        num_products=args.products,
        days=args.days,
        output_dir=args.output,
        random_seed=args.seed,
        min_days=args.min_days,
        max_days=args.max_days,
    )
