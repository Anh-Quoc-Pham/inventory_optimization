"""
Verify Inventory Imbalances
--------------------------
This script analyzes generated inventory data to verify that meaningful imbalances exist.
It checks for products that have both excess and shortage across different stores.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(data_dir="data"):
    """Load necessary data files."""
    sales_path = os.path.join(data_dir, "sales_data.csv")
    inventory_path = os.path.join(data_dir, "inventory_data.csv")
    stores_path = os.path.join(data_dir, "stores.csv")

    # Check if files exist
    for path in [sales_path, inventory_path, stores_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file {path} not found.")

    # Load data
    sales_df = pd.read_csv(sales_path, parse_dates=["date"])
    inventory_df = pd.read_csv(inventory_path)
    stores_df = pd.read_csv(stores_path)

    return sales_df, inventory_df, stores_df


def analyze_imbalances(sales_df, inventory_df, min_days=7, max_days=30):
    """Analyze inventory imbalances to verify meaningful transfers are possible."""
    # Calculate average daily sales
    days_in_data = (sales_df["date"].max() - sales_df["date"].min()).days + 1

    avg_sales = (
        sales_df.groupby(["store_id", "product_id"])["quantity"].sum().reset_index()
    )
    avg_sales["avg_daily_sales"] = avg_sales["quantity"] / days_in_data
    avg_sales.drop("quantity", axis=1, inplace=True)

    # Merge with inventory
    analysis_df = pd.merge(
        inventory_df, avg_sales, on=["store_id", "product_id"], how="left"
    )

    # Fill NaN values for products with no sales
    analysis_df["avg_daily_sales"].fillna(0.01, inplace=True)

    # Calculate days of inventory
    analysis_df["days_of_inventory"] = (
        analysis_df["current_stock"] / analysis_df["avg_daily_sales"]
    )

    # Classify inventory status
    analysis_df["status"] = "Balanced"
    analysis_df.loc[analysis_df["days_of_inventory"] < min_days, "status"] = "Shortage"
    analysis_df.loc[analysis_df["days_of_inventory"] > max_days, "status"] = "Excess"

    # Count items in each status
    status_counts = analysis_df["status"].value_counts()

    print("Inventory Status Distribution:")
    for status, count in status_counts.items():
        percentage = count / len(analysis_df) * 100
        print(f"- {status}: {count} items ({percentage:.1f}%)")

    # Analyze products with both excess and shortage
    product_stats = []

    for product_id in analysis_df["product_id"].unique():
        product_df = analysis_df[analysis_df["product_id"] == product_id]
        excess_count = len(product_df[product_df["status"] == "Excess"])
        shortage_count = len(product_df[product_df["status"] == "Shortage"])
        balanced_count = len(product_df[product_df["status"] == "Balanced"])

        product_stats.append(
            {
                "product_id": product_id,
                "excess_count": excess_count,
                "shortage_count": shortage_count,
                "balanced_count": balanced_count,
                "total_stores": len(product_df),
                "has_both": excess_count > 0 and shortage_count > 0,
            }
        )

    # Convert to DataFrame for analysis
    product_stats_df = pd.DataFrame(product_stats)

    # Count products with both excess and shortage
    transferable_count = product_stats_df["has_both"].sum()
    transfer_percentage = transferable_count / len(product_stats_df) * 100

    print(
        f"\nProducts with both excess and shortage: {transferable_count} out of {len(product_stats_df)}"
    )
    print(f"Percentage of transferable products: {transfer_percentage:.1f}%")

    # Get examples of products with significant imbalances
    significant_imbalances = (
        product_stats_df[
            (product_stats_df["excess_count"] >= 2)
            & (product_stats_df["shortage_count"] >= 2)
        ]
        .sort_values(by=["excess_count", "shortage_count"], ascending=False)
        .head(10)
    )

    print(
        "\nTop products with significant imbalances (at least 2 stores with excess and 2 with shortage):"
    )
    for _, row in significant_imbalances.iterrows():
        print(
            f"- Product {row['product_id']}: {row['excess_count']} stores with excess, "
            + f"{row['shortage_count']} stores with shortage, {row['balanced_count']} balanced"
        )

    return analysis_df, product_stats_df


def visualize_imbalances(
    analysis_df, product_stats_df, stores_df, output_dir="visualizations"
):
    """Create visualizations of inventory imbalances."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    # 1. Create a bar chart of inventory status distribution
    plt.figure(figsize=(10, 6))
    status_counts = analysis_df["status"].value_counts()

    status_colors = {
        "Excess": "#FF8042",  # Orange
        "Shortage": "#0088FE",  # Blue
        "Balanced": "#00C49F",  # Green
    }

    bars = plt.bar(
        status_counts.index,
        status_counts.values,
        color=[status_colors.get(s, "#999999") for s in status_counts.index],
    )

    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{height} ({height/len(analysis_df)*100:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Number of Items")
    plt.title("Inventory Status Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inventory_status_distribution.png"), dpi=300)
    plt.close()

    # 2. Create a pie chart of transferable vs non-transferable products
    plt.figure(figsize=(8, 8))
    transferable = product_stats_df["has_both"].sum()
    non_transferable = len(product_stats_df) - transferable

    plt.pie(
        [transferable, non_transferable],
        labels=["Transferable", "Non-transferable"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90,
    )

    plt.axis("equal")
    plt.title("Products with Both Excess and Shortage (Transferable)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transferable_products.png"), dpi=300)
    plt.close()

    # 3. Select a product with significant imbalances for detailed analysis
    significant_products = product_stats_df[
        (product_stats_df["excess_count"] >= 2)
        & (product_stats_df["shortage_count"] >= 2)
    ].sort_values(by=["excess_count", "shortage_count"], ascending=False)

    if not significant_products.empty:
        example_product_id = significant_products.iloc[0]["product_id"]

        # Get data for this product across all stores
        product_data = analysis_df[
            analysis_df["product_id"] == example_product_id
        ].copy()

        # Add store names
        store_name_map = stores_df.set_index("store_id")["store_name"].to_dict()
        product_data["store_name"] = product_data["store_id"].map(store_name_map)

        # Sort by days of inventory
        product_data = product_data.sort_values("days_of_inventory")

        # Create a bar chart showing days of inventory by store
        plt.figure(figsize=(12, 8))

        # Create bars with colors based on status
        bars = plt.barh(
            product_data["store_name"],
            product_data["days_of_inventory"],
            color=[status_colors.get(s, "#999999") for s in product_data["status"]],
        )

        # Add threshold lines
        plt.axvline(
            x=7,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Shortage Threshold (7 days)",
        )
        plt.axvline(
            x=30,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Excess Threshold (30 days)",
        )

        plt.xlabel("Days of Inventory")
        plt.title(f"Inventory Status Across Stores for Product {example_product_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"product_{example_product_id}_imbalances.png"),
            dpi=300,
        )
        plt.close()

        print(f"\nDetailed visualization created for Product {example_product_id}")

    print(f"\nAll visualizations saved to {output_dir} directory")


def main():
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)

    try:
        # Load data
        sales_df, inventory_df, stores_df = load_data()

        # Analyze imbalances
        analysis_df, product_stats_df = analyze_imbalances(sales_df, inventory_df)

        # Visualize imbalances
        visualize_imbalances(analysis_df, product_stats_df, stores_df)

        # Provide a summary assessment
        transferable_pct = product_stats_df["has_both"].mean() * 100

        if transferable_pct >= 30:
            print(
                "\n✅ ASSESSMENT: Data has good inventory imbalances for meaningful optimization!"
            )
        elif transferable_pct >= 15:
            print(
                "\n⚠️ ASSESSMENT: Data has moderate inventory imbalances. Optimization should be possible."
            )
        else:
            print(
                "\n❌ ASSESSMENT: Data has few meaningful inventory imbalances. Consider regenerating data."
            )

        # Provide recommendations if needed
        if transferable_pct < 15:
            print("\nRECOMMENDATIONS:")
            print("1. Regenerate data with a different random seed")
            print(
                "2. Modify inventory_generator.py to increase the percentage of products with imbalances"
            )
            print(
                "3. Try running the inventory generator with different min_days and max_days parameters"
            )

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
