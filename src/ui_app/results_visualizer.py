import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsVisualizer:
    def __init__(self, data_dir="data", results_dir="results"):
        """
        Initialize the results visualizer.

        Args:
            data_dir: Directory containing original data files
            results_dir: Directory containing optimization results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir

    def load_data(self):
        """
        Load all relevant data files for visualization.

        Returns:
            Dict containing loaded DataFrames
        """
        try:
            # Load original data
            stores_df = pd.read_csv(f"{self.data_dir}/stores.csv")
            products_df = pd.read_csv(f"{self.data_dir}/products.csv")

            # Load transfer plans and impact data for each algorithm
            transfer_plans = {}
            impact_data = {}
            algorithms = ["rule_based", "lp", "ga"]

            for algo in algorithms:
                # Load transfer plans
                try:
                    transfer_plan_file = f'{self.results_dir}/{algo.lower().replace(" ", "_")}_transfers.csv'
                    # print(transfer_plan_file)
                    transfer_plans[algo] = pd.read_csv(transfer_plan_file)
                    # print(transfer_plans[algo].head())
                except FileNotFoundError:
                    transfer_plans[algo] = pd.DataFrame()

                # Load impact data
                try:
                    impact_file = f'{self.results_dir}/{algo.lower().replace(" ", "_")}_impact.csv'
                    impact_data[algo] = pd.read_csv(impact_file)
                except FileNotFoundError:
                    impact_data[algo] = pd.DataFrame()

            # Load inventory data before and after transfers
            analysis_df = pd.read_csv(f"{self.results_dir}/inventory_analysis.csv")

            return {
                "stores": stores_df,
                "products": products_df,
                "transfer_plans": transfer_plans,
                "impact_data": impact_data,
                "analysis_df": analysis_df,
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def create_store_inventory_histogram(
        self,
        analysis_df,
        status_column="inventory_status",
        title="Inventory Status Distribution",
    ):
        """
        Create a histogram of inventory status by store.

        Args:
            analysis_df: DataFrame with inventory analysis
            status_column: Column containing inventory status
            title: Title for the plot

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        status_counts = analysis_df.groupby(status_column).size()

        plt.bar(status_counts.index, status_counts.values)
        plt.title(title)
        plt.xlabel("Inventory Status")
        plt.ylabel("Number of Items")

        # Add percentage labels
        total = len(analysis_df)
        for i, v in enumerate(status_counts):
            plt.text(i, v, f"{v} ({v/total*100:.1f}%)", ha="center", va="bottom")

        plt.tight_layout()
        return plt.gcf()

    def create_city_inventory_histogram(self, analysis_df):
        """
        Create a stacked bar chart of inventory status by city.

        Args:
            analysis_df: DataFrame with inventory analysis

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=(12, 6))

        # Group by city and inventory status
        city_status = (
            analysis_df.groupby(["city", "inventory_status"])
            .size()
            .unstack(fill_value=0)
        )

        # Plot stacked bar chart
        city_status.plot(kind="bar", stacked=True)
        plt.title("Inventory Status Distribution by City")
        plt.xlabel("City")
        plt.ylabel("Number of Items")
        plt.legend(title="Inventory Status", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return plt.gcf()

    def create_product_category_histogram(self, analysis_df):
        """
        Create a stacked bar chart of inventory status by product category.

        Args:
            analysis_df: DataFrame with inventory analysis

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=(12, 6))

        # Group by category and inventory status
        category_status = (
            analysis_df.groupby(["category", "inventory_status"])
            .size()
            .unstack(fill_value=0)
        )

        # Plot stacked bar chart
        category_status.plot(kind="bar", stacked=True)
        plt.title("Inventory Status Distribution by Product Category")
        plt.xlabel("Product Category")
        plt.ylabel("Number of Items")
        plt.legend(title="Inventory Status", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return plt.gcf()

    def create_algorithm_comparison_chart(self, transfer_plans, algorithms):
        """
        Create a bar chart comparing transfer volumes for different algorithms.

        Args:
            transfer_plans: Dictionary of transfer plans
            algorithms: List of algorithm names

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=(10, 6))

        # Prepare data
        transfer_volumes = []
        for algo in algorithms:
            plan = transfer_plans.get(algo, pd.DataFrame())
            transfer_volumes.append(plan["units"].sum() if not plan.empty else 0)

        plt.bar(algorithms, transfer_volumes)
        plt.title("Total Units Transferred by Algorithm")
        plt.xlabel("Algorithm")
        plt.ylabel("Total Units Transferred")

        # Add value labels
        for i, v in enumerate(transfer_volumes):
            plt.text(i, v, f"{v:,}", ha="center", va="bottom")

        plt.tight_layout()
        return plt.gcf()

    def create_transfer_cost_comparison(self, transfer_plans, algorithms):
        """
        Create a bar chart comparing transfer costs for different algorithms.

        Args:
            transfer_plans: Dictionary of transfer plans
            algorithms: List of algorithm names

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=(10, 6))

        # Prepare data
        transfer_costs = []
        for algo in algorithms:
            plan = transfer_plans.get(algo, pd.DataFrame())
            transfer_costs.append(plan["transport_cost"].sum() if not plan.empty else 0)

        plt.bar(algorithms, transfer_costs)
        plt.title("Total Transport Cost by Algorithm")
        plt.xlabel("Algorithm")
        plt.ylabel("Total Transport Cost (VND)")

        # Add value labels
        for i, v in enumerate(transfer_costs):
            plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

        plt.tight_layout()
        return plt.gcf()
