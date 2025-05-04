"""
Linear Programming Optimization Engine
-------------------------------------------
Implements a linear programming approach for inventory transfer optimization.
Formulates and solves the transfer problem as a transportation problem.
Fixes handling of store IDs for distance and cost lookups.
"""

import numpy as np
import pandas as pd
import pulp
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum


class LinearProgrammingOptimizer:
    def __init__(self, distance_matrix=None, transport_cost_matrix=None):
        """
        Initialize the linear programming optimization engine.

        Args:
            distance_matrix: Matrix of distances between stores
            transport_cost_matrix: Matrix of transport costs between stores
        """
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.transfer_plan = None
        self.model = None
        self.status = None

    def load_matrices(self, distance_path, cost_path):
        """
        Load distance and transport cost matrices from CSV files.

        Args:
            distance_path: Path to distance matrix CSV
            cost_path: Path to transport cost matrix CSV
        """
        print("Loading distance and transport cost matrices...")

        # Load matrices with store IDs as integers
        self.distance_matrix = pd.read_csv(distance_path, index_col=0)
        self.transport_cost_matrix = pd.read_csv(cost_path, index_col=0)

        # Ensure indices and columns are integers
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)
        self.transport_cost_matrix.index = self.transport_cost_matrix.index.astype(int)
        self.transport_cost_matrix.columns = self.transport_cost_matrix.columns.astype(
            int
        )

    def optimize(self, excess_inventory, needed_inventory, time_limit=300):
        """
        Generate a transfer plan using linear programming.

        Args:
            excess_inventory: DataFrame containing excess inventory
            needed_inventory: DataFrame containing needed inventory
            time_limit: Time limit for solver in seconds (default: 300s)

        Returns:
            DataFrame containing transfer recommendations
        """
        print("Generating linear programming-based transfer plan...")

        # If either of these is empty, there's nothing to optimize
        if excess_inventory.empty or needed_inventory.empty:
            print("No excess or needed inventory found. No transfers needed.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        # Create dictionaries for supply (excess) and demand (needed)
        supply = {}
        for _, row in excess_inventory.iterrows():
            supply[(row["store_id"], row["product_id"])] = row["excess_units"]

        demand = {}
        for _, row in needed_inventory.iterrows():
            demand[(row["store_id"], row["product_id"])] = row["needed_units"]

        # Create cost dictionary
        cost = {}
        for from_store, from_product in supply.keys():
            for to_store, to_product in demand.keys():
                # Can only transfer the same product
                if (
                    from_product == to_product and from_store != to_store
                ):  # Avoid self-transfers
                    # Get cost directly from the matrix using integer IDs
                    if (
                        from_store in self.transport_cost_matrix.index
                        and to_store in self.transport_cost_matrix.columns
                    ):
                        cost[(from_store, from_product, to_store, to_product)] = float(
                            self.transport_cost_matrix.loc[from_store, to_store]
                        )
                    else:
                        # If cost not available, use a high default cost
                        cost[(from_store, from_product, to_store, to_product)] = 1000000

        # Create optimization model
        model = LpProblem("Inventory_Transfer_Optimization", LpMinimize)
        self.model = model

        # Define decision variables
        transfer_vars = {}
        for from_store, from_product in supply.keys():
            for to_store, to_product in demand.keys():
                if (
                    from_product == to_product and from_store != to_store
                ):  # Same product, different stores
                    var_key = (from_store, from_product, to_store, to_product)
                    if var_key in cost:  # Only create variables for valid transfers
                        transfer_vars[var_key] = LpVariable(
                            f"transfer_{from_store}_{from_product}_{to_store}_{to_product}",
                            lowBound=0,
                            upBound=min(
                                supply[(from_store, from_product)],
                                demand[(to_store, to_product)],
                            ),
                            cat="Integer",
                        )

        # Define objective function (minimize total transport cost)
        if transfer_vars:
            model += lpSum(
                [cost[var_key] * transfer_vars[var_key] for var_key in transfer_vars]
            )
        else:
            print("No valid transfers possible.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        # Add supply constraints (can't transfer more than excess inventory)
        for from_store, from_product in supply.keys():
            relevant_vars = [
                transfer_vars[key]
                for key in transfer_vars
                if key[0] == from_store and key[1] == from_product
            ]

            if relevant_vars:  # Only add constraint if there are relevant variables
                model += (
                    lpSum(relevant_vars) <= supply[(from_store, from_product)],
                    f"Supply_{from_store}_{from_product}",
                )

        # Add demand constraints (can't transfer more than needed inventory)
        for to_store, to_product in demand.keys():
            relevant_vars = [
                transfer_vars[key]
                for key in transfer_vars
                if key[2] == to_store and key[3] == to_product
            ]

            if relevant_vars:  # Only add constraint if there are relevant variables
                model += (
                    lpSum(relevant_vars) <= demand[(to_store, to_product)],
                    f"Demand_{to_store}_{to_product}",
                )

        # Solve the model with time limit
        self.status = model.solve(PULP_CBC_CMD(msg=True, timeLimit=time_limit))

        # Extract the solution
        transfers = []

        for var_key in transfer_vars:
            from_store, from_product, to_store, to_product = var_key
            transfer_amount = transfer_vars[var_key].value()

            if transfer_amount and transfer_amount > 0:
                # Get distance from matrix using integer IDs
                if (
                    from_store in self.distance_matrix.index
                    and to_store in self.distance_matrix.columns
                ):
                    distance = float(self.distance_matrix.loc[from_store, to_store])
                else:
                    distance = 0

                # Calculate transport cost
                transport_cost = cost[var_key] * transfer_amount

                transfers.append(
                    {
                        "from_store_id": from_store,
                        "to_store_id": to_store,
                        "product_id": from_product,
                        "units": int(transfer_amount),
                        "distance_km": distance,
                        "transport_cost": transport_cost,
                    }
                )

        # Create DataFrame from transfers
        self.transfer_plan = pd.DataFrame(transfers)

        if not self.transfer_plan.empty:
            # Calculate total cost and summary metrics
            total_units = self.transfer_plan["units"].sum()
            total_cost = self.transfer_plan["transport_cost"].sum()
            avg_cost_per_unit = total_cost / total_units if total_units > 0 else 0

            print(f"Linear Programming Transfer Plan Summary:")
            print(f"- Optimization status: {LpStatus[self.status]}")
            print(f"- Total transfers: {len(self.transfer_plan)}")
            print(f"- Total units to transfer: {total_units}")
            print(f"- Total transport cost: {total_cost:,.0f} VND")
            print(f"- Average cost per unit: {avg_cost_per_unit:,.0f} VND")
        else:
            print(
                f"No transfers recommended. Optimization status: {LpStatus[self.status]}"
            )

        return self.transfer_plan

    def add_store_product_names(self, stores_df=None, products_df=None):
        """
        Add store and product names to the transfer plan for better readability.

        Args:
            stores_df: DataFrame containing store information
            products_df: DataFrame containing product information
        """
        if self.transfer_plan is None or self.transfer_plan.empty:
            return

        if stores_df is not None:
            # Add store names
            store_name_map = stores_df.set_index("store_id")["store_name"].to_dict()
            self.transfer_plan["from_store"] = self.transfer_plan["from_store_id"].map(
                store_name_map
            )
            self.transfer_plan["to_store"] = self.transfer_plan["to_store_id"].map(
                store_name_map
            )

        if products_df is not None:
            # Add product names
            product_name_map = products_df.set_index("product_id")[
                "product_name"
            ].to_dict()
            self.transfer_plan["product"] = self.transfer_plan["product_id"].map(
                product_name_map
            )


if __name__ == "__main__":
    import os
    import sys

    src_path = os.path.abspath(os.path.join("..", ""))

    # print(sys.path)
    if src_path not in sys.path:
        print(src_path)
        sys.path.append(src_path)

    from src.engine.analyzer import InventoryAnalyzer

    # Check if data files exist
    data_dir = "data"
    required_files = [
        "sales_data.csv",
        "inventory_data.csv",
        "distance_matrix.csv",
        "transport_cost_matrix.csv",
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"Required file {file} not found. Please run data generator first.")
            exit(1)

    # Create analyzer
    analyzer = InventoryAnalyzer()

    # Load data
    analyzer.load_data(
        sales_path=os.path.join(data_dir, "sales_data.csv"),
        inventory_path=os.path.join(data_dir, "inventory_data.csv"),
        stores_path=(
            os.path.join(data_dir, "stores.csv")
            if os.path.exists(os.path.join(data_dir, "stores.csv"))
            else None
        ),
        products_path=(
            os.path.join(data_dir, "products.csv")
            if os.path.exists(os.path.join(data_dir, "products.csv"))
            else None
        ),
    )

    # Analyze data
    analyzer.analyze_sales_data()

    # Identify imbalances
    excess_df, needed_df = analyzer.identify_inventory_imbalances()

    # Create optimizer
    optimizer = LinearProgrammingOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(data_dir, "transport_cost_matrix.csv"),
    )

    # Generate transfer plan
    transfer_plan = optimizer.optimize(excess_df, needed_df)

    # Add store and product names if data available
    if os.path.exists(os.path.join(data_dir, "stores.csv")) and os.path.exists(
        os.path.join(data_dir, "products.csv")
    ):
        stores_df = pd.read_csv(os.path.join(data_dir, "stores.csv"))
        products_df = pd.read_csv(os.path.join(data_dir, "products.csv"))
        optimizer.add_store_product_names(stores_df, products_df)

    # Evaluate impact
    if not transfer_plan.empty:
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save transfer plan
        output_path = os.path.join(data_dir, "lp_transfers.csv")
        transfer_plan.to_csv(output_path, index=False)
        print(f"Transfer plan saved to {output_path}")
