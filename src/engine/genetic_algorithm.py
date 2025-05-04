"""
Genetic Algorithm Optimization Engine
------------------------------------------
Implements a genetic algorithm approach for inventory transfer optimization.
Handles complex constraints and non-linear cost functions.
Fixes handling of store IDs for distance and cost lookups.
"""

import random

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from tqdm import tqdm


class GeneticAlgorithmOptimizer:
    def __init__(
        self, distance_matrix=None, transport_cost_matrix=None, random_seed=42
    ):
        """
        Initialize the genetic algorithm optimization engine.

        Args:
            distance_matrix: Matrix of distances between stores
            transport_cost_matrix: Matrix of transport costs between stores
            random_seed: Random seed for reproducibility
        """
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.transfer_plan = None
        self.random_seed = random_seed
        self.best_solution = None
        self.best_fitness = None

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

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

    def optimize(
        self,
        excess_inventory,
        needed_inventory,
        population_size=100,
        num_generations=50,
        crossover_prob=0.7,
        mutation_prob=0.2,
        tournament_size=3,
        verbose=True,
    ):
        """
        Generate a transfer plan using a genetic algorithm approach.

        Args:
            excess_inventory: DataFrame containing excess inventory
            needed_inventory: DataFrame containing needed inventory
            population_size: Size of the population (default: 100)
            num_generations: Number of generations to evolve (default: 50)
            crossover_prob: Crossover probability (default: 0.7)
            mutation_prob: Mutation probability (default: 0.2)
            tournament_size: Tournament size for selection (default: 3)
            verbose: Whether to show progress (default: True)

        Returns:
            DataFrame containing transfer recommendations
        """
        print("Generating genetic algorithm-based transfer plan...")

        # If either of these is empty, there's nothing to optimize
        if excess_inventory.empty or needed_inventory.empty:
            print("No excess or needed inventory found. No transfers needed.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        # Group by product to match supply and demand
        excess_by_product = excess_inventory.groupby("product_id")["excess_units"].sum()
        needed_by_product = needed_inventory.groupby("product_id")["needed_units"].sum()

        # Get products that have both excess and need
        valid_products = list(
            set(excess_by_product.index) & set(needed_by_product.index)
        )

        if not valid_products:
            print("No products with both excess and need found.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        # Filter to valid products
        excess_df = excess_inventory[
            excess_inventory["product_id"].isin(valid_products)
        ].copy()
        needed_df = needed_inventory[
            needed_inventory["product_id"].isin(valid_products)
        ].copy()

        # Create lookup for excess and needed items
        excess_items = []
        for _, row in excess_df.iterrows():
            excess_items.append(
                (row["store_id"], row["product_id"], row["excess_units"])
            )

        needed_items = []
        for _, row in needed_df.iterrows():
            needed_items.append(
                (row["store_id"], row["product_id"], row["needed_units"])
            )

        # Set up the genetic algorithm
        # If creator already has FitnessMin, skip creation
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # If creator already has Individual, skip creation
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Define how to generate an individual (a transfer plan)
        def create_transfer_plan():
            """Create a random transfer plan."""
            transfer_plan = []

            # Create a copy of excess and needed items
            excess_copy = excess_items.copy()
            needed_copy = needed_items.copy()

            # Shuffle to introduce randomness
            random.shuffle(excess_copy)
            random.shuffle(needed_copy)

            # Match excess to need
            for product_id in valid_products:
                # Filter by product
                product_excess = [item for item in excess_copy if item[1] == product_id]
                product_needed = [item for item in needed_copy if item[1] == product_id]

                if not product_excess or not product_needed:
                    continue

                # Initialize remaining units
                excess_remaining = {item[0]: item[2] for item in product_excess}
                needed_remaining = {item[0]: item[2] for item in product_needed}

                # For each need, try to fulfill from excess
                for need_store, _, _ in product_needed:
                    if needed_remaining[need_store] <= 0:
                        continue

                    # Sort excess stores by distance (closest first)
                    pairs_to_sort = [
                        (store, product_id)
                        for store, prod_id, _ in product_excess
                        if prod_id == product_id and store != need_store
                    ]

                    # Define a key function for sorting by distance
                    def distance_key(pair):
                        store, _ = pair
                        if (
                            store in self.distance_matrix.index
                            and need_store in self.distance_matrix.columns
                        ):
                            return float(self.distance_matrix.loc[store, need_store])
                        return float("inf")  # Use infinity if distance not found

                    # Sort by distance
                    sorted_excess = sorted(pairs_to_sort, key=distance_key)

                    # Try to fulfill from each excess store
                    for excess_store, _ in sorted_excess:
                        if excess_store == need_store:  # Skip self-transfers
                            continue

                        if excess_remaining[excess_store] <= 0:
                            continue

                        # Calculate units to transfer
                        transfer_units = min(
                            excess_remaining[excess_store], needed_remaining[need_store]
                        )

                        if transfer_units > 0:
                            # Add to transfer plan
                            transfer_plan.append(
                                (excess_store, product_id, need_store, transfer_units)
                            )

                            # Update remaining units
                            excess_remaining[excess_store] -= transfer_units
                            needed_remaining[need_store] -= transfer_units

                        # Check if need is fulfilled
                        if needed_remaining[need_store] <= 0:
                            break

            return transfer_plan

        # Register how to create individuals
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, create_transfer_plan
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Define the evaluation function
        def evaluate_transfer_plan(individual):
            """Evaluate the cost of a transfer plan."""
            total_cost = 0

            for from_store, product_id, to_store, units in individual:
                # Calculate transport cost directly from the matrix
                if (
                    from_store in self.transport_cost_matrix.index
                    and to_store in self.transport_cost_matrix.columns
                ):
                    cost_per_unit = float(
                        self.transport_cost_matrix.loc[from_store, to_store]
                    )
                    total_cost += cost_per_unit * units
                else:
                    # Fallback to distance-based cost if transport cost not available
                    if (
                        from_store in self.distance_matrix.index
                        and to_store in self.distance_matrix.columns
                    ):
                        distance = float(self.distance_matrix.loc[from_store, to_store])
                        total_cost += distance * 10000 * units  # Basic cost estimate
                    else:
                        # High cost for unknown distances
                        total_cost += 1000000 * units

            return (total_cost,)

        # Register the evaluation function
        toolbox.register("evaluate", evaluate_transfer_plan)

        # Define crossover, mutation, and selection operators
        def crossover(ind1, ind2):
            """Custom crossover for transfer plans."""
            # Create children as copies of parents
            child1 = creator.Individual(ind1)
            child2 = creator.Individual(ind2)

            # Perform crossover
            if (
                len(ind1) > 0 and len(ind2) > 0
            ):  # Only crossover if both parents have transfers
                # Choose crossover point
                point = random.randint(1, min(len(ind1), len(ind2)) - 1)

                # Swap transfers after crossover point
                child1 = creator.Individual(ind1[:point] + ind2[point:])
                child2 = creator.Individual(ind2[:point] + ind1[point:])

                # Validate and fix children
                child1 = validate_and_fix(child1)
                child2 = validate_and_fix(child2)

            return child1, child2

        def mutate(individual):
            """Custom mutation for transfer plans."""
            # Make a copy of the individual
            mutant = creator.Individual(individual)

            if len(mutant) > 0:  # Only mutate if there are transfers
                # Choose a random transfer to mutate
                idx = random.randint(0, len(mutant) - 1)
                transfer = mutant[idx]

                # Extract transfer details
                from_store, product_id, to_store, units = transfer

                # Select mutation type
                mutation_type = random.randint(0, 2)

                if mutation_type == 0:  # Change source store
                    # Find another store with excess of this product
                    other_excess = [
                        item
                        for item in excess_items
                        if item[0] != from_store and item[1] == product_id
                    ]

                    if other_excess:
                        new_from_store = random.choice(other_excess)[0]
                        mutant[idx] = (new_from_store, product_id, to_store, units)

                elif mutation_type == 1:  # Change destination store
                    # Find another store with need for this product
                    other_needed = [
                        item
                        for item in needed_items
                        if item[0] != to_store and item[1] == product_id
                    ]

                    if other_needed:
                        new_to_store = random.choice(other_needed)[0]
                        mutant[idx] = (from_store, product_id, new_to_store, units)

                else:  # Change units
                    # Find max available excess and need
                    max_excess = next(
                        (
                            item[2]
                            for item in excess_items
                            if item[0] == from_store and item[1] == product_id
                        ),
                        0,
                    )
                    max_need = next(
                        (
                            item[2]
                            for item in needed_items
                            if item[0] == to_store and item[1] == product_id
                        ),
                        0,
                    )

                    # Calculate remaining units from other transfers
                    excess_used = sum(
                        t[3]
                        for t in mutant
                        if t[0] == from_store and t[1] == product_id and t != transfer
                    )
                    need_filled = sum(
                        t[3]
                        for t in mutant
                        if t[2] == to_store and t[1] == product_id and t != transfer
                    )

                    # Calculate available range
                    min_units = 1
                    max_units = min(max_excess - excess_used, max_need - need_filled)

                    if max_units > min_units:
                        new_units = random.randint(min_units, max_units)
                        mutant[idx] = (from_store, product_id, to_store, new_units)

            # Validate and fix the mutant
            mutant = validate_and_fix(mutant)
            return (mutant,)

        def validate_and_fix(individual):
            """Validate and fix a transfer plan to ensure it respects constraints."""
            # Create a copy to work with
            fixed = []

            # Track how much has been transferred from each excess item
            excess_used = {(item[0], item[1]): 0 for item in excess_items}

            # Track how much has been transferred to each needed item
            need_filled = {(item[0], item[1]): 0 for item in needed_items}

            # Process each transfer
            for transfer in individual:
                from_store, product_id, to_store, units = transfer

                # Skip self-transfers
                if from_store == to_store:
                    continue

                # Get maximum available excess
                max_excess = next(
                    (
                        item[2]
                        for item in excess_items
                        if item[0] == from_store and item[1] == product_id
                    ),
                    0,
                )

                # Get maximum needed
                max_need = next(
                    (
                        item[2]
                        for item in needed_items
                        if item[0] == to_store and item[1] == product_id
                    ),
                    0,
                )

                # Check if this transfer is valid
                if (
                    max_excess == 0
                    or max_need == 0
                    or from_store == to_store
                    or units <= 0
                ):
                    continue  # Skip invalid transfer

                # Calculate remaining capacity
                remaining_excess = max_excess - excess_used.get(
                    (from_store, product_id), 0
                )
                remaining_need = max_need - need_filled.get((to_store, product_id), 0)

                if remaining_excess <= 0 or remaining_need <= 0:
                    continue  # No capacity left

                # Adjust units to respect constraints
                adjusted_units = min(units, remaining_excess, remaining_need)

                if adjusted_units > 0:
                    # Add valid transfer
                    fixed.append((from_store, product_id, to_store, adjusted_units))

                    # Update tracking
                    excess_used[(from_store, product_id)] = (
                        excess_used.get((from_store, product_id), 0) + adjusted_units
                    )
                    need_filled[(to_store, product_id)] = (
                        need_filled.get((to_store, product_id), 0) + adjusted_units
                    )

            return creator.Individual(fixed)

        # Register the genetic operators
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)

        # Create the initial population
        population = toolbox.population(n=population_size)

        # Track the best individual
        hof = tools.HallOfFame(1)

        # Statistics to track
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the genetic algorithm
        if verbose:
            print(
                f"Running genetic algorithm with {population_size} individuals for {num_generations} generations..."
            )

            # Run with progress bar
            algorithms.eaSimple(
                population,
                toolbox,
                cxpb=crossover_prob,
                mutpb=mutation_prob,
                ngen=num_generations,
                stats=stats,
                halloffame=hof,
                verbose=False,
            )

            # Display progress manually
            for gen in tqdm(range(num_generations), desc="Generations"):
                # Evaluate and gather stats for this generation
                fits = [
                    ind.fitness.values[0] for ind in population if ind.fitness.valid
                ]
                if fits:
                    min_fit = min(fits)
                    avg_fit = sum(fits) / len(fits)
                    if gen % 10 == 0:  # Only show every 10 generations
                        print(
                            f"Generation {gen}: Min={min_fit:,.0f}, Avg={avg_fit:,.0f}"
                        )
        else:
            # Run without verbose output
            algorithms.eaSimple(
                population,
                toolbox,
                cxpb=crossover_prob,
                mutpb=mutation_prob,
                ngen=num_generations,
                stats=stats,
                halloffame=hof,
                verbose=False,
            )

        # Store best solution and fitness
        if len(hof) > 0:
            self.best_solution = hof[0]
            self.best_fitness = hof[0].fitness.values[0]
        else:
            print("No solution found.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        # Get the best solution
        best_individual = hof[0]

        # Convert the best solution to a transfer plan
        transfers = []

        for from_store, product_id, to_store, units in best_individual:
            # Get distance from matrix
            if (
                from_store in self.distance_matrix.index
                and to_store in self.distance_matrix.columns
            ):
                distance = float(self.distance_matrix.loc[from_store, to_store])
            else:
                distance = 0

            # Get transport cost from matrix
            if (
                from_store in self.transport_cost_matrix.index
                and to_store in self.transport_cost_matrix.columns
            ):
                transport_cost = (
                    float(self.transport_cost_matrix.loc[from_store, to_store]) * units
                )
            else:
                transport_cost = distance * 10000 * units  # Basic cost estimate

            transfers.append(
                {
                    "from_store_id": from_store,
                    "to_store_id": to_store,
                    "product_id": product_id,
                    "units": int(units),
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

            print(f"Genetic Algorithm Transfer Plan Summary:")
            print(f"- Best fitness (total cost): {self.best_fitness:,.0f}")
            print(f"- Total transfers: {len(self.transfer_plan)}")
            print(f"- Total units to transfer: {total_units}")
            print(f"- Total transport cost: {total_cost:,.0f} VND")
            print(f"- Average cost per unit: {avg_cost_per_unit:,.0f} VND")
        else:
            print("No transfers recommended by genetic algorithm.")

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
    optimizer = GeneticAlgorithmOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(data_dir, "transport_cost_matrix.csv"),
    )

    # Generate transfer plan (using a smaller population and fewer generations for testing)
    transfer_plan = optimizer.optimize(
        excess_df,
        needed_df,
        population_size=50,  # Smaller for testing
        num_generations=20,  # Fewer for testing
        verbose=True,
    )

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
        output_path = os.path.join(data_dir, "ga_transfers.csv")
        transfer_plan.to_csv(output_path, index=False)
        print(f"Transfer plan saved to {output_path}")
