"""
Visualization Module
-----------------
Creates visualizations for the inventory transfer optimization system.
Includes maps, charts, and graphs to visualize inventory status and transfer plans.
"""

import base64
import io
import os

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium import plugins
from matplotlib.colors import LinearSegmentedColormap


class InventoryVisualizer:
    def __init__(self, stores_df=None, output_dir="visualizations"):
        """
        Initialize the visualizer with store data.

        Args:
            stores_df: DataFrame containing store information
            output_dir: Directory to save visualizations
        """
        self.stores_df = stores_df
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up color palettes
        self.status_colors = {
            "Excess": "#FF8042",  # Orange
            "Needed": "#0088FE",  # Blue
            "Balanced": "#00C49F",  # Green
        }

        # Set up plot style
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("colorblind")

    def load_stores(self, stores_path):
        """
        Load store data from CSV file.

        Args:
            stores_path: Path to stores CSV file
        """
        self.stores_df = pd.read_csv(stores_path)

    def visualize_inventory_status(self, analysis_df, save_html=True, show_map=True):
        """
        Create visualizations of inventory status across stores and products.

        Args:
            analysis_df: DataFrame with inventory analysis data
            save_html: Whether to save the map as an HTML file
            show_map: Whether to return the map object

        Returns:
            Folium map object if show_map is True
        """
        print("Generating inventory status visualizations...")

        if self.stores_df is None:
            print(
                "Store data not available. Map visualization requires store location data."
            )
            return None

        # Create an interactive map showing store locations and inventory status
        map_center = [16.8, 107.0]  # Center of Vietnam
        m = folium.Map(location=map_center, zoom_start=6, tiles="CartoDB positron")

        # Count excess and needed items per store
        store_status = (
            analysis_df.groupby("store_id")["inventory_status"]
            .value_counts()
            .unstack()
            .fillna(0)
        )

        # Create a store lookup for easier access
        store_lookup = self.stores_df.set_index("store_id").to_dict(orient="index")

        # Add store markers with pie charts showing inventory status
        for store_id, status_counts in store_status.iterrows():
            # Get store details from lookup
            if store_id in store_lookup:
                store = store_lookup[store_id]

                # Get inventory status counts for this store
                excess = status_counts.get("Excess", 0)
                needed = status_counts.get("Needed", 0)
                balanced = status_counts.get("Balanced", 0)

                total = excess + needed + balanced

                # Create popup content
                popup_content = f"""
                <b>{store['store_name']}</b><br>
                City: {store['city']}<br>
                <hr>
                <b>Inventory Status:</b><br>
                Excess items: {int(excess)} ({excess/total*100:.1f}%)<br>
                Needed items: {int(needed)} ({needed/total*100:.1f}%)<br>
                Balanced items: {int(balanced)} ({balanced/total*100:.1f}%)<br>
                """

                # Determine marker color based on dominant status
                if excess > needed and excess > balanced:
                    color = "red"  # More excess inventory
                elif needed > excess and needed > balanced:
                    color = "orange"  # More needed inventory
                else:
                    color = "green"  # Balanced

                # Add marker
                folium.Marker(
                    location=[store["latitude"], store["longitude"]],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=color, icon="info-sign"),
                    tooltip=store["store_name"],
                ).add_to(m)

        # Save map to HTML file
        if save_html:
            output_path = os.path.join(self.output_dir, "inventory_status_map.html")
            m.save(output_path)
            print(f"Inventory status map saved to {output_path}")

        # Create additional visualizations
        self._create_status_bar_charts(analysis_df)

        if "city" in analysis_df.columns:
            self._create_city_charts(analysis_df)

        if "category" in analysis_df.columns:
            self._create_category_charts(analysis_df)

        if show_map:
            return m

        return None

    def visualize_inventory_distribution_comparison(
        self,
        analysis_df,
        post_analysis,
        store_names,
        product_names,
        algorithm_name,
        save_png=True,
        show_plot=True,
    ):
        """
        Visualize inventory distribution before and after transfers for the largest stores.

        Args:
            analysis_df: DataFrame with original inventory analysis data (before)
            post_analysis: DataFrame with post-transfer inventory analysis data (after)
            store_names: Dictionary mapping store IDs to store names
            product_names: Dictionary mapping product IDs to product names
            algorithm_name: Name of the optimization algorithm used
            save_png: Whether to save the plot as a PNG file
            show_plot: Whether to show the plot

        Returns:
            Matplotlib figure if show_plot is True
        """
        print(f"Generating inventory distribution comparison for {algorithm_name}...")

        # Find the 5 largest stores by total inventory
        store_inventory = (
            analysis_df.groupby("store_id")["current_stock"]
            .sum()
            .sort_values(ascending=False)
        )
        top_stores = store_inventory.head(5).index.tolist()

        # Filter data for top stores
        before_data = analysis_df[analysis_df["store_id"].isin(top_stores)]
        after_data = post_analysis[post_analysis["store_id"].isin(top_stores)]

        # Get store names for top stores
        store_labels = [
            store_names.get(store_id, f"Store {store_id}") for store_id in top_stores
        ]

        # Create figure with subplots for each store
        fig, axes = plt.subplots(len(top_stores), 2, figsize=(15, 4 * len(top_stores)))

        # For a single store case, make axes indexable as 2D
        if len(top_stores) == 1:
            axes = np.array([axes]).reshape(1, 2)

        # Plot inventory status for each store
        for i, store_id in enumerate(top_stores):
            store_name = store_labels[i]

            # Before data
            before_store = before_data[before_data["store_id"] == store_id]
            status_counts_before = before_store["inventory_status"].value_counts()

            # After data
            after_store = after_data[after_data["store_id"] == store_id]
            status_counts_after = after_store["inventory_status"].value_counts()

            # Plot before
            status_counts_before.plot(
                kind="pie",
                ax=axes[i, 0],
                autopct="%1.1f%%",
                startangle=90,
                colors=[
                    self.status_colors.get(s, "#999999")
                    for s in status_counts_before.index
                ],
                wedgeprops={"edgecolor": "w", "linewidth": 1},
                labels=None,  # Remove default labels
            )
            axes[i, 0].set_title(f"{store_name} - Before")
            axes[i, 0].set_ylabel("")

            # Add custom legend for the first subplot
            if i == 0:
                axes[i, 0].legend(
                    status_counts_before.index,
                    loc="upper right",
                    bbox_to_anchor=(0, 0, 0.95, 0.95),
                )

            # Plot after
            status_counts_after.plot(
                kind="pie",
                ax=axes[i, 1],
                autopct="%1.1f%%",
                startangle=90,
                colors=[
                    self.status_colors.get(s, "#999999")
                    for s in status_counts_after.index
                ],
                wedgeprops={"edgecolor": "w", "linewidth": 1},
                labels=None,  # Remove default labels
            )
            axes[i, 1].set_title(f"{store_name} - After")
            axes[i, 1].set_ylabel("")

            # Add custom legend for the first subplot
            if i == 0:
                axes[i, 1].legend(
                    status_counts_after.index,
                    loc="upper right",
                    bbox_to_anchor=(0, 0, 0.95, 0.95),
                )

        plt.suptitle(
            f"Inventory Status Distribution Before and After - {algorithm_name}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Create top products by transfers figure
        fig2 = self._create_top_products_transfers(
            analysis_df, post_analysis, product_names, algorithm_name
        )

        # Save figure
        if save_png:
            output_path = os.path.join(
                self.output_dir,
                f"inventory_distribution_{algorithm_name.lower().replace(' ', '_')}.png",
            )
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(
                f"Inventory distribution comparison for {algorithm_name} saved to {output_path}"
            )

        # If not showing plot, close figure to free memory
        if not show_plot:
            plt.close(fig)
            plt.close(fig2)
            return None

        return fig

    def _create_top_products_transfers(
        self, analysis_df, post_analysis, product_names, algorithm_name
    ):
        """Create visualization for top products by transfer volume."""
        # Calculate the difference in inventory before and after transfers
        merged_df = pd.merge(
            analysis_df[["store_id", "product_id", "current_stock"]],
            post_analysis[["store_id", "product_id", "current_stock"]],
            on=["store_id", "product_id"],
            suffixes=("_before", "_after"),
        )

        # Calculate absolute change in inventory
        merged_df["abs_change"] = abs(
            merged_df["current_stock_after"] - merged_df["current_stock_before"]
        )

        # Group by product_id and sum the absolute changes
        product_changes = (
            merged_df.groupby("product_id")["abs_change"]
            .sum()
            .sort_values(ascending=False)
        )

        # Get top 10 products
        top_products = product_changes.head(10)

        # Replace product IDs with names
        top_products.index = [
            product_names.get(pid, f"Product {pid}") for pid in top_products.index
        ]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create horizontal bar chart
        bars = ax.barh(top_products.index, top_products.values, color="#1f77b4")

        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 10,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                ha="left",
                va="center",
            )

        ax.set_xlabel("Total Units Transferred")
        ax.set_title(f"Top 10 Products by Transfer Volume - {algorithm_name}")

        # Invert y-axis to have largest at top
        ax.invert_yaxis()

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(
            self.output_dir,
            f"top_products_{algorithm_name.lower().replace(' ', '_')}.png",
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(
            f"Top products by transfer volume for {algorithm_name} saved to {output_path}"
        )

        return fig

    def visualize_transfer_plan(self, transfer_plan, save_html=True, show_map=True):
        """
        Visualize the transfer plan on a map and with charts.

        Args:
            transfer_plan: DataFrame containing transfer recommendations
            save_html: Whether to save the map as an HTML file
            show_map: Whether to return the map object

        Returns:
            Folium map object if show_map is True
        """
        if transfer_plan is None or transfer_plan.empty:
            print("No transfer plan to visualize.")
            return None

        if self.stores_df is None:
            print(
                "Store data not available. Map visualization requires store location data."
            )
            return None

        print("Generating transfer plan visualizations...")

        # Create a map showing transfers
        map_center = [16.8, 107.0]  # Center of Vietnam
        m = folium.Map(location=map_center, zoom_start=6, tiles="CartoDB positron")

        # Add store markers
        for _, store in self.stores_df.iterrows():
            folium.CircleMarker(
                location=[store["latitude"], store["longitude"]],
                radius=5,
                color="blue",
                fill=True,
                fill_opacity=0.7,
                popup=store["store_name"],
            ).add_to(m)

        # Group transfers by origin-destination pairs
        transfer_summary = (
            transfer_plan.groupby(["from_store_id", "to_store_id"])
            .agg({"units": "sum", "transport_cost": "sum"})
            .reset_index()
        )

        # Create a store lookup for easier access
        store_lookup = self.stores_df.set_index("store_id").to_dict(orient="index")

        # Add transfer lines
        for _, row in transfer_summary.iterrows():
            from_store_id = row["from_store_id"]
            to_store_id = row["to_store_id"]

            if from_store_id in store_lookup and to_store_id in store_lookup:
                from_store = store_lookup[from_store_id]
                to_store = store_lookup[to_store_id]

                # Create line weight based on units transferred
                units = row["units"]
                weight = 1 + min(5, units / 100)  # Scale line thickness

                # Create popup content
                popup_content = f"""
                <b>Transfer Summary:</b><br>
                From: {from_store['store_name']}<br>
                To: {to_store['store_name']}<br>
                Units: {units}<br>
                Cost: {row['transport_cost']:,.0f} VND
                """

                # Add line representing transfer
                folium.PolyLine(
                    locations=[
                        [from_store["latitude"], from_store["longitude"]],
                        [to_store["latitude"], to_store["longitude"]],
                    ],
                    color="red",
                    weight=weight,
                    opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300),
                ).add_to(m)

                # Add arrow marker at midpoint
                mid_lat = (from_store["latitude"] + to_store["latitude"]) / 2
                mid_lon = (from_store["longitude"] + to_store["longitude"]) / 2

                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="font-size: 12pt; color: red;">&rarr;</div>',
                    ),
                ).add_to(m)

        # Save map to HTML file
        if save_html:
            output_path = os.path.join(self.output_dir, "transfer_plan_map.html")
            m.save(output_path)
            print(f"Transfer plan map saved to {output_path}")

        # Create additional visualizations
        self._create_transfer_charts(transfer_plan)

        if show_map:
            return m

        return None

    def visualize_impact(
        self,
        impact_df,
        analysis_df=None,
        post_analysis=None,
        store_names=None,
        product_names=None,
        algorithm_name="Transfer Plan",
        save_png=True,
        show_plot=True,
    ):
        """
        Visualize the impact of the transfer plan.

        Args:
            impact_df: DataFrame with impact analysis
            analysis_df: Original inventory analysis DataFrame (before transfer)
            post_analysis: Post-transfer inventory analysis DataFrame
            store_names: Dictionary mapping store IDs to store names
            product_names: Dictionary mapping product IDs to product names
            algorithm_name: Name of the optimization algorithm used
            save_png: Whether to save the plots as PNG files
            show_plot: Whether to show the plots

        Returns:
            List of matplotlib figures if show_plot is True
        """
        if impact_df is None:
            print("No impact data to visualize.")
            return None

        print(f"Generating impact visualizations for {algorithm_name}...")

        figures = []

        # Extract key metrics
        before = impact_df["Before Transfer"]
        after = impact_df["After Transfer"]
        improvement = impact_df["Improvement"]

        # Create inventory status chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # Prepare data for bar chart
        status_labels = ["Excess Items", "Needed Items", "Balanced Items"]
        before_values = [before.get(label, 0) for label in status_labels]
        after_values = [after.get(label, 0) for label in status_labels]

        x = np.arange(len(status_labels))
        width = 0.35

        # Create grouped bar chart
        ax1.bar(
            x - width / 2,
            before_values,
            width,
            label="Before Transfer",
            color="#FF9999",
        )
        ax1.bar(
            x + width / 2, after_values, width, label="After Transfer", color="#99FF99"
        )

        # Add text labels above bars
        for i, v in enumerate(before_values):
            ax1.text(i - width / 2, v + 5, str(int(v)), ha="center")

        for i, v in enumerate(after_values):
            ax1.text(i + width / 2, v + 5, str(int(v)), ha="center")

        # Add percentage change
        for i, label in enumerate(status_labels):
            if label in improvement:
                change = improvement.get(label)
                if isinstance(change, (int, float)):
                    if change > 0:
                        label_text = f"+{change}"
                    else:
                        label_text = f"{change}"

                    # Position the text between the bars
                    ax1.text(
                        i,
                        max(before_values[i], after_values[i]) + 20,
                        label_text,
                        ha="center",
                        fontweight="bold",
                    )

        ax1.set_ylabel("Number of Items")
        ax1.set_title(f"Impact on Inventory Status - {algorithm_name}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(status_labels)
        ax1.legend()

        plt.tight_layout()
        figures.append(fig1)

        if save_png:
            output_path = os.path.join(
                self.output_dir,
                f"impact_inventory_status_{algorithm_name.lower().replace(' ', '_')}.png",
            )
            fig1.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Impact chart for {algorithm_name} saved to {output_path}")

        # Create inventory distribution comparison if data available
        if (
            analysis_df is not None
            and post_analysis is not None
            and store_names is not None
        ):
            dist_fig = self.visualize_inventory_distribution_comparison(
                analysis_df,
                post_analysis,
                store_names,
                product_names,
                algorithm_name,
                save_png=save_png,
                show_plot=show_plot,
            )
            if dist_fig is not None:
                figures.append(dist_fig)

        # Create key metrics chart
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Prepare data for key metrics chart
        metric_labels = [
            "Avg Days of Inventory",
            "Inventory Imbalance (StdDev)",
            "Product Turnover",
        ]
        before_metrics = [before.get(label, 0) for label in metric_labels]
        after_metrics = [after.get(label, 0) for label in metric_labels]

        x = np.arange(len(metric_labels))

        # Create grouped bar chart
        ax2.bar(
            x - width / 2,
            before_metrics,
            width,
            label="Before Transfer",
            color="#FF9999",
        )
        ax2.bar(
            x + width / 2, after_metrics, width, label="After Transfer", color="#99FF99"
        )

        # Add text labels above bars
        for i, v in enumerate(before_metrics):
            ax2.text(i - width / 2, v + 0.5, f"{v:.1f}", ha="center")

        for i, v in enumerate(after_metrics):
            ax2.text(i + width / 2, v + 0.5, f"{v:.1f}", ha="center")

        # Add improvement percentages
        improvement_metrics = {
            "Avg Days of Inventory": (
                (after_metrics[0] - before_metrics[0]) / before_metrics[0] * 100
                if before_metrics[0] > 0
                else 0
            ),
            "Inventory Imbalance (StdDev)": (
                (after_metrics[1] - before_metrics[1]) / before_metrics[1] * 100
                if before_metrics[1] > 0
                else 0
            ),
            "Product Turnover": (
                (after_metrics[2] - before_metrics[2]) / before_metrics[2] * 100
                if before_metrics[2] > 0
                else 0
            ),
        }

        for i, label in enumerate(metric_labels):
            if label in improvement_metrics:
                change = improvement_metrics[label]
                change_text = f"{change:.1f}%" if change >= 0 else f"{change:.1f}%"
                ax2.text(
                    i,
                    max(before_metrics[i], after_metrics[i]) + 1,
                    change_text,
                    ha="center",
                    fontweight="bold",
                )

        ax2.set_ylabel("Value")
        ax2.set_title(f"Impact on Key Performance Metrics - {algorithm_name}")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels)
        ax2.legend()

        plt.tight_layout()
        figures.append(fig2)

        if save_png:
            output_path = os.path.join(
                self.output_dir,
                f"impact_performance_metrics_{algorithm_name.lower().replace(' ', '_')}.png",
            )
            fig2.savefig(output_path, dpi=300, bbox_inches="tight")
            print(
                f"Performance metrics chart for {algorithm_name} saved to {output_path}"
            )

        # If not showing plots, close figures to free memory
        if not show_plot:
            for fig in figures:
                plt.close(fig)
            return None

        return figures

    def _create_status_bar_charts(self, analysis_df):
        """Create bar charts showing inventory status distribution."""
        # Count items in each status
        status_counts = analysis_df["inventory_status"].value_counts()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bar chart
        bars = ax.bar(
            status_counts.index,
            status_counts.values,
            color=[self.status_colors.get(s, "#999999") for s in status_counts.index],
        )

        # Add text labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height} ({height/len(analysis_df)*100:.1f}%)",
                ha="center",
                va="bottom",
            )

        ax.set_ylabel("Number of Items")
        ax.set_title("Inventory Status Distribution")

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(self.output_dir, "inventory_status_distribution.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Inventory status distribution chart saved to {output_path}")

        plt.close(fig)

    def _create_city_charts(self, analysis_df):
        """Create charts showing inventory status by city."""
        # Group by city and inventory status
        city_status = (
            analysis_df.groupby(["city", "inventory_status"]).size().unstack().fillna(0)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create stacked bar chart
        city_status.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[self.status_colors.get(s, "#999999") for s in city_status.columns],
        )

        ax.set_ylabel("Number of Items")
        ax.set_title("Inventory Status by City")
        ax.legend(title="Status")

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(self.output_dir, "inventory_status_by_city.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Inventory status by city chart saved to {output_path}")

        plt.close(fig)

    def _create_category_charts(self, analysis_df):
        """Create charts showing inventory status by product category."""
        # Group by category and inventory status
        category_status = (
            analysis_df.groupby(["category", "inventory_status"])
            .size()
            .unstack()
            .fillna(0)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create stacked bar chart
        category_status.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[
                self.status_colors.get(s, "#999999") for s in category_status.columns
            ],
        )

        ax.set_ylabel("Number of Items")
        ax.set_title("Inventory Status by Product Category")
        ax.legend(title="Status")

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(self.output_dir, "inventory_status_by_category.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Inventory status by category chart saved to {output_path}")

        plt.close(fig)

    def _create_transfer_charts(self, transfer_plan):
        """Create charts showing transfer distribution."""
        # Create figure for units transferred by store
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Group by from_store_id and sum units
        from_units = (
            transfer_plan.groupby("from_store_id")["units"]
            .sum()
            .sort_values(ascending=False)
        )

        # Group by to_store_id and sum units
        to_units = (
            transfer_plan.groupby("to_store_id")["units"]
            .sum()
            .sort_values(ascending=False)
        )

        # If store names are available, use them
        if "from_store" in transfer_plan.columns:
            # Create mapping from ID to name
            from_store_name_map = transfer_plan.set_index("from_store_id")[
                "from_store"
            ].to_dict()
            to_store_name_map = transfer_plan.set_index("to_store_id")[
                "to_store"
            ].to_dict()

            # Create named series
            from_units_named = pd.Series(
                from_units.values,
                index=[
                    from_store_name_map.get(idx, f"Store {idx}")
                    for idx in from_units.index
                ],
                name=from_units.name,
            )

            to_units_named = pd.Series(
                to_units.values,
                index=[
                    to_store_name_map.get(idx, f"Store {idx}") for idx in to_units.index
                ],
                name=to_units.name,
            )

            # Plot with names
            from_units_named.head(10).plot(kind="barh", ax=ax1, color="#FF8042")
            to_units_named.head(10).plot(kind="barh", ax=ax2, color="#0088FE")
        else:
            # Plot with IDs
            from_units.head(10).plot(kind="barh", ax=ax1, color="#FF8042")
            to_units.head(10).plot(kind="barh", ax=ax2, color="#0088FE")

        ax1.set_xlabel("Units Transferred Out")
        ax1.set_title("Top 10 Source Stores")
        ax1.invert_yaxis()  # Invert y-axis to have largest at top

        ax2.set_xlabel("Units Transferred In")
        ax2.set_title("Top 10 Destination Stores")
        ax2.invert_yaxis()  # Invert y-axis to have largest at top

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(self.output_dir, "transfer_by_store.png")
        fig1.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Transfer by store chart saved to {output_path}")

        plt.close(fig1)

        # Create figure for units transferred by product
        if "product_id" in transfer_plan.columns:
            fig2, ax = plt.subplots(figsize=(10, 6))

            # Group by product_id and sum units
            product_units = (
                transfer_plan.groupby("product_id")["units"]
                .sum()
                .sort_values(ascending=False)
            )

            # If product names are available, use them
            if "product" in transfer_plan.columns:
                # Create mapping from ID to name
                product_name_map = transfer_plan.set_index("product_id")[
                    "product"
                ].to_dict()

                # Create named series
                product_units_named = pd.Series(
                    product_units.values,
                    index=[
                        product_name_map.get(idx, f"Product {idx}")
                        for idx in product_units.index
                    ],
                    name=product_units.name,
                )

                # Plot with names
                product_units_named.head(15).plot(kind="bar", ax=ax, color="#8884d8")
            else:
                # Plot with IDs
                product_units.head(15).plot(kind="bar", ax=ax, color="#8884d8")

            ax.set_xlabel("Product")
            ax.set_ylabel("Units Transferred")
            ax.set_title("Top 15 Products by Transfer Volume")

            # Rotate x labels for better readability if using long product names
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()

            # Save figure
            output_path = os.path.join(self.output_dir, "transfer_by_product.png")
            fig2.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Transfer by product chart saved to {output_path}")

            plt.close(fig2)

    def compare_algorithms(self, results_dict, save_png=True, show_plot=True):
        """
        Compare results from different optimization algorithms.

        Args:
            results_dict: Dictionary with algorithm names as keys and
                         impact DataFrames as values
            save_png: Whether to save the plots as PNG files
            show_plot: Whether to show the plots

        Returns:
            Matplotlib figure if show_plot is True
        """
        if not results_dict:
            print("No results to compare.")
            return None

        print("Generating algorithm comparison visualizations...")

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Prepare data for comparison
        algorithms = list(results_dict.keys())

        # Metrics for comparison
        cost_reduction = []
        transport_cost = []
        excess_reduction = []
        balanced_increase = []

        for alg, impact_df in results_dict.items():
            if (
                "Transfer Plan" in impact_df
                and "Total Transport Cost" in impact_df["Transfer Plan"]
            ):
                transport_cost.append(
                    impact_df["Transfer Plan"]["Total Transport Cost"]
                )
            else:
                transport_cost.append(0)

            if (
                "Improvement" in impact_df
                and "Reduction in Excess Value" in impact_df["Improvement"]
            ):
                cost_reduction.append(
                    impact_df["Improvement"]["Reduction in Excess Value"]
                )
            else:
                cost_reduction.append(0)

            if (
                "Improvement" in impact_df
                and "Reduction in Excess Items" in impact_df["Improvement"]
            ):
                excess_reduction.append(
                    impact_df["Improvement"]["Reduction in Excess Items"]
                )
            else:
                excess_reduction.append(0)

            if (
                "Improvement" in impact_df
                and "Increase in Balanced Items" in impact_df["Improvement"]
            ):
                balanced_increase.append(
                    impact_df["Improvement"]["Increase in Balanced Items"]
                )
            else:
                balanced_increase.append(0)

        # First plot: Cost metrics
        x = np.arange(len(algorithms))
        width = 0.35

        # Convert to millions for better readability
        cost_reduction_mil = [c / 1000000 for c in cost_reduction]
        transport_cost_mil = [c / 1000000 for c in transport_cost]
        net_saving_mil = [
            (r - t) / 1000000 for r, t in zip(cost_reduction, transport_cost)
        ]

        # Create stacked bar chart for costs
        ax1.bar(x, cost_reduction_mil, width, label="Cost Reduction")
        ax1.bar(
            x,
            [-t for t in transport_cost_mil],
            width,
            bottom=[0] * len(algorithms),
            label="Transport Cost",
            color="#FF9999",
        )

        # Add net saving bars
        for i, saving in enumerate(net_saving_mil):
            ax1.text(
                i,
                cost_reduction_mil[i] - transport_cost_mil[i] / 2,
                f"Net: {saving:.1f}M",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

        ax1.set_ylabel("Million VND")
        ax1.set_title("Cost Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.legend()

        # Add horizontal line at 0
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Second plot: Inventory improvement metrics
        width = 0.3
        x = np.arange(len(algorithms))

        ax2.bar(
            x - width / 2,
            excess_reduction,
            width,
            label="Excess Items Reduced",
            color="#FF8042",
        )
        ax2.bar(
            x + width / 2,
            balanced_increase,
            width,
            label="Balanced Items Increased",
            color="#00C49F",
        )

        ax2.set_ylabel("Number of Items")
        ax2.set_title("Inventory Improvement")
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms)
        ax2.legend()

        plt.tight_layout()

        if save_png:
            output_path = os.path.join(self.output_dir, "algorithm_comparison.png")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Algorithm comparison chart saved to {output_path}")

        # If not showing plot, close figure to free memory
        if not show_plot:
            plt.close(fig)
            return None

        return fig
