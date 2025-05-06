import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from results_visualizer import ResultsVisualizer


def main():
    # Set up the page configuration
    st.set_page_config(
        page_title="Inventory Transfer Optimization Results",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    # Title and description
    st.title("Inventory Transfer Optimization Results")
    st.write(
        "Comparing Rule-Based, Linear Programming, and Genetic Algorithm approaches"
    )

    # Load results
    visualizer = ResultsVisualizer()
    data = visualizer.load_data()

    if data is None:
        st.error("Failed to load data. Please ensure all required files are present.")
        return

    # Sidebar for visualization selection
    st.sidebar.header("Visualization Options")
    viz_options = [
        "Inventory Status Distribution",
        "City-wise Inventory Distribution",
        "Product Category Inventory Distribution",
        "Algorithm Transfer Volume Comparison",
        "Algorithm Transfer Cost Comparison",
    ]
    selected_viz = st.sidebar.selectbox("Select Visualization", viz_options)

    # Create figures based on selection
    plt.close("all")  # Close any existing plots
    if selected_viz == "Inventory Status Distribution":
        fig = visualizer.create_store_inventory_histogram(data["analysis_df"])
        st.pyplot(fig)

    elif selected_viz == "City-wise Inventory Distribution":
        fig = visualizer.create_city_inventory_histogram(data["analysis_df"])
        st.pyplot(fig)

    elif selected_viz == "Product Category Inventory Distribution":
        fig = visualizer.create_product_category_histogram(data["analysis_df"])
        st.pyplot(fig)

    elif selected_viz == "Algorithm Transfer Volume Comparison":
        algorithms = ["Rule-Based", "Linear Programming", "Genetic Algorithm"]
        fig = visualizer.create_algorithm_comparison_chart(
            data["transfer_plans"], algorithms
        )
        st.pyplot(fig)

    elif selected_viz == "Algorithm Transfer Cost Comparison":
        algorithms = ["Rule-Based", "Linear Programming", "Genetic Algorithm"]
        fig = visualizer.create_transfer_cost_comparison(
            data["transfer_plans"], algorithms
        )
        st.pyplot(fig)

    # Detailed comparison section
    st.header("Detailed Algorithm Comparison")

    # Create columns for algorithm comparison
    col1, col2, col3 = st.columns(3)

    algorithms = ["Rule-Based", "Linear Programming", "Genetic Algorithm"]

    for i, algo in enumerate([col1, col2, col3], 0):
        with algo:
            st.subheader(algorithms[i])
            transfer_plan = data["transfer_plans"].get(algorithms[i], pd.DataFrame())

            if not transfer_plan.empty:
                st.write(f"Total Transfers: {len(transfer_plan)}")
                st.write(f"Total Units Transferred: {transfer_plan['units'].sum():,}")
                st.write(
                    f"Total Transport Cost: {transfer_plan['transport_cost'].sum():,.0f} VND"
                )
            else:
                st.write("No transfer plan available")


if __name__ == "__main__":
    main()
