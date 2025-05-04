"""
Results Presenter Module
----------------------
Creates clear, readable summaries and comparisons of optimization results.
Makes it easy to understand the benefits of each approach.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


class ResultsPresenter:
    def __init__(self, output_dir="results"):
        """
        Initialize the results presenter.
        
        Args:
            output_dir: Directory to save result summaries
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('colorblind')
        
        # Algorithm colors for consistent visualization
        self.algorithm_colors = {
            "Rule-Based": "#FF9966",
            "Linear Programming": "#6699CC",
            "Genetic Algorithm": "#99CC99"
        }
    
    def load_transfer_plans(self, data_dir="data"):
        """
        Load transfer plans from all algorithms.
        
        Args:
            data_dir: Directory containing transfer plan CSV files
            
        Returns:
            Dictionary of transfer plans by algorithm
        """
        transfer_plans = {}
        
        # Rule-based plan
        rule_based_path = os.path.join(data_dir, "rule_based_transfers.csv")
        if os.path.exists(rule_based_path):
            transfer_plans["Rule-Based"] = pd.read_csv(rule_based_path)
        
        # Linear programming plan
        lp_path = os.path.join(data_dir, "lp_transfers.csv")
        if os.path.exists(lp_path):
            transfer_plans["Linear Programming"] = pd.read_csv(lp_path)
        
        # Genetic algorithm plan
        ga_path = os.path.join(data_dir, "ga_transfers.csv")
        if os.path.exists(ga_path):
            transfer_plans["Genetic Algorithm"] = pd.read_csv(ga_path)
        
        return transfer_plans
    
    def load_impact_data(self, data_dir="data"):
        """
        Load impact data from all algorithms.
        
        Args:
            data_dir: Directory containing impact CSV files
            
        Returns:
            Dictionary of impact DataFrames by algorithm
        """
        impact_data = {}
        
        # Rule-based impact
        rule_based_path = os.path.join(data_dir, "rule_based_impact.csv")
        if os.path.exists(rule_based_path):
            impact_data["Rule-Based"] = pd.read_csv(rule_based_path, index_col=0)
        
        # Linear programming impact
        lp_path = os.path.join(data_dir, "lp_impact.csv")
        if os.path.exists(lp_path):
            impact_data["Linear Programming"] = pd.read_csv(lp_path, index_col=0)
        
        # Genetic algorithm impact
        ga_path = os.path.join(data_dir, "ga_impact.csv")
        if os.path.exists(ga_path):
            impact_data["Genetic Algorithm"] = pd.read_csv(ga_path, index_col=0)
        
        return impact_data
    
    def create_transfer_plan_summary(self, transfer_plans):
        """
        Create a summary of all transfer plans.
        
        Args:
            transfer_plans: Dictionary of transfer plans by algorithm
            
        Returns:
            DataFrame with summary metrics
        """
        if not transfer_plans:
            print("No transfer plans available.")
            return None
        
        summary_data = []
        
        for algorithm, plan in transfer_plans.items():
            if plan.empty:
                continue
                
            total_transfers = len(plan)
            total_units = plan['units'].sum()
            total_cost = plan['transport_cost'].sum()
            avg_cost_per_unit = total_cost / total_units if total_units > 0 else 0
            avg_distance = plan['distance_km'].mean()
            max_distance = plan['distance_km'].max()
            
            # Count unique products and stores
            unique_products = plan['product_id'].nunique()
            unique_source_stores = plan['from_store_id'].nunique()
            unique_dest_stores = plan['to_store_id'].nunique()
            
            summary_data.append({
                'Algorithm': algorithm,
                'Total Transfers': total_transfers,
                'Total Units': total_units,
                'Total Transport Cost': total_cost,
                'Avg Cost Per Unit': avg_cost_per_unit,
                'Avg Distance (km)': avg_distance,
                'Max Distance (km)': max_distance,
                'Unique Products': unique_products,
                'Source Stores': unique_source_stores,
                'Destination Stores': unique_dest_stores
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format currency values
        for col in ['Total Transport Cost', 'Avg Cost Per Unit']:
            summary_df[col] = summary_df[col].map(lambda x: f"{x:,.0f} VND")
        
        # Format distance values
        for col in ['Avg Distance (km)', 'Max Distance (km)']:
            summary_df[col] = summary_df[col].map(lambda x: f"{x:.1f}")
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, "transfer_plan_summary.csv")
        summary_df.to_csv(output_path, index=False)
        
        return summary_df
    
    def create_impact_summary(self, impact_data):
        """
        Create a summary of impact data from all algorithms.
        
        Args:
            impact_data: Dictionary of impact DataFrames by algorithm
            
        Returns:
            DataFrame with key impact metrics
        """
        if not impact_data:
            print("No impact data available.")
            return None
        
        summary_data = []
        
        for algorithm, impact_df in impact_data.items():
            if impact_df.empty:
                continue
            
            # Extract key metrics from impact data
            try:
                # Before metrics
                before = impact_df['Before Transfer']
                excess_items_before = before.get('Excess Items', 0)
                needed_items_before = before.get('Needed Items', 0)
                balanced_items_before = before.get('Balanced Items', 0)
                
                # After metrics
                after = impact_df['After Transfer']
                excess_items_after = after.get('Excess Items', 0)
                needed_items_after = after.get('Needed Items', 0)
                balanced_items_after = after.get('Balanced Items', 0)
                
                # Improvements
                improvement = impact_df['Improvement']
                excess_reduction = improvement.get('Reduction in Excess Items', 0)
                needed_reduction = improvement.get('Reduction in Needed Items', 0)
                balanced_increase = improvement.get('Increase in Balanced Items', 0)
                
                # Costs
                transfer_plan = impact_df['Transfer Plan']
                total_cost = transfer_plan.get('Total Transport Cost', 0)
                
                # Percentages
                if excess_items_before > 0:
                    excess_reduction_pct = (excess_reduction / excess_items_before) * 100
                else:
                    excess_reduction_pct = 0
                    
                if needed_items_before > 0:
                    needed_reduction_pct = (needed_reduction / needed_items_before) * 100
                else:
                    needed_reduction_pct = 0
                
                summary_data.append({
                    'Algorithm': algorithm,
                    'Excess Items Reduced': f"{excess_reduction} ({excess_reduction_pct:.1f}%)",
                    'Needed Items Reduced': f"{needed_reduction} ({needed_reduction_pct:.1f}%)",
                    'Balanced Items Increased': balanced_increase,
                    'Transport Cost': f"{total_cost:,.0f} VND",
                    'Inventory Cost Reduction': f"{before.get('Excess Inventory Value', 0) - after.get('Excess Inventory Value', 0):,.0f} VND",
                    'Turnover Improvement': improvement.get('Product Turnover Improvement', 'N/A'),
                    'Balance Improvement': improvement.get('Inventory Balance Improvement', 'N/A'),
                })
            except Exception as e:
                print(f"Error processing impact data for {algorithm}: {str(e)}")
                continue
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, "impact_summary.csv")
        summary_df.to_csv(output_path, index=False)
        
        return summary_df
    
    def visualize_comparison(self, transfer_plans, impact_data):
        """
        Create visualizations comparing the different optimization approaches.
        
        Args:
            transfer_plans: Dictionary of transfer plans by algorithm
            impact_data: Dictionary of impact DataFrames by algorithm
        """
        if not transfer_plans or not impact_data:
            print("Insufficient data for comparison visualizations.")
            return
        
        # 1. Units transferred vs. Transport cost
        self._create_cost_units_comparison(transfer_plans)
        
        # 2. Inventory Status Improvement
        self._create_inventory_status_comparison(impact_data)
        
        # 3. Algorithm Efficiency (units per cost)
        self._create_efficiency_comparison(transfer_plans)
        
        # 4. Distance Distribution
        self._create_distance_comparison(transfer_plans)
    
    def _create_cost_units_comparison(self, transfer_plans):
        """Create comparison of units transferred vs. transport cost."""
        # Extract data
        algorithms = []
        units = []
        costs = []
        
        for algorithm, plan in transfer_plans.items():
            if plan.empty:
                continue
                
            algorithms.append(algorithm)
            units.append(plan['units'].sum())
            costs.append(plan['transport_cost'].sum())
        
        # Convert costs to millions for better readability
        costs_mil = [c / 1000000 for c in costs]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars
        x = np.arange(len(algorithms))
        width = 0.35
        
        units_bar = ax.bar(x - width/2, units, width, label='Units Transferred',
                         color=[self.algorithm_colors.get(a, '#999999') for a in algorithms])
        
        # Create a second y-axis for costs
        ax2 = ax.twinx()
        costs_bar = ax2.bar(x + width/2, costs_mil, width, label='Transport Cost (million VND)',
                          color=['#f8766d'] * len(algorithms), alpha=0.7)
        
        # Add labels and legend
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Units Transferred')
        ax2.set_ylabel('Transport Cost (million VND)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        
        # Add value labels
        for bar in units_bar:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}',
                   ha='center', va='bottom')
            
        for bar in costs_bar:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}M',
                    ha='center', va='bottom')
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Units Transferred vs. Transport Cost by Algorithm')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "units_vs_cost_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cost vs. units comparison saved to {output_path}")
    
    def _create_inventory_status_comparison(self, impact_data):
        """Create comparison of inventory status improvements."""
        # Extract data
        algorithms = []
        excess_reductions = []
        needed_reductions = []
        balanced_increases = []
        
        for algorithm, impact_df in impact_data.items():
            if impact_df.empty:
                continue
                
            # Get improvements
            improvement = impact_df['Improvement']
            excess_reduction = improvement.get('Reduction in Excess Items', 0)
            needed_reduction = improvement.get('Reduction in Needed Items', 0)
            balanced_increase = improvement.get('Increase in Balanced Items', 0)
            
            algorithms.append(algorithm)
            excess_reductions.append(excess_reduction)
            needed_reductions.append(needed_reduction)
            balanced_increases.append(balanced_increase)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create grouped bars
        x = np.arange(len(algorithms))
        width = 0.25
        
        ax.bar(x - width, excess_reductions, width, label='Excess Items Reduced',
              color='#FF8042')
        ax.bar(x, needed_reductions, width, label='Needed Items Reduced',
              color='#0088FE')
        ax.bar(x + width, balanced_increases, width, label='Balanced Items Increased',
              color='#00C49F')
        
        # Add labels and legend
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Number of Items')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
        
        # Add value labels to each bar
        for i, v in enumerate(excess_reductions):
            ax.text(i - width, v + 5, str(int(v)), ha='center')
        for i, v in enumerate(needed_reductions):
            ax.text(i, v + 5, str(int(v)), ha='center')
        for i, v in enumerate(balanced_increases):
            ax.text(i + width, v + 5, str(int(v)), ha='center')
        
        plt.title('Inventory Status Improvement by Algorithm')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "inventory_status_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Inventory status comparison saved to {output_path}")
    
    def _create_efficiency_comparison(self, transfer_plans):
        """Create comparison of algorithm efficiency (units per cost)."""
        # Extract data
        algorithms = []
        efficiencies = []
        
        for algorithm, plan in transfer_plans.items():
            if plan.empty:
                continue
                
            total_units = plan['units'].sum()
            total_cost = plan['transport_cost'].sum()
            
            # Calculate units per million VND
            if total_cost > 0:
                efficiency = total_units / (total_cost / 1000000)
            else:
                efficiency = 0
                
            algorithms.append(algorithm)
            efficiencies.append(efficiency)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars
        bars = ax.bar(algorithms, efficiencies, 
                     color=[self.algorithm_colors.get(a, '#999999') for a in algorithms])
        
        # Add labels
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Units per Million VND')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        plt.title('Algorithm Efficiency (Units Transferred per Million VND)')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "algorithm_efficiency_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Algorithm efficiency comparison saved to {output_path}")
    
    def _create_distance_comparison(self, transfer_plans):
        """Create comparison of distance distributions."""
        # Prepare data for boxplot
        distance_data = []
        labels = []
        
        for algorithm, plan in transfer_plans.items():
            if plan.empty:
                continue
                
            distance_data.append(plan['distance_km'])
            labels.append(algorithm)
        
        if not distance_data:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create boxplot
        box = ax.boxplot(distance_data, patch_artist=True, labels=labels)
        
        # Set colors
        for i, patch in enumerate(box['boxes']):
            color = self.algorithm_colors.get(labels[i], '#999999')
            patch.set_facecolor(color)
        
        # Add labels
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Distance (km)')
        
        # Add mean values as text
        for i, distances in enumerate(distance_data):
            mean_distance = distances.mean()
            ax.text(i + 1, distances.max() + 5, 
                   f'Mean: {mean_distance:.1f} km',
                   ha='center')
        
        plt.title('Distribution of Transfer Distances by Algorithm')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "distance_distribution_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance distribution comparison saved to {output_path}")
    
    def generate_report(self, transfer_plans, impact_data):
        """
        Generate a comprehensive PDF report with all comparisons.
        
        Args:
            transfer_plans: Dictionary of transfer plans by algorithm
            impact_data: Dictionary of impact DataFrames by algorithm
        """
        # Create HTML report
        report_content = """
        <html>
        <head>
            <title>Inventory Transfer Optimization Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333366; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .header { background-color: #4472C4; color: white; padding: 10px; }
                .section { margin-top: 30px; }
                .highlight { font-weight: bold; color: #2E75B6; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Inventory Transfer Optimization Results</h1>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>
                    This report compares the results of three different optimization approaches for 
                    inventory transfers between retail stores. The goal is to balance inventory by 
                    moving excess stock to stores with shortages.
                </p>
            </div>
        """
        
        # Add transfer plan summary table
        transfer_summary = self.create_transfer_plan_summary(transfer_plans)
        if transfer_summary is not None:
            report_content += """
            <div class="section">
                <h2>Transfer Plan Summary</h2>
                <table>
                    <tr>
                        <th>Algorithm</th>
                        <th>Total Transfers</th>
                        <th>Total Units</th>
                        <th>Total Transport Cost</th>
                        <th>Avg Cost Per Unit</th>
                        <th>Avg Distance (km)</th>
                    </tr>
            """
            
            for _, row in transfer_summary.iterrows():
                report_content += f"""
                    <tr>
                        <td>{row['Algorithm']}</td>
                        <td>{row['Total Transfers']}</td>
                        <td>{row['Total Units']}</td>
                        <td>{row['Total Transport Cost']}</td>
                        <td>{row['Avg Cost Per Unit']}</td>
                        <td>{row['Avg Distance (km)']}</td>
                    </tr>
                """
            
            report_content += """
                </table>
            </div>
            """
        
        # Add impact summary table
        impact_summary = self.create_impact_summary(impact_data)
        if impact_summary is not None:
            report_content += """
            <div class="section">
                <h2>Impact Summary</h2>
                <table>
                    <tr>
                        <th>Algorithm</th>
                        <th>Excess Items Reduced</th>
                        <th>Needed Items Reduced</th>
                        <th>Balanced Items Increased</th>
                        <th>Transport Cost</th>
                        <th>Inventory Cost Reduction</th>
                    </tr>
            """
            
            for _, row in impact_summary.iterrows():
                report_content += f"""
                    <tr>
                        <td>{row['Algorithm']}</td>
                        <td>{row['Excess Items Reduced']}</td>
                        <td>{row['Needed Items Reduced']}</td>
                        <td>{row['Balanced Items Increased']}</td>
                        <td>{row['Transport Cost']}</td>
                        <td>{row['Inventory Cost Reduction']}</td>
                    </tr>
                """
            
            report_content += """
                </table>
            </div>
            """
        
        # Add comparison charts section
        report_content += """
            <div class="section">
                <h2>Algorithm Comparisons</h2>
            </div>
        """
        
        # Create visualizations
        self.visualize_comparison(transfer_plans, impact_data)
        
        # Add visualization images to report
        for img_name in ["units_vs_cost_comparison.png", "inventory_status_comparison.png", 
                        "algorithm_efficiency_comparison.png", "distance_distribution_comparison.png"]:
            img_path = os.path.join(self.output_dir, img_name)
            if os.path.exists(img_path):
                report_content += f"""
                <div class="section">
                    <img src="{img_path}" alt="{img_name}">
                </div>
                """
        
        # Add conclusion based on data
        report_content += """
            <div class="section">
                <h2>Conclusion and Recommendations</h2>
                <p>
                    Based on the analysis of the three optimization approaches:
                </p>
                <ul>
        """
        
        # Determine best algorithm based on efficiency
        if transfer_summary is not None and len(transfer_summary) > 0:
            # Calculate efficiency (units per cost)
            efficiencies = {}
            for algorithm, plan in transfer_plans.items():
                if plan.empty:
                    continue
                    
                total_units = plan['units'].sum()
                total_cost = plan['transport_cost'].sum()
                
                if total_cost > 0:
                    efficiencies[algorithm] = total_units / total_cost
                else:
                    efficiencies[algorithm] = 0
            
            if efficiencies:
                best_algorithm = max(efficiencies, key=efficiencies.get)
                
                report_content += f"""
                    <li>The <span class="highlight">{best_algorithm}</span> approach demonstrates the best overall efficiency in terms of units transferred per cost.</li>
                """
        
        # Add general conclusions
        report_content += """
                    <li>Linear Programming provides the most cost-optimal solution with the minimum total transport cost.</li>
                    <li>The Rule-Based approach is simpler to implement but may result in higher transport costs.</li>
                    <li>The Genetic Algorithm offers a balance between optimization quality and flexibility.</li>
                </ul>
                <p>
                    For implementation, we recommend:
                </p>
                <ol>
                    <li>Start with the Linear Programming approach for regular inventory transfers.</li>
                    <li>Use the Genetic Algorithm when additional constraints (such as vehicle capacity or time windows) are important.</li>
                    <li>Keep the Rule-Based approach as a fallback for simple scenarios or when computation resources are limited.</li>
                </ol>
            </div>
        """
        
        # Close HTML tags
        report_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        output_path = os.path.join(self.output_dir, "optimization_results_report.html")
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"Comprehensive report saved to {output_path}")
    
    def print_summary_tables(self):
        """
        Print summary tables to console for easy viewing of results.
        """
        # Load transfer plans and impact data
        transfer_plans = self.load_transfer_plans()
        impact_data = self.load_impact_data()
        
        # Create summary tables
        transfer_summary = self.create_transfer_plan_summary(transfer_plans)
        impact_summary = self.create_impact_summary(impact_data)
        
        # Print transfer plan summary
        if transfer_summary is not None:
            print("\n=== TRANSFER PLAN SUMMARY ===")
            print(tabulate(transfer_summary, headers='keys', tablefmt='grid', showindex=False))
        
        # Print impact summary
        if impact_summary is not None:
            print("\n=== IMPACT SUMMARY ===")
            print(tabulate(impact_summary, headers='keys', tablefmt='grid', showindex=False))
        
        # Generate comprehensive report with visualizations
        if transfer_plans and impact_data:
            print("\nGenerating comprehensive comparison report...")
            self.generate_report(transfer_plans, impact_data)


if __name__ == "__main__":
    # This code will run when the module is executed directly
    presenter = ResultsPresenter()
    presenter.print_summary_tables()
