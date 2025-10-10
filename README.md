# Inventory Transfer Optimization System

A data-driven solution for optimizing inventory transfers between retail stores, minimizing unnecessary stockpiling, maximizing product turnover, and optimizing transportation costs.

## Project Overview

This system helps a retail chain with multiple stores to efficiently balance inventory by recommending optimal product transfers between locations. It uses historical sales data and geographical distances to minimize overstock, reduce shortages, and optimize transport costs.

## Key Features

- **Smart Inventory Analysis**: Automatically identifies excess and needed inventory using sales patterns
- **Multiple Optimization Algorithms**:
  - Rule-based: Quick, simple heuristic approach
  - Genetic algorithm: Advanced optimization for complex scenarios
- **Transport Cost Optimization**: Minimizes shipping costs by considering distances
- **Simplified Results**: Focus on actionable insights with 4 essential output files

## Architecture

The system is organized into modular components:

```
src/
│
├── data_generator/  # Data generation components
│   ├── __init__.py
│   ├── store_generator.py       # Generates store data
│   ├── product_generator.py     # Generates product data
│   ├── sales_generator.py       # Generates sales data
│   ├── inventory_generator.py   # Generates inventory data
│   ├── distance_calculator.py   # Calculates distances and costs
│   └── main.py                  # Coordinates data generation
│
├── engine/          # Optimization engines
│   ├── __init__.py
│   ├── data_model.py            # Core data model classes
│   ├── analyzer.py              # Analyzes inventory status
│   ├── rule_based.py            # Rule-based optimization
│   ├── genetic_algorithm.py     # Genetic algorithm optimization
│   └── visualizer.py            # Visualization functions
│
└── main.py          # Main application entry point
```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/inventory-transfer-optimization.git
   cd inventory-transfer-optimization
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Generate Data & Run Optimization (One Command)

```bash
python src/main.py --generate-data --all
```

This creates sample data and runs optimization with both algorithms.

### 2. Check Your Results

After running, you'll find **4 essential files** in the `results` folder:

- 📋 `EXECUTIVE_SUMMARY.txt` - Your main report
- 📈 `INVENTORY_OVERVIEW.png` - Visual dashboard
- 📝 `TRANSFER_PLAN.csv` - Actionable transfer list
- 📄 `TRANSFER_SUMMARY.txt` - Quick implementation guide

### 3. Clean Up (Optional)

```bash
python cleanup_results.py
```

Removes complex/unnecessary files, keeps only the 4 essential outputs.

## 📖 Detailed Usage

### Individual Algorithm Testing

```bash
# Rule-based only (fast)
python src/main.py --rule-based

# Genetic algorithm only (more thorough)
python src/main.py --ga
```

### Command Line Arguments

- `--data-dir`: Directory for input data (default: "data")
- `--results-dir`: Directory for optimization results (default: "results")
- `--vis-dir`: Directory for visualizations (default: "visualizations")
- `--seed`: Random seed for reproducibility (default: 42)
- `--min-days`: Minimum days of inventory threshold (default: 7)
- `--max-days`: Maximum days of inventory threshold (default: 30)
- `--ga-population`: GA population size (default: 100)
- `--ga-generations`: GA number of generations (default: 50)

## Integration with Real Systems

To use the system with real data, replace the data generation step with data import from your existing systems. The required files are:

- `stores.csv`: Store information with locations
- `products.csv`: Product information
- `sales_data.csv`: Historical sales records
- `inventory_data.csv`: Current inventory levels
- `distance_matrix.csv`: Distances between stores
- `transport_cost_matrix.csv`: Transport costs between stores

## Output and Visualizations

The system generates several outputs:

1. **Analysis Results**:

   - Inventory status analysis
   - Excess and needed inventory identification

2. **Transfer Recommendations**:

   - Detailed transfer plans from each algorithm
   - Impact analysis showing expected improvements

3. **Visualizations**:
   - Interactive maps showing inventory status
   - Transfer flow visualizations
   - Comparative charts for optimization algorithms

All outputs are saved to the specified results and visualization directories.

## Performance Comparison

The two optimization approaches offer different trade-offs:

- **Rule-based**: Fastest execution, intuitive results, good for simple scenarios
- **Genetic Algorithm**: Advanced optimization for complex scenarios with multiple constraints

## License

This project is licensed under the MIT License - see the LICENSE file for details.
