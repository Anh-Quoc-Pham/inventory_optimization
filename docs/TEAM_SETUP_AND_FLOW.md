# Team Setup and Project Flow Guide

Tai lieu nay giup team:

- Setup moi truong nhanh tren Windows/macOS/Linux.
- Chay du an theo cac mode can thiet.
- Hieu flow xu ly du lieu va luong toi uu trong he thong.

## 1) Prerequisites

- Python 3.9+
- Git
- Khuyen nghi: Conda (de dong bo package nhanh)

## 2) Setup Moi Truong

### Cach A - Conda (khuyen nghi)

```bash
git clone <repo-url>
cd inventory_optimization
conda env create -f environment.yml
conda activate inventory_optimization
```

### Cach B - venv + pip

```bash
git clone <repo-url>
cd inventory_optimization
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Kiem tra nhanh sau khi setup

```bash
python --version
python src/main.py --help
```

Neu lenh help chay duoc thi moi truong da san sang.

## 4) Cac lenh chay chinh

### 4.1 Generate data

```bash
python src/main.py --generate-data
```

### 4.2 Chay Rule-Based

```bash
python src/main.py --rule-based
```

### 4.3 Chay Genetic Algorithm

```bash
python src/main.py --ga
```

### 4.4 Chay Primal Simplex

```bash
python src/main.py --simplex
```

### 4.5 Chay toan bo baseline (Rule-Based + GA + Simplex)

```bash
python src/main.py --all
```

### 4.6 Chay benchmark Chuong 3 (artifact phuc vu report)

```bash
python scripts/run_chapter3_benchmark.py
```

### 4.7 Mo notebook phan tich full flow

```bash
jupyter notebook notebooks/01_full_process_step_by_step.ipynb
```

## 5) Flow du an (from data to artifact)

```text
Data Inputs (sales, inventory, stores, products, matrices)
        |
        v
InventoryAnalyzer
  - analyze_sales_data()
  - identify_inventory_imbalances()
        |
        v
Transportation Instance Builder (theo product)
        |
        v
Optimizers
  - RuleBasedOptimizer
  - GeneticAlgorithmOptimizer
  - PrimalSimplexSolver (LP transport)
        |
        v
Impact Evaluation
  - evaluate_plan_impact()
        |
        v
ResultsManager + Visualizations
  - result_summary.txt
  - best_transfer_plan.csv
  - chapter3 benchmark artifacts
```

## 6) Cac output quan trong

### Ket qua tong quat

- results/result_summary.txt
- results/best_transfer_plan.csv
- results/inventory_analysis.csv
- results/rule_based_impact.csv
- results/ga_impact.csv
- results/simplex_impact.csv

### Benchmark Chuong 3

- results/chapter3_benchmark/chapter3_summary.csv
- results/chapter3_benchmark/chapter3_algorithm_comparison.csv
- results/chapter3_benchmark/chapter3_best_transfer_plan.csv
- results/chapter3_benchmark/chapter3_metadata.json
- visualizations/chapter3_benchmark/network_overview.png
- visualizations/chapter3_benchmark/hub_analysis.png
- visualizations/chapter3_benchmark/flow_intensity_map.png
- visualizations/chapter3_benchmark/chapter3_cost_heatmap_product_1.png

## 7) Team workflow de lam viec on dinh

1. Luon pull code moi nhat truoc khi chay benchmark.
2. Neu thay doi config, ghi ro trong PR mo ta thong so da doi.
3. Truoc khi tao PR:
   - Chay python src/main.py --simplex
   - Chay python scripts/run_chapter3_benchmark.py
   - Kiem tra chapter3_algorithm_comparison.csv
4. Neu notebook co update ket qua, commit kem theo output artifact lien quan.
5. Giu convention ten file artifact trong chapter3_benchmark de doi chieu report.

## 8) Troubleshooting nhanh

### Loi ModuleNotFoundError: src

- Dam bao dang o dung thu muc goc repo truoc khi chay lenh.
- Uu tien chay qua script da bootstrap path: scripts/run_chapter3_benchmark.py

### GA chay rat lau

- Giam tham so khi debug:
  - --ga-population
  - --ga-generations

### Khong co du lieu de toi uu

- Chay lai generate data:

```bash
python src/main.py --generate-data
```

### Muon reset ket qua benchmark

- Xoa folder output benchmark roi chay lai:

```bash
# Windows PowerShell
Remove-Item -Recurse -Force results\chapter3_benchmark
Remove-Item -Recurse -Force visualizations\chapter3_benchmark
python scripts/run_chapter3_benchmark.py
```

---

Neu can onboarding nhanh cho thanh vien moi, doc file nay truoc, sau do mo docs/logic.md de hieu logic thuat toan va benchmark chapter 3.
