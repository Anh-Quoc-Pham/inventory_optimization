import sys
from pathlib import Path

if __name__ == "__main__" or "src.engine" not in sys.modules:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import time
from src.utils.logger import get_optimization_logger


class SimplexOptimizer:
    def __init__(self, distance_matrix=None, transport_cost_matrix=None):
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.transfer_plan = None
        self.logger_system = get_optimization_logger()

    def load_matrices(self, distance_path, cost_path):
        """
        Khởi tạo và nạp dữ liệu ma trận (Được chuẩn hóa theo cấu trúc của file bài giảng)
        """
        # Load matrices with store IDs as integers
        self.distance_matrix = pd.read_csv(distance_path, index_col=0)
        self.transport_cost_matrix = pd.read_csv(cost_path, index_col=0)

        # Ensure indices and columns are integers
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)
        self.transport_cost_matrix.index = self.transport_cost_matrix.index.astype(int)
        self.transport_cost_matrix.columns = self.transport_cost_matrix.columns.astype(int)

    def optimize(self, excess_inventory, needed_inventory):
        start_time = time.time()
        self.logger_system.log_execution_start("simplex_optimization", {"algorithm": "Simplex (Linear Programming)"})

        # Xử lý chuẩn khi inventory rỗng
        if excess_inventory.empty or needed_inventory.empty:
            self.logger_system.log_progress("simplex_optimization",
                                            "No excess or needed inventory found. No transfers needed.")
            self.transfer_plan = pd.DataFrame()
            execution_time = time.time() - start_time
            self.logger_system.log_execution_end(
                "simplex_optimization", execution_time,
                {"transfers_generated": 0, "reason": "No excess or needed inventory"}
            )
            return self.transfer_plan

        all_transfers = []

        # Tối ưu hóa theo từng sản phẩm
        products = set(excess_inventory['product_id']) & set(needed_inventory['product_id'])

        for product_id in products:
            sources = excess_inventory[excess_inventory['product_id'] == product_id].copy()
            sinks = needed_inventory[needed_inventory['product_id'] == product_id].copy()

            n_sources = len(sources)
            n_sinks = len(sinks)

            if n_sources == 0 or n_sinks == 0:
                continue

            # 1. Hàm mục tiêu (Cost vector c)
            costs = []
            M = 1e6  # Trọng số "thưởng" cực lớn để "ép" mô hình chuyển lượng hàng hóa tối đa có thể

            for _, src in sources.iterrows():
                for _, snk in sinks.iterrows():
                    src_id = src['store_id']
                    snk_id = snk['store_id']

                    # Ràng buộc: Không tự chuyển (x[i, i, p] = 0)
                    if src_id == snk_id:
                        costs.append(1e9)
                        continue

                    cost = self.transport_cost_matrix.loc[src_id, snk_id]
                    if np.isnan(cost) or cost <= 0:
                        dist = self.distance_matrix.loc[src_id, snk_id]
                        cost = dist * 1000 if not np.isnan(dist) else 1e9

                    # Trừ đi điểm thưởng M để máy tính đánh giá việc "chuyển hàng" mang lại lợi ích tốt hơn là "không làm gì" (kết quả 0)
                    costs.append(cost - M)

            # 2. Ràng buộc bất đẳng thức (A_ub * x <= b_ub)
            A_ub = []
            b_ub = []

            # Ràng buộc cung cấp (Supply): Tổng xuất <= excess_units
            for i in range(n_sources):
                row = [0] * (n_sources * n_sinks)
                for j in range(n_sinks):
                    row[i * n_sinks + j] = 1
                A_ub.append(row)
                b_ub.append(sources.iloc[i]['excess_units'])

            # Ràng buộc nhu cầu (Demand): Tổng nhập <= needed_units
            for j in range(n_sinks):
                row = [0] * (n_sources * n_sinks)
                for i in range(n_sources):
                    row[i * n_sinks + j] = 1
                A_ub.append(row)
                b_ub.append(sinks.iloc[j]['needed_units'])

            bounds = [(0, None) for _ in range(n_sources * n_sinks)]

            # 3. Giải bài toán bằng linprog
            res = linprog(costs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if res.success:
                x = res.x.reshape((n_sources, n_sinks))
                for i in range(n_sources):
                    for j in range(n_sinks):
                        units = round(x[i, j])
                        if units > 0:
                            src_id = sources.iloc[i]['store_id']
                            snk_id = sinks.iloc[j]['store_id']

                            dist = self.distance_matrix.loc[src_id, snk_id]
                            t_cost = self.transport_cost_matrix.loc[src_id, snk_id]
                            if np.isnan(t_cost) or t_cost <= 0:
                                t_cost = dist * 1000

                            all_transfers.append({
                                "from_store_id": src_id,
                                "to_store_id": snk_id,
                                "product_id": product_id,
                                "units": units,
                                "distance_km": dist,
                                "transport_cost": t_cost * units
                            })

        self.transfer_plan = pd.DataFrame(all_transfers)
        execution_time = time.time() - start_time

        results = {
            "transfers_generated": len(self.transfer_plan),
        }
        self.logger_system.log_execution_end("simplex_optimization", execution_time, results)

        return self.transfer_plan


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Tự động xác định thư mục gốc của dự án
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"

    # Tạo thư mục results nếu chưa có
    results_dir.mkdir(exist_ok=True)

    print("Testing Simplex Optimizer (Linear Programming)")
    print("=" * 70)

    # 1. Kiểm tra các file dữ liệu đầu vào cần thiết
    required_files = [
        "sales_data.csv",
        "inventory_data.csv",
        "distance_matrix.csv",
        "transport_cost_matrix.csv",
    ]

    print("Checking required data files...")
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"Required file {file} not found. Please run data generator first.")
            exit(1)
        else:
            print(f"[OK] {file}")

    # 2. Khởi tạo Analyzer và Phân tích tồn kho (Tìm hàng thừa/thiếu)
    try:
        from src.engine.analyzer import InventoryAnalyzer

        print("\nLoading and analyzing data...")
        analyzer = InventoryAnalyzer()

        analyzer.load_data(
            sales_path=str(data_dir / "sales_data.csv"),
            inventory_path=str(data_dir / "inventory_data.csv"),
            stores_path=str(data_dir / "stores.csv") if (data_dir / "stores.csv").exists() else None,
            products_path=str(data_dir / "products.csv") if (data_dir / "products.csv").exists() else None,
        )

        analyzer.analyze_sales_data()
        excess_df, needed_df = analyzer.identify_inventory_imbalances()

        print(f"Analysis complete:")
        print(f"   • Products with excess: {len(excess_df)}")
        print(f"   • Products needed: {len(needed_df)}")
        print(f"   • Total excess units: {excess_df['excess_units'].sum():,}")
        print(f"   • Total needed units: {needed_df['needed_units'].sum():,}")

    except ImportError:
        print("Error: Could not import InventoryAnalyzer. Ensure src/engine/analyzer.py exists.")
        exit(1)

    # 3. Chạy thuật toán Simplex
    print("\nCreating Simplex optimizer...")
    optimizer = SimplexOptimizer()

    # Nạp ma trận
    optimizer.load_matrices(
        distance_path=str(data_dir / "distance_matrix.csv"),
        cost_path=str(data_dir / "transport_cost_matrix.csv"),
    )

    print("Running Simplex optimization...")
    transfer_plan = optimizer.optimize(excess_df, needed_df)

    # 4. Đánh giá và lưu kết quả
    if transfer_plan is not None and not transfer_plan.empty:
        output_path = str(results_dir / "simplex_transfers_test.csv")
        transfer_plan.to_csv(output_path, index=False)

        print(f"\nResults saved to: {output_path}")
        print("Simplex Algorithm test completed successfully!")

        total_units = transfer_plan['units'].sum()
        total_cost = transfer_plan['transport_cost'].sum()
        avg_cost = total_cost / total_units if total_units > 0 else 0

        print(f"\nSimplex Results Summary:")
        print(f"   • Total transfers: {len(transfer_plan)}")
        print(f"   • Total units to transfer: {total_units}")
        print(f"   • Total transport cost: {total_cost:,.0f} VND")
        print(f"   • Average cost per unit: {avg_cost:,.0f} VND")

        print(f"\nSample transfers (top 3):")
        cols_to_show = ["from_store_id", "to_store_id", "product_id", "units", "transport_cost"]
        print(transfer_plan[cols_to_show].head(3).to_string(index=False))
    else:
        print("\nNo transfers were generated. This might indicate:")
        print("   • No beneficial transfers exist")
        print("   • Excess and Needed inventory do not match")