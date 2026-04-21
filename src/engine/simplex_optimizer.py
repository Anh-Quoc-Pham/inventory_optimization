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

    def optimize(self, excess_inventory, needed_inventory):
        start_time = time.time()
        self.logger_system.log_execution_start("simplex_optimization", {"algorithm": "Simplex (Linear Programming)"})

        if excess_inventory.empty or needed_inventory.empty:
            return pd.DataFrame()

        all_transfers = []

        # Chúng ta tối ưu hóa theo từng sản phẩm để giảm kích thước ma trận
        products = set(excess_inventory['product_id']) & set(needed_inventory['product_id'])

        for product_id in products:
            sources = excess_inventory[excess_inventory['product_id'] == product_id].copy()
            sinks = needed_inventory[needed_inventory['product_id'] == product_id].copy()

            n_sources = len(sources)
            n_sinks = len(sinks)

            # 1. Hàm mục tiêu (Cost vector c)
            # x_ij là lượng chuyển từ source i đến sink j
            costs = []
            for _, src in sources.iterrows():
                for _, snk in sinks.iterrows():
                    cost = self.transport_cost_matrix.loc[src['store_id'], snk['store_id']]
                    costs.append(cost if not np.isnan(cost) else 1e9)

            # 2. Ràng buộc bất đẳng thức (A_ub * x <= b_ub)
            A_ub = []
            b_ub = []

            # Ràng buộc nguồn: Tổng xuất <= excess_units
            for i in range(n_sources):
                row = [0] * (n_sources * n_sinks)
                for j in range(n_sinks):
                    row[i * n_sinks + j] = 1
                A_ub.append(row)
                b_ub.append(sources.iloc[i]['excess_units'])

            # Ràng buộc đích: Tổng nhập <= needed_units (Dùng dấu âm để biến thành <=)
            # Hoặc dùng A_eq nếu bắt buộc phải thỏa mãn 100% nhu cầu
            for j in range(n_sinks):
                row = [0] * (n_sources * n_sinks)
                for i in range(n_sources):
                    row[i * n_sinks + j] = 1
                # Ở đây dùng ràng buộc <= để linh hoạt nếu excess không đủ bù need
                A_ub.append([-val for val in row])
                b_ub.append(-min(sinks.iloc[j]['needed_units'], sources['excess_units'].sum()))

            # 3. Giải bài toán bằng linprog (phương pháp 'highs' là hiện đại và nhanh nhất)
            res = linprog(costs, A_ub=A_ub, b_ub=b_ub, method='highs')

            if res.success:
                x = res.x.reshape((n_sources, n_sinks))
                for i in range(n_sources):
                    for j in range(n_sinks):
                        units = round(x[i, j])
                        if units > 0:
                            src_id = sources.iloc[i]['store_id']
                            snk_id = sinks.iloc[j]['store_id']
                            all_transfers.append({
                                "from_store_id": src_id,
                                "to_store_id": snk_id,
                                "product_id": product_id,
                                "units": units,
                                "distance_km": self.distance_matrix.loc[src_id, snk_id],
                                "transport_cost": self.transport_cost_matrix.loc[src_id, snk_id] * units
                            })

        self.transfer_plan = pd.DataFrame(all_transfers)
        execution_time = time.time() - start_time
        self.logger_system.log_execution_end("simplex_optimization", execution_time,
                                             {"transfers": len(self.transfer_plan)})
        return self.transfer_plan

