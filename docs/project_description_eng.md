### **Project Title:**

**Optimizing Inter-Store Inventory Transfers Using Sales Data and Geographic Distance**

---

### **1. Project Background and Objective**

**Background:**
A retail chain with 20 stores located in three main cities—Hanoi, Da Nang, and Ho Chi Minh City—is facing inventory imbalance issues. Some products sell slowly at certain stores while selling quickly at others. This results in overstock at some locations and shortages at others.

**Objective:**
Develop a system or algorithm to recommend inventory transfers between stores in order to:

- **Minimize unnecessary stockpiling**,
- **Maximize product turnover**, and
- **Optimize transportation costs between stores**.

---

### **2. Project Scope**

- **Input Data:**

  - Historical sales data for each product at each store.
  - Current inventory levels of each product per store.
  - Distance or transportation cost matrix between stores (based on geographic coordinates).

- **Expected Output:**

  - A system/algorithm that periodically (e.g., weekly) recommends how much inventory should be transferred between stores.

- **Two main modeling approaches:**

  1. **Rule-based system:** Using fixed thresholds and predefined rules.
  2. **Optimization techniques:** Using algorithms such as Genetic Algorithms or Transportation Problems.

---

### **3. Technical Details and Proposed Solution**

#### **A. Rule-Based Approach**

- Define optimal inventory thresholds for each product at each store based on past sales.
- Mark items as "excess" if inventory exceeds thresholds and sales are slow.
- Mark items as "needed" if inventory is low and sales are fast.
- Propose transfer plans from excess to needed locations, prioritizing shortest distances and feasible volumes.

#### **B. Optimization-Based Approach**

- **Input:**

  - Demand matrix (forecasted need) per product per store.
  - Supply matrix (available stock).
  - Transportation cost/distance matrix between stores.

- **Formulate the problem as:**

  - A **Transportation Problem** or **Linear Programming (LP)** task to find the optimal shipment plan minimizing total transfer cost while meeting demand.
  - **Genetic Algorithm (GA)** to explore near-optimal transfer plans under constraints (e.g., vehicle capacity, shipping frequency limits, etc.).

---

### **4. Implementation Steps**

1. **Data Collection**

   - Collect 12 months of historical sales per product/store.
   - Get inventory data over time.
   - Gather geographic coordinates or shipping rates between stores.

2. **Data Preprocessing & Analysis**

   - Cluster products based on sales velocity.
   - Estimate average daily demand per product per store.
   - Standardize units and align inventory formats.

3. **Model Development**

   - Build the rule-based model as a baseline.
   - Develop and tune optimization models (LP, GA).
   - Compare performance using KPIs like transportation cost, inventory reduction, and post-transfer sales.

4. **Pilot Testing**

   - Apply the model to a smaller store group.
   - Evaluate effectiveness and fine-tune parameters.

5. **Production Deployment & Dashboarding**

   - Create dashboards to report inventory, transfer suggestions, and KPIs.
   - Train users and integrate with existing systems (e.g., ERP or WMS).

---

### **5. Expected Outcomes**

- Reduce overstock inventory by at least 20–30%.
- Accelerate inventory turnover across regions.
- Optimize inter-store transfer costs.
- Provide a scalable system for periodic operation and future store expansion.
