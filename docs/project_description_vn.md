# Miêu tả tiếng Việt

### **Tên dự án:**

Tối ưu hoá vận chuyển hàng giữa các cửa hàng dựa trên dữ liệu bán hàng và khoảng cách địa lý

---

### **1. Bối cảnh và Mục tiêu dự án**

**Bối cảnh:**
Một chuỗi bán lẻ với 20 cửa hàng phân bố tại ba khu vực chính: Hà Nội, Đà Nẵng và TP. Hồ Chí Minh đang gặp phải vấn đề liên quan đến hàng tồn kho không đều giữa các cửa hàng. Một số mặt hàng bán chậm tại cửa hàng này lại có thể bán chạy tại cửa hàng khác, dẫn đến hiện tượng tồn kho ở nơi này và thiếu hàng ở nơi khác.

**Mục tiêu:**
Xây dựng một hệ thống/thuật toán đề xuất luân chuyển hàng giữa các cửa hàng để:

- **Tối thiểu hóa hàng tồn kho không cần thiết**,
- **Tối đa hóa khả năng tiêu thụ hàng hóa**,
- **Tối ưu hóa chi phí vận chuyển giữa các cửa hàng**.

---

### **2. Phạm vi dự án**

- Dữ liệu đầu vào:

  - Lịch sử bán hàng của từng cửa hàng theo từng mặt hàng.
  - Số lượng tồn kho hiện tại của từng mặt hàng ở mỗi cửa hàng.
  - Khoảng cách hoặc chi phí vận chuyển giữa các cửa hàng (có thể dựa trên tọa độ địa lý).

- Sản phẩm đầu ra:

  - Một hệ thống/thuật toán có khả năng đề xuất lượng hàng nên luân chuyển giữa các cửa hàng theo định kỳ (ví dụ: hàng tuần).

- Hai hướng tiếp cận chính sẽ được triển khai và so sánh:

  1. **Rule-based (hệ thống luật cứng):** Dựa vào các ngưỡng cố định như mức tồn kho tối thiểu/tối đa, tỷ lệ bán trung bình,...
  2. **Phương pháp tối ưu hóa (metaheuristics):** Áp dụng các thuật toán như di truyền (genetic algorithm), bài toán vận tải (transportation problem), hoặc phương pháp tuyến tính.

---

### **3. Chi tiết kỹ thuật và giải pháp đề xuất**

#### **A. Rule-based Approach**

- Xác định ngưỡng tồn kho tối ưu cho từng cửa hàng/mặt hàng dựa trên dữ liệu quá khứ.
- Nếu tồn kho vượt ngưỡng + bán chậm → đánh dấu là dư thừa.
- Nếu tồn kho thấp hơn ngưỡng + bán nhanh → đánh dấu là thiếu.
- Dựa trên bảng khoảng cách giữa các cửa hàng, đề xuất luân chuyển từ nơi dư thừa đến nơi thiếu, ưu tiên khoảng cách ngắn nhất và số lượng hợp lý.

#### **B. Optimization-based Approach**

- **Input:**

  - Ma trận nhu cầu (demand) tại mỗi cửa hàng cho từng mặt hàng.
  - Ma trận nguồn cung (supply).
  - Ma trận chi phí vận chuyển giữa các cửa hàng.

- **Bài toán được mô hình hoá như:**

  - **Bài toán vận tải (Transportation Problem)** hoặc **Linear Programming (LP)** để tìm luồng hàng hóa tối ưu sao cho tổng chi phí vận chuyển thấp nhất mà vẫn đáp ứng được nhu cầu.
  - **Thuật toán di truyền (Genetic Algorithm)** để tìm phương án chuyển hàng với chi phí tối ưu trong điều kiện ràng buộc (ví dụ: giới hạn trọng tải, giới hạn số lần vận chuyển,...).

---

### **4. Các bước triển khai**

1. **Thu thập dữ liệu**

   - Bán hàng 12 tháng gần nhất theo từng cửa hàng/mặt hàng.
   - Dữ liệu tồn kho theo thời gian.
   - Tọa độ địa lý/cước phí vận chuyển giữa các cửa hàng.

2. **Tiền xử lý và phân tích dữ liệu**

   - Nhóm mặt hàng theo tốc độ tiêu thụ.
   - Ước lượng nhu cầu trung bình/ngày của từng mặt hàng tại mỗi cửa hàng.
   - Chuẩn hóa đơn vị đo và đồng bộ thông tin.

3. **Phát triển mô hình**

   - Xây dựng mô hình rule-based để làm baseline.
   - Phát triển và tinh chỉnh thuật toán tối ưu (LP, GA).
   - So sánh hiệu quả dựa trên KPI như: chi phí vận chuyển, tồn kho giảm, hàng bán được sau điều chuyển.

4. **Triển khai thử nghiệm (pilot)**

   - Áp dụng mô hình trên một nhóm nhỏ cửa hàng.
   - Đánh giá hiệu quả và điều chỉnh tham số.

5. **Triển khai thực tế và xây dựng dashboard**

   - Thiết kế dashboard báo cáo tồn kho, đề xuất vận chuyển và KPI.
   - Đào tạo người dùng và tích hợp với hệ thống hiện có (ERP/WMS nếu có).

---

### **5. Kết quả kỳ vọng**

- Giảm ít nhất 20-30% lượng hàng tồn kho không cần thiết.
- Tăng tốc độ luân chuyển hàng giữa các khu vực.
- Tối ưu chi phí vận chuyển hàng nội bộ.
- Hệ thống có thể vận hành định kỳ và mở rộng cho các khu vực/cửa hàng mới.
