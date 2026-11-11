# PF–MPC Swarm Simulation

Mã nguồn mô phỏng dùng trong báo cáo nghiên cứu về điều khiển chuyển động bầy UAV bằng trường thế năng (Potential Field) và điều khiển dự báo mô hình (Model Predictive Control).

## Cấu trúc

- `case1_offset_gate/`: bầy UAV đi qua khe hẹp lệch tâm.
- `case2_right_of_way/`: hai tác tử gặp nhau tại nút thắt hẹp.
- `case3_dynamic_feasibility/`: tránh vật cản lớn trong điều kiện giới hạn gia tốc.

Mỗi thư mục thí nghiệm gồm:

- `config.py`: cấu hình môi trường và tham số mô phỏng.
- `controllers.py`: bộ điều khiển Potential Field và MPC.
- `simulate.py`: chạy mô phỏng và lưu dữ liệu.
- `plot_results.py`: sinh hình, bảng số liệu và video so sánh.
- `results/`: kết quả tham chiếu đã dùng trong báo cáo.

Riêng `case2_right_of_way/` có thêm `single_agent_check.py` để kiểm tra rằng nút thắt có thể đi qua khi chỉ có một tác tử, qua đó phân biệt lỗi hình học với hiện tượng bế tắc do tương tác đa tác tử.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy mô phỏng

Ví dụ với môi trường 1:

```bash
cd case1_offset_gate
python3 simulate.py
python3 plot_results.py
```

Thực hiện tương tự cho `case2_right_of_way` và `case3_dynamic_feasibility`.

## Ghi chú

Mã nguồn này được sử dụng để tái lập các thí nghiệm và hình minh họa trong báo cáo. Các tham số từng trường hợp được lưu trực tiếp trong `config.py` của từng thư mục.
