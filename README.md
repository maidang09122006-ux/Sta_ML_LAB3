# Phân loại ảnh X-Quang Viêm Phổi bằng Custom SVM (Numpy)

Bài tập thực hành triển khai thuật toán Soft-margin SVM từ con số 0 sử dụng Numpy và Stochastic Gradient Descent (SGD). Đồng thời so sánh hiệu suất với mô hình LinearSVC của thư viện scikit-learn.

## Cấu trúc thư mục code
* `LAB3_SVM.py`: Module chứa Class SVM tự xây dựng (tính Hinge Loss, cập nhật trọng số bằng SGD).
* `LAB3_main.py`: Module tiền xử lý dữ liệu (đọc ảnh, chuyển Grayscale, resize 128x128, chuẩn hóa Z-score chống Data Leakage) và lưu ra file `.npz`.
* `LAB3_24520253.py`: File chạy chính. Tải dữ liệu đã xử lý, huấn luyện cả 2 mô hình, in bảng so sánh (Precision, Recall, F1) và vẽ biểu đồ.

## Hướng dẫn chạy code
**Bước 1:** Đảm bảo đặt thư mục dataset `chest_xray` nằm cùng cấp với các file code.
**Bước 2:** Chạy file xử lý dữ liệu (chỉ cần chạy 1 lần) để tạo ra file ma trận `processed_data.npz`:
```bash
python LAB3_main.py
