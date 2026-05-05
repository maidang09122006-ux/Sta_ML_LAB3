import numpy as np
import matplotlib.pyplot as plt
from LAB3_SVM import SVM 
from sklearn.svm import LinearSVC 
from sklearn.metrics import precision_score, recall_score, f1_score

print("Đang load dữ liệu từ file nén...")
data = np.load("processed_data.npz")

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print(f"✅ Đã load xong! X_train shape: {X_train.shape}")

# ===================== ASSIGNMENT 1: CUSTOM SVM =====================
print("\n" + "="*60)
print("ASSIGNMENT 1: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH CUSTOM SVM")
print("="*60)

model = SVM(C=1, lr=0.01, n_epochs=1000) 
model.fit(X_train, y_train)

scores = model.get_metrics(X_test, y_test)

# ===================== ASSIGNMENT 2: SKLEARN SVM =====================
print("\n" + "="*60)
print("ASSIGNMENT 2: SVM TỪ THƯ VIỆN SKLEARN")
print("="*60)

print("Đang huấn luyện mô hình Sklearn LinearSVC (Tối ưu cho dữ liệu ảnh lớn)...")
# Dùng LinearSVC giải quyết triệt để lỗi không hội tụ của Sklearn
svm_sklearn = LinearSVC(C=1.0, max_iter=5000, dual=True, random_state=42)
svm_sklearn.fit(X_train, y_train)
y_pred_sk = svm_sklearn.predict(X_test)

# Thêm pos_label=-1 cho Sklearn
scores_sk = {
    "Precision": precision_score(y_test, y_pred_sk, pos_label=-1),
    "Recall": recall_score(y_test, y_pred_sk, pos_label=-1),
    "F1": f1_score(y_test, y_pred_sk, pos_label=-1)
}

# ===================== BẢNG SO SÁNH =====================
print("\n" + "="*60)
print("SO SÁNH KẾT QUẢ (TEST SET - Class: PNEUMONIA)")
print("="*60)
print(f"{'Metric':<15} {'Custom SVM':<15} {'Sklearn SVM':<15}")
print("-" * 45)
for m in scores:
    print(f"{m:<15} {scores[m]:<15.4f} {scores_sk[m]:<15.4f}")

# ===================== TRỰC QUAN HÓA BÁO CÁO =====================
print("\nĐang hiển thị biểu đồ báo cáo...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ 1: Loss Function
ax1.plot(model.losses, color='blue', linewidth=2)
ax1.set_title("Sự hội tụ Loss của Custom SVM (SGD)")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Total Loss")
ax1.grid(True)

# Biểu đồ 2: So sánh Bar Chart
x = np.arange(3)
metrics = ['Precision', 'Recall', 'F1']
ax2.bar(x - 0.2, [scores[m] for m in metrics], 0.4, label='Custom SVM', color='dodgerblue')
ax2.bar(x + 0.2, [scores_sk[m] for m in metrics], 0.4, label='Sklearn SVM', color='darkorange')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.set_ylim(0, 1.1)
ax2.set_title("So sánh Hiệu suất trên Test Set (Bệnh Viêm Phổi)")
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()