import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

BASE_DIR = "chest_xray"

def collect_data(split: str = "train"):
    normal = "NORMAL"      # 1
    pneumonia = "PNEUMONIA" # -1
    
    images = []
    labels = []
    
    for img_file in tqdm(os.listdir(os.path.join(BASE_DIR, split, normal)), desc=f"Split {split} - Collecting {normal}"):
        img = cv.imread(os.path.join(BASE_DIR, split, normal, img_file))
        if img is not None: 
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            images.append(cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT).reshape(-1))
            labels.append(1)
            
    for img_file in tqdm(os.listdir(os.path.join(BASE_DIR, split, pneumonia)), desc=f"Split {split} - Collecting {pneumonia}"):
        img = cv.imread(os.path.join(BASE_DIR, split, pneumonia, img_file))
        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            images.append(cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT).reshape(-1))
            labels.append(-1)
            
    # Chỉ stack thành ma trận và scale nhẹ pixel về khoảng [0, 1] cho an toàn
    X = np.stack(images, axis=0) / 255.0
    y = np.array(labels)
    
    return X, y

if __name__ == "__main__":
    print("Đang tải dữ liệu tập Train...")
    X_train, y_train = collect_data("train")
    
    print("\nĐang tải dữ liệu tập Test...")
    X_test, y_test = collect_data("test")
    
    print("\n--- CHUẨN HÓA DỮ LIỆU (Chống Data Leakage) ---")
    # Tính Mean và Std CHỈ DỰA TRÊN TẬP TRAIN
    mean_train = X_train.mean()
    std_train = X_train.std()
    
    # Dùng Mean/Std của Train để áp dụng cho CẢ Train và Test
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train
    print("Hoàn tất chuẩn hóa Z-score đúng quy chuẩn!")
    
    # Lưu ma trận đã xử lý chuẩn chỉ ra file để dùng cho các bài tập
    print("\nĐang lưu dữ liệu ra file 'processed_data.npz'...")
    np.savez("processed_data.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("Đã lưu thành công!")