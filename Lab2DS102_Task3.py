import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
data = pd.read_csv('C:\\Users\\pc\\Downloads\\predict+students+dropout+and+academic+success\\data.csv', delimiter=';')

# Kiểm tra và loại bỏ các giá trị không hợp lệ trong cột 'Target'
data['Target'] = data['Target'].apply(lambda x: 0 if x == 'Dropout' else (1 if x == 'Graduate' else 2))

# Xóa các dòng có giá trị NaN trong cột 'Target'
data_cleaned = data.dropna(subset=['Target'])

# Tách các đặc trưng và mục tiêu (target)
X_cleaned = data_cleaned.drop(columns=['Target'])
y_cleaned = data_cleaned['Target'].astype(int)

# Đảm bảo tất cả các đặc trưng là kiểu số
X_cleaned = X_cleaned.apply(pd.to_numeric, errors='coerce')

# Xóa các dòng có giá trị NaN trong các đặc trưng
X_cleaned = X_cleaned.dropna()

# Đồng bộ hóa lại y_cleaned sau khi xóa các dòng trong X_cleaned
y_cleaned = y_cleaned[X_cleaned.index]

# Chuẩn hóa các đặc trưng
scaler = StandardScaler()
X_cleaned = scaler.fit_transform(X_cleaned)

# Khởi tạo mô hình Logistic Regression cho bài toán phân loại 3 lớp
log_reg = OneVsRestClassifier(LogisticRegression(max_iter=1000))

# Huấn luyện mô hình Logistic Regression
log_reg.fit(X_cleaned, y_cleaned)

# Dự đoán
y_pred = log_reg.predict(X_cleaned)

# Đánh giá mô hình
accuracy = accuracy_score(y_cleaned, y_pred)
print(f"Accuracy of Logistic Regression: {accuracy * 100:.2f}%")

# Trực quan hóa kết quả
plt.scatter(range(len(y_cleaned)), y_cleaned, color='blue', label='True Labels', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Labels', alpha=0.5)
plt.title('True vs Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()
