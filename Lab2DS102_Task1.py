import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Định nghĩa hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm tính toán mất mát (cross-entropy loss)
def compute_loss(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    loss = -(1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return loss

# Hàm Gradient Descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    loss_history = []
    
    for i in range(num_iterations):
        # Dự đoán (hàm sigmoid)
        h = sigmoid(np.dot(X, theta))
        
        # Tính gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Cập nhật trọng số
        theta = theta - learning_rate * gradient
        
        # Tính và lưu lại mất mát sau mỗi vòng lặp
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)
    
    return theta, loss_history

# Đọc lại dữ liệu với đúng dấu phân cách
data = pd.read_csv('C:\\Users\\pc\\Downloads\\predict+students+dropout+and+academic+success\\data.csv', delimiter=';')

# Kiểm tra và loại bỏ các giá trị không hợp lệ trong cột 'Target'
data['Target'] = data['Target'].apply(lambda x: 1 if x == 'Graduate' else (0 if x == 'Dropout' else np.nan))

# Xóa các dòng có giá trị NaN trong cột 'Target'
data_cleaned = data.dropna(subset=['Target'])

# Tách các đặc trưng và mục tiêu (target)
X_cleaned = data_cleaned.drop(columns=['Target'])
y_cleaned = data_cleaned['Target']

# Đảm bảo tất cả các đặc trưng là kiểu số
X_cleaned = X_cleaned.apply(pd.to_numeric, errors='coerce')

# Xóa các dòng có giá trị NaN trong các đặc trưng
X_cleaned = X_cleaned.dropna()

# Đồng bộ hóa lại y_cleaned sau khi xóa các dòng trong X_cleaned
y_cleaned = y_cleaned[X_cleaned.index]

# Chuẩn hóa các đặc trưng
X_cleaned = (X_cleaned - np.mean(X_cleaned, axis=0)) / np.std(X_cleaned, axis=0)

# Thêm cột intercept vào X_cleaned
X_cleaned = np.c_[np.ones((X_cleaned.shape[0], 1)), X_cleaned]

# Kiểm tra lại số lượng dòng trong X và y
print(f"X_cleaned shape: {X_cleaned.shape}")
print(f"y_cleaned length: {len(y_cleaned)}")

# Tiến hành huấn luyện mô hình logistic regression với Gradient Descent
theta_cleaned = np.zeros(X_cleaned.shape[1])

# Thực thi Gradient Descent
theta_cleaned, loss_history_cleaned = gradient_descent(X_cleaned, y_cleaned.values, theta_cleaned, learning_rate=0.01, num_iterations=1000)

# Hiển thị đồ thị mất mát
plt.plot(loss_history_cleaned)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during Gradient Descent')
plt.show()

# Kết quả của theta
print(f"Final theta: {theta_cleaned}")
