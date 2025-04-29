import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Định nghĩa hàm softmax
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# Hàm mất mát (cross-entropy loss)
def compute_loss(X, y, theta):
    m = len(y)
    y_hat = softmax(np.dot(X, theta))
    loss = -np.sum(np.log(y_hat[range(m), y])) / m
    return loss

# Thuật toán Gradient Descent cho Softmax Regression
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    loss_history = []
    
    for i in range(num_iterations):
        # Dự đoán với softmax
        y_hat = softmax(np.dot(X, theta))
        
        # Tính gradient
        gradient = np.dot(X.T, y_hat - np.eye(len(np.unique(y)))[y]) / m
        
        # Cập nhật trọng số
        theta -= learning_rate * gradient
        
        # Tính và lưu lại mất mát
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)
    
    return theta, loss_history

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
X_cleaned = (X_cleaned - np.mean(X_cleaned, axis=0)) / np.std(X_cleaned, axis=0)

# Thêm cột intercept vào X_cleaned
X_cleaned = np.c_[np.ones((X_cleaned.shape[0], 1)), X_cleaned]

# Kiểm tra lại số lượng dòng trong X và y
print(f"X_cleaned shape: {X_cleaned.shape}")
print(f"y_cleaned length: {len(y_cleaned)}")

# Tiến hành huấn luyện mô hình Softmax Regression với Gradient Descent
theta_cleaned = np.zeros((X_cleaned.shape[1], len(np.unique(y_cleaned))))

# Thực thi Gradient Descent
theta_cleaned, loss_history_cleaned = gradient_descent(X_cleaned, y_cleaned.values, theta_cleaned, learning_rate=0.01, num_iterations=1000)

# Hiển thị đồ thị mất mát
plt.plot(loss_history_cleaned)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during Gradient Descent (Softmax Regression)')
plt.show()

# Kết quả của theta
print(f"Final theta: {theta_cleaned}")
