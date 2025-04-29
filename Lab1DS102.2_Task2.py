import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


file_path = 'C:\\Users\\pc\\Downloads\\forest+fires\\forestfires.csv'
df = pd.read_csv(file_path)


def convert_day(day: str) -> int:
    day_dict = {
        "sun": 0,
        "mon": 1,
        "tue": 2,
        "wed": 3,
        "thu": 4,
        "fri": 5,
        "sat": 6,
    }
    return day_dict[day]

def convert_month(month: str) -> int:
    month_dict = {  
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return month_dict[month]


df['month'] = df['month'].apply(convert_month)
df['day'] = df['day'].apply(convert_day)


X = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day']].values
y = df['area'].values


X = np.c_[np.ones(X.shape[0]), X]


def linear_regression(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# theta cho dữ liệu chưa chuẩn hóa 
theta_original = linear_regression(X, y)

# Dự đoán giá trị của area với dữ liệu chưa chuẩn hóa 
y_pred_original = X.dot(theta_original)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day']].values)


X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# Tính toán theta cho dữ liệu đã chuẩn hóa 
theta_scaled = linear_regression(X_scaled, y)

# Dự đoán giá trị của area với dữ liệu đã chuẩn hóa 
y_pred_scaled = X_scaled.dot(theta_scaled)

# In ra kết quả
print("Hệ số hồi quy (theta) với dữ liệu chưa chuẩn hóa:", theta_original)
print("Dự đoán area với dữ liệu chưa chuẩn hóa: ", y_pred_original[:5])

print("Hệ số hồi quy (theta) với dữ liệu đã chuẩn hóa:", theta_scaled)
print("Dự đoán area với dữ liệu đã chuẩn hóa: ", y_pred_scaled[:5])

# So sánh kết quả của hai mô hình
from sklearn.metrics import mean_squared_error

mse_original = mean_squared_error(y, y_pred_original)
mse_scaled = mean_squared_error(y, y_pred_scaled)

print(f"MSE (Mean Squared Error) với dữ liệu chưa chuẩn hóa: {mse_original}")
print(f"MSE (Mean Squared Error) với dữ liệu đã chuẩn hóa: {mse_scaled}")

