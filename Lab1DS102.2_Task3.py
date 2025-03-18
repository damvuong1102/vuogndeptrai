import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tính toán ma trận hệ số tương quan
correlation_matrix = pd.DataFrame(X_scaled, columns=df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day']].columns)
correlation_matrix = correlation_matrix.corr()

# In ra ma trận hệ số tương quan
print("Ma trận hệ số tương quan:")
print(correlation_matrix)

# Lọc các thuộc tính có hệ số tương quan > 0.9 (hoặc < -0.9)
threshold = 0.9
columns_to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            columns_to_drop.add(colname)

# Hiển thị các thuộc tính sẽ bị loại bỏ
print("Các thuộc tính sẽ bị loại bỏ do có hệ số tương quan cao:")
print(columns_to_drop)

# Loại bỏ các thuộc tính có hệ số tương quan cao
X_filtered = np.delete(X_scaled, [list(correlation_matrix.columns).index(col) for col in columns_to_drop], axis=1)

# Thêm cột 1 vào X_filtered cho bias term
X_filtered = np.c_[np.ones(X_filtered.shape[0]), X_filtered]

# Hàm hồi quy tuyến tính
def linear_regression(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# Tính toán theta cho dữ liệu đã lọc
theta_filtered = linear_regression(X_filtered, y)

# Dự đoán giá trị của area với dữ liệu đã lọc
y_pred_filtered = X_filtered.dot(theta_filtered)

# In ra kết quả
print("Hệ số theta với dữ liệu đã lọc các thuộc tính correlated:", theta_filtered)
print("Dự đoán area với dữ liệu đã lọc: ", y_pred_filtered[:5])

# So sánh MSE giữa dữ liệu đã lọc và dữ liệu chưa lọc
mse_filtered = mean_squared_error(y, y_pred_filtered)
print(f"MSE (Mean Squared Error) với dữ liệu đã lọc: {mse_filtered}")
