file_path = 'C:\\Users\\pc\\Downloads\\forest+fires\\forestfires.csv'  
import pandas as pd # type: ignore

df = pd.read_csv(file_path)
print(df.head())

####### Bài 1

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
    month_dict = {  # Cần phải gán vào biến month_dict
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

import numpy as np # type: ignore


X = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day']].values
y = df['area'].values

X = np.c_[np.ones(X.shape[0]), X]

X = np.array(X)
y = np.array(y)

def linear_regression(X, y):
    
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

theta = linear_regression(X, y)

y_pred = X.dot(theta)

print("Hệ số hồi quy (theta):", theta)
print("Dự đoán area: ", y_pred[:5])


