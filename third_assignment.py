import numpy as np
from sklearn.linear_model import LinearRegression

# данные
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
])
y = np.array([6, 9, 12, 15])

# модель
model = LinearRegression()
model.fit(X, y)

# вывод коэффициентов
print("Коэффициенты:", model.coef_)
print("Свободный член (intercept):", model.intercept_)

# предсказание для нового примера
print("Предсказание для [5,6,7]:", model.predict([[5,6,7]]))