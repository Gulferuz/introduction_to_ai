import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# данные (пример, можно свои)
x = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 3, 5, 7, 9])

# преобразование в полиномиальные признаки степени 2
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# модель
model = LinearRegression()
model.fit(x_poly, y)

# предсказание
y_pred = model.predict(x_poly)

# график
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.title("Полиномиальная регрессия")
plt.show()