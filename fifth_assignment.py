import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

# данные
x = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 3, 5, 7, 9])

# полиномиальные признаки степени 2
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# обычная полиномиальная регрессия (без регуляризации)
model = LinearRegression()
model.fit(x_poly, y)
y_pred = model.predict(x_poly)

# Ridge регрессия
ridge = Ridge(alpha=1.0)
ridge.fit(x_poly, y)
y_ridge = ridge.predict(x_poly)

# Lasso регрессия
lasso = Lasso(alpha=0.1)
lasso.fit(x_poly, y)
y_lasso = lasso.predict(x_poly)

# график
plt.scatter(x, y, color='blue', label="Данные")
plt.plot(x, y_pred, color='green', label="Без регуляризации")
plt.plot(x, y_ridge, color='red', linestyle='--', label="Ridge")
plt.plot(x, y_lasso, color='orange', linestyle=':', label="Lasso")
plt.title("Полиномиальная регрессия с регуляризацией")
plt.legend()
plt.show()