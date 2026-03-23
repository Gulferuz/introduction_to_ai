import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # для 3D-графика

# данные
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = np.array([5, 7, 9, 11])

# модель
model = LinearRegression()
model.fit(X, y)

# предсказание
print("Предсказание для [5, 6]:", model.predict([[5, 6]]))

# визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# синие точки — данные
ax.scatter(X[:,0], X[:,1], y, color='blue', s=50)

# предсказание плоскости
x1 = np.linspace(1, 5, 10)
x2 = np.linspace(2, 6, 10)
x1, x2 = np.meshgrid(x1, x2)
y_pred = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

# красная плоскость — модель
ax.plot_surface(x1, x2, y_pred, color='red', alpha=0.5)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.title("Множественная линейная регрессия")
plt.show()