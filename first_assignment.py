import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# данные
x = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 3, 5, 7, 9])

# модель
model = LinearRegression()
model.fit(x, y)

# предсказание
y_pred = model.predict(x)

# график
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.title("Линейная регрессия")
plt.show(block=True)