import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(x, y)

# Получение коэффициентов (y = kx + b)
print("Наклон (коэффициент k):", model.coef_[0])
print("Смещение (сечение b):", model.intercept_)

y_pred = model.predict(np.array([[6]]))
print("Прогноз для x = 6:", y_pred[0])

# Маттерьял есть на https://proglib.io/p/linear-regression