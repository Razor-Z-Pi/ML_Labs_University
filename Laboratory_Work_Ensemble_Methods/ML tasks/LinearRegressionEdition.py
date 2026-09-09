import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Признак
y = 2.5 * X + np.random.randn(100, 1) * 2 + 5 # Цель: y = 2.5x + 5 + шум

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(f"Коэффициент: {model.coef_[0][0]:.2f}")
print(f"Свободный член: {model.intercept_[0]:.2f}")