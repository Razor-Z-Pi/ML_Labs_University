import numpy as np

np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Признак
y = 2.5 * X + np.random.randn(100, 1) * 2 + 5 # Цель: y = 2.5x + 5 + шум

X_b = np.c_[np.ones((len(X), 1)), X]  # Матрица [1, x]

# Нормальное уравнение
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

intercept = theta_best[0][0]
slope = theta_best[1][0]

print(f"Коэффициент: {slope:.2f}")
print(f"Свободный член: {intercept:.2f}")

# Предсказание
y_pred_manual = X_b.dot(theta_best)