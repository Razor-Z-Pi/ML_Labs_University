import numpy as np

np.random.seed(42)
X = np.random.rand(100, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

X_b = np.c_[np.ones((len(X), 1)), X]  # Добавляем Перехват
m, n = X_b.shape

alpha = 1.0

# Матрица штрафа
A = np.eye(n)
A[0][0] = 0

# Расчет коэффициентов
theta_ridge = np.linalg.inv(X_b.T.dot(X_b) + alpha * A).dot(X_b.T).dot(y)

print("Веса (коэффициенты):", theta_ridge[1:])
print("Перехват:", theta_ridge[0])