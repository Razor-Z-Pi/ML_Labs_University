from sklearn.linear_model import Ridge
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

# Создаем модель с альфа (сила регуляризации!!!)
ridge_model = Ridge(alpha = 1.0)
ridge_model.fit(X, y)

print("Веса (коэффициенты):", ridge_model.coef_)
print("Перехват:", ridge_model.intercept_)