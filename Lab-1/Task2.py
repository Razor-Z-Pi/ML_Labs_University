import numpy as np
import matplotlib.pyplot as plt

X = np.array([3, 8, 5, 10, 7, 6, 4, 9, 1, 2])
Y = np.array([6, 5, 9, 1, 8, 9, 8, 4, 2, 4])

X_range = np.linspace(min(X), max(X), 100)

def linear_model(x):
    return 6.067 - 0.085 * x

def quad_model(x):
    return -2.017 + 3.957 * x - 0.367 * x**2

def exp_model(x):
    return 5.918 * np.exp(-0.043 * x)

plt.figure(figsize=(10, 6))

plt.scatter(X, Y, color='blue', label='Исходные данные')

plt.plot(X_range, linear_model(X_range), color='red', label='Линейная модель: y = 6.067 - 0.085x')

plt.plot(X_range, quad_model(X_range), color='green', label='Квадратичная модель: y = -2.017 + 3.957x - 0.367x²')

plt.plot(X_range, exp_model(X_range), color='purple', label='Экспоненциальная модель: y = 5.918e^(-0.043x)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Регрессионные модели')
plt.legend()
plt.grid(True)
plt.show()