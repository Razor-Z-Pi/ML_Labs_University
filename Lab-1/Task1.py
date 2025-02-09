import matplotlib.pyplot as plt
import numpy as np

X = np.array([2, 3, 4, 5])
Y = np.array([13, 9, 8, 7])

n = len(X)

sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_XY = np.sum(X * Y)
sum_X2 = np.sum(X**2)

a1 = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X**2)
a0 = (sum_Y - a1 * sum_X) / n

print(f"Уравнение регрессии: Y = {a0:.2f} + {a1:.2f} * X")

x_star = 4.5
y_star = a0 + a1 * x_star
print(f"Предсказанное значение Y для X = {x_star}: {y_star:.2f}")

plt.scatter(X, a0 + a1 * X, color='blue', label='Регрессия данные')
plt.plot(X, a0 + a1 * X, color='red', label='Линия регрессии')
plt.scatter([x_star], [y_star], color='green', label=f'Предсказание (X = {x_star})')
plt.scatter([y_star], [x_star], color='yellow', label=f'Предсказание (Y = {y_star})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Линейная регрессия')
plt.grid(True)
plt.show()