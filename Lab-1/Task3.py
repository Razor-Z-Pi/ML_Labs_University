import numpy as np
import matplotlib.pyplot as plt

groups = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
prices = np.array([50, 60, 70, 80, 95, 100, 115, 120, 105, 120, 130, 110, 150, 190, 120, 130, 220, 145, 265, 270])

n = len(groups)

sum_X = np.sum(groups)
sum_Y = np.sum(prices)
sum_XY = np.sum(groups * prices)
sum_X2 = np.sum(groups**2)

a1 = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X**2)
a0 = (sum_Y - a1 * sum_X) / n

print(f"Уравнение линейной регрессии: y = {a0:.2f} + {a1:.2f} * x")

secret = a0 + a1 * groups

plt.figure(figsize=(10, 6))
plt.scatter(groups, prices, color='blue', label='Исходные данные')
plt.plot(groups, secret, color='red', label='Линейная регрессия')
plt.xlabel('Группа')
plt.ylabel('Цена')
plt.title('Линейная регрессия: Зависимость цены от группы')
plt.grid(True)
plt.legend()
plt.show()

total = np.sum((prices - np.mean(prices))**2) 
result = np.sum((prices - secret)**2)
otv = 1 - (total / result)
print(f"Коэффициент детерминации (R^2): {otv:.2f}")