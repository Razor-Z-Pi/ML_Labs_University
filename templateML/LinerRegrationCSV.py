import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Линейная регрессия: Предсказание зарплаты по опыту
# Загрузка данных
data = pd.read_csv('salaries.csv')
X = data[['years']]
y = data['salary']

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Визуализация
plt.scatter(X, y, color='blue', label='Данные')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Прогноз')
plt.title('Зарплата vs Опыт')
plt.xlabel('Опыт (лет)')
plt.ylabel('Зарплата ($)')
plt.legend()
plt.show()

# Предсказание
print(f"Прогноз зарплаты для 8 лет опыта: ${model.predict([[8]])[0]:.2f}")