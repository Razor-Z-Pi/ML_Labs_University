import numpy as np
from sklearn.linear_model import LinearRegression

# Линейная регрессия: Предсказание цены дома
# Данные: площадь (м²) -> цена (тыс. $)
X = np.array([[50], [70], [90], [110]])  # Площадь
y = np.array([300, 400, 500, 600])       # Цена

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X, y)

# Предсказание для дома 80 м²
pred = model.predict([[80]])
print(f"Предсказанная цена: {pred[0]:.2f} тыс. $")

# Коэффициенты
print(f"Уравнение: цена = {model.coef_[0]:.2f} * площадь + {model.intercept_:.2f}")