from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Логистическая регрессия: Классификация ирисов
# Загружаем данные
iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)  # Длина лепестка
y = (iris.target == 0).astype(int)   # 1 для Setosa, 0 для остальных

# Обучаем модель
model = LogisticRegression()
model.fit(X, y)

# Предсказание для лепестка длиной 2 см
pred = model.predict([[2]])
print("Это Setosa!" if pred[0] == 1 else "Это не Setosa")

# Вероятности
prob = model.predict_proba([[2]])
print(f"Вероятности: Not-Setosa={prob[0][0]:.2f}, Setosa={prob[0][1]:.2f}")