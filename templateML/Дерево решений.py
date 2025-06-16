from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Дерево решений: Прогноз погоды
# Данные: [солнечно (1/0), температура], метка (1-гулять, 0-не гулять)
X = [[1, 25], [0, 10], [1, 15], [0, 20]]
y = [1, 0, 1, 0]

# Обучаем дерево
model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

# Визуализация дерева
plot_tree(model, feature_names=["Солнце", "Температура"], class_names=["Не гулять", "Гулять"])
plt.show()

# Предсказание для солнечно, 18°
pred = model.predict([[1, 18]])
print("Гулять!" if pred[0] == 1 else "Сидим дома")