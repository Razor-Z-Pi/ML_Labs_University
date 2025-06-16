from sklearn.neighbors import KNeighborsClassifier

# KNN: Классификация фруктов
# Данные: [[вес (г), текстура (0-шершавый, 1-гладкий)], метка (0-яблоко, 1-апельсин)]
X = [[150, 0], [170, 0], [130, 1], [140, 1]]
y = [0, 0, 1, 1]

# Модель KNN (k=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Предсказание для фрукта: 160г, шершавый
pred = model.predict([[160, 0]])
print("Яблоко" if pred[0] == 0 else "Апельсин")