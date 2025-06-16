from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# K-means: Кластеризация цветов
# Данные: [[длина, ширина], ...]
X = [[1.5, 0.5], [2.0, 1.0], [3.0, 1.5], [5.0, 2.5], [6.0, 3.0]]

# Кластеризация (k=2)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels_

# Визуализация
plt.scatter(*zip(*X), c=labels)
plt.xlabel('Длина'), plt.ylabel('Ширина')
plt.title('Кластеры цветов')
plt.show()