import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X1 = np.random.randn(50, 2) + [2, 2]
X2 = np.random.randn(50, 2) + [-2, -2]
X3 = np.random.randn(50, 2) + [2, -4]
X = np.vstack([X1, X2, X3])

def kmeans_manual(X, k, max_iters=100, tol=1e-4):
    # Случайная инициализация центроидов
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    
    for i in range(max_iters):
        # Вычисляем расстояния от точек до центроидов
        # (X[:, np.newaxis] - centroids) -> магия трансляции :)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Назначаем кластер (индекс минимального расстояния)
        labels = np.argmin(distances, axis=1)
        
        # Пересчитываем центроиды
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # Проверка сходимости
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
            
        centroids = new_centroids
        
    return centroids, labels

# Используем те же данные <X>
centers, labels = kmeans_manual(X, k=3)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
plt.title("K-means (Ручная реализация)")
plt.show()