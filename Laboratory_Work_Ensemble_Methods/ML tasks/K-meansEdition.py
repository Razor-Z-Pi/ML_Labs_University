import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

np.random.seed(42)
X1 = np.random.randn(50, 2) + [2, 2]
X2 = np.random.randn(50, 2) + [-2, -2]
X3 = np.random.randn(50, 2) + [2, -4]
X = np.vstack([X1, X2, X3])

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=40)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
plt.title("K-means (sklearn)")
plt.show()