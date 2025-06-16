import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# K-средних: Кластеризация покупателей

data = pd.read_csv('mall_customers.csv')
X = data[['annual_income', 'spending_score']]

# Кластеризация
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)

# Визуализация
plt.scatter(X['annual_income'], X['spending_score'], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=300, c='red', marker='X')
plt.title('Кластеризация покупателей')
plt.xlabel('Годовой доход')
plt.ylabel('Баллы трат')
plt.show()