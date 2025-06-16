import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Метод опорных векторов: Классификация цветов
data = pd.read_csv('flowers.csv')
X = data.iloc[:, :-1]
y = data['species']

# Уменьшение размерности для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Обучение SVM
model = SVC(kernel='rbf')
model.fit(X_pca, y)

# Визуализация
plt.figure(figsize=(10,6))
for species in y.unique():
    plt.scatter(X_pca[y==species, 0], X_pca[y==species, 1], label=species)
    
plt.title('Классификация цветов методом SVM')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()