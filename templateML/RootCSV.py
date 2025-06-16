import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Дерево решений: Прогноз выживания на Титанике
data = pd.read_csv('titanic.csv')

# Преобразование категориальных данных
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])

X = data[['pclass', 'age', 'sex']]
y = data['survived']

# Обучение модели
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Визуализация дерева
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=['Класс', 'Возраст', 'Пол'], 
          class_names=['Погиб', 'Выжил'], filled=True)
plt.show()