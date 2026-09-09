from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Простые данные: <*{Вес(г), Диаметр_шляпки(см)}*>
# 1 - съедобный, 0 - ядовитый
X = [[50, 5], [60, 6], [30, 4], [80, 10], [20, 3], [90, 12]]
y = [1, 1, 1, 0, 1, 0]  # Допустим, слишком большие ядовитые

clf = DecisionTreeClassifier(max_depth=2, criterion='gini')
clf.fit(X, y)

# Предсказание нового гриба (вес 55г, шляпка 7 см)
new_mushroom = [[55, 7]]
prediction = clf.predict(new_mushroom)
print("Класс нового гриба (1-съедобный, 0-яд):", prediction[0])

# Визуализация дерева
plot_tree(clf, filled = True, feature_names = ['Вес', 'Диаметр'], class_names = ['Ядовитый', 'Съедобный'])
plt.show()