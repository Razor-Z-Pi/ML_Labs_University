# Импортирую функцию для загрузки классического набора данных Ирисы
from sklearn.datasets import load_iris
# Импортирую функцию для разделения данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
# Импортирую класс для создания модели дерева решений
from sklearn.tree import DecisionTreeClassifier
# Импортирую метрику для оценки точности модели
from sklearn.metrics import accuracy_score

# Загружаю набор данных Ирисы
data = load_iris()
# Сохраняю признаки (характеристики цветков) в переменную X
X = data.data
# Сохраняю целевую переменную (классы цветков) в переменную y
y = data.target

# Разделяю данные на обучающую и тестовую выборки
# Выделяю 30 процентов данных для тестирования и фиксирую random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 99)

# Создаю экземпляр классификатора на основе дерева решений
clf = DecisionTreeClassifier(random_state = 1)
# Обучаю модель на тренировочных данных
clf.fit(X_train, y_train)

# Делаю предсказания на основе тестовых данных
y_pred = clf.predict(X_test)
# Вычисляю точность модели, сравнивая предсказанные значения с реальными
accuracy = accuracy_score(y_test, y_pred)
# Вывожу полученную точность на экран
print(f'Accuracy: {accuracy}')