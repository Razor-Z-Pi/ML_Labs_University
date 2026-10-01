# Импортирую функцию для загрузки набора данных Ирисы
from sklearn.datasets import load_iris
# Импортирую функцию для разделения данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
# Импортирую класс для создания модели случайного леса
from sklearn.ensemble import RandomForestClassifier
# Импортирую метрику для оценки точности модели
from sklearn.metrics import accuracy_score

# Загружаю набор данных Ирисы
data = load_iris()
# Сохраняю признаки в переменную X
X = data.data
# Сохраняю целевую переменную в переменную y
y = data.target

# Разделяю данные на обучающую и тестовую выборки
# Выделяю 30 процентов данных для тестирования и фиксирую random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 99)

# Создаю модель случайного леса, состоящую из 100 деревьев
# Также фиксирую random_state для воспроизводимости результатов
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
# Обучаю модель на тренировочных данных
rf_model.fit(X_train, y_train)

# Выполняю прогнозирование на тестовых данных
y_pred = rf_model.predict(X_test)
# Вычисляю точность модели, сравнивая предсказанные значения с реальными
accuracy = accuracy_score(y_test, y_pred)
# Вывожу точность на экран, форматируя число до 8 знаков после запятой
print(f'tochnost modeli Random Forest: {accuracy:.8f}')