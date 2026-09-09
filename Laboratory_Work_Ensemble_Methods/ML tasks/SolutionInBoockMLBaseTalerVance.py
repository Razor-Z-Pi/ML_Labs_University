# – Линейная и логистическая регрессия
# – Регуляризация: Ridge и Lasso
# – Проблемы переобучения

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Создадим искусственный набор данных
data = {
    'Площадь': [30, 45, 50, 60, 75, 90, 110, 150, 200, 250],
    'Комнат': [1, 1, 2, 2, 3, 3, 4, 5, 6, 7],
    'Удаленность_от_центра': [10, 8, 7, 5, 4, 3, 2, 1, 1, 1],
    'Цена': [3.5, 5.0, 6.0, 7.5, 9.0, 11.0, 13.5, 18.0, 24.0, 30.0]
}

df = pd.DataFrame(data)
# Разделим данные на входные признаки (X) и целевую переменную (y)
X = df[['Площадь', 'Комнат', 'Удаленность_от_центра']]
y = df['Цена']

# Разделим выборку на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель
model.fit(X_train, y_train)

# Предсказываем цены квартир на тестовых данных
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")

plt.scatter(y_test, y_pred)
plt.xlabel("Фактическая цена (млн)")
plt.ylabel("Предсказанная цена (млн)")
plt.title("Сравнение предсказанных и реальных цен")
plt.show()

# Второе решение

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Загружаем данные Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Оставляем только нужные колонки
df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch']]

# Удаляем строки с пропущенными значениями
df.dropna(inplace=True)

# Разделяем на признаки и целевую переменную
X = df[['Pclass', 'Age', 'SibSp', 'Parch']]
y = df['Survived']

# Стандартизация данных для улучшения качества модели
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
random_state=42)

# Создаем модель логистической регрессии
log_model = LogisticRegression()

# Обучаем модель
log_model.fit(X_train, y_train)

# Предсказываем выживаемость пассажиров
y_pred = log_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

feature_names = ['Pclass', 'Age', 'SibSp', 'Parch']
coefficients = log_model.coef_[0]
# Визуализируем влияние факторов
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_names, y=coefficients)
plt.xlabel("Признаки")
plt.ylabel("Коэффициенты модели")
plt.title("Влияние факторов на вероятность выживания")
plt.show()