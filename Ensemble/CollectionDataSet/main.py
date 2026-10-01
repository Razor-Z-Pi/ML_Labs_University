import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, 
                             roc_curve, auc)

# Генерация синтетического датасета для задачи "Обнаружение мошенничества"
# Создаем 1000 образцов, 20 признаков, дисбаланс классов (5% мошенничества)
# Это имитирует реальную ситуацию, где мошеннических операций гораздо меньше, чем обычных
X, y = make_classification(n_samples = 1000, n_features = 20, n_informative = 10, 
                           n_redundant = 5, n_classes = 2, weights = [0.95, 0.05], 
                           random_state = 42)

# Разделение данных на обучающую и тестовую выборки
# Используем стратификацию, чтобы сохранить пропорцию классов в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 42, stratify = y)

# Создание и обучение модели Случайный лес (Random Forest)
# n_estimators = 100 - количество деревьев в лесу
# random_state = 42 - для воспроизводимости результатов
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = rf_model.predict(X_test)

# Построение матрицы ошибок (Confusion Matrix)
# Матрица показывает: TN (True Negative), FP (False Positive), FN (False Negative), TP (True Positive)
cm = confusion_matrix(y_test, y_pred)

# Расчет метрик качества на основе матрицы ошибок
# Accuracy - общая точность (правильные ответы / все ответы)
accuracy = accuracy_score(y_test, y_pred)
# Precision - точность (из тех, кого назвали мошенниками, сколько реально мошенники)
precision = precision_score(y_test, y_pred)
# Recall - полнота (из всех реальных мошенников, сколько мы смогли найти)
recall = recall_score(y_test, y_pred)
# F1-score - гармоническое среднее между Precision и Recall
f1 = f1_score(y_test, y_pred)

print("#" * 70)
print("Метрика качественной модели (Random Forest)")
print("#" * 70)
print(f"Accuracy (Общая точность): {accuracy:.4f}")
print(f"Precision (Точность):      {precision:.4f}")
print(f"Recall (Полнота):          {recall:.4f}")
print(f"F1-score:                  {f1:.4f}")
print("#" * 70)

# Визуализация матрицы ошибок с помощью Seaborn и Matplotlib
plt.figure(figsize=(8, 6))
# Строим тепловую карту (heatmap) на основе матрицы ошибок
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', 
            xticklabels=['Обычная (0)', 'Мошенничество (1)'], 
            yticklabels=['Обычная (0)', 'Мошенничество (1)'])
plt.title('Матрица ошибок (Confusion Matrix) для Random Forest')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi = 300)
plt.show()

# Дополнительная визуализация: ROC-кривая
# Она показывает соотношение TPR (True Positive Rate) и FPR (False Positive Rate)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1] # Вероятности принадлежности к классу 1
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Ложноположительный результат)')
plt.ylabel('True Positive Rate (Истинноположительный результат)')
plt.title('ROC-кривая для модели Random Forest')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi = 300)
plt.show()