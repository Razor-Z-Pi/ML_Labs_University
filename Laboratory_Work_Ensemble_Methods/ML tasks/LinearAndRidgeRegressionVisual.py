import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)

print("*" * 60)
print("Визуализация линейной и гребновой регрессии")
print("*" * 60)

print("\n1. Линейная регрессия")
print("*" * 40)

np.random.seed(42)
X = np.random.rand(200, 1) * 10  # Признак от 0 до 10
y = 2.5 * X + np.random.randn(200, 1) * 2 + 5  # y = 2.5x + 5 + шум

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# Создание точек для линии регрессии
X_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = linear_model.predict(X_line)

# Метрики
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Коэффициент наклона: {linear_model.coef_[0][0]:.3f}")
print(f"Свободный член: {linear_model.intercept_[0]:.3f}")
print(f"MSE (тренируем): {train_mse:.3f}")
print(f"MSE (тест): {test_mse:.3f}")
print(f"R² (тест): {r2:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График № 1: Данные и линия регрессии
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, alpha = 0.6, label = 'Обучающие данные', color = 'blue', edgecolors = 'black', linewidth = 0.5)
ax1.scatter(X_test, y_test, alpha = 0.6, label = 'Тестовые данные', color = 'green', edgecolors = 'black', linewidth = 0.5)
ax1.plot(X_line, y_line, color = 'red', linewidth = 2, label = f'Линия регрессии: y = {linear_model.coef_[0][0]:.2f}x + {linear_model.intercept_[0]:.2f}')
ax1.set_xlabel('Признак X', fontsize = 12)
ax1.set_ylabel('Целевая переменная Y', fontsize = 12)
ax1.set_title('Линейная регрессия: данные и модель', fontsize = 14, fontweight = 'bold')
ax1.legend(loc = 'upper left')
ax1.grid(True, alpha = 0.3)

# График № 2: Остатки
ax2 = axes[0, 1]
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred
ax2.scatter(y_train_pred, residuals_train, alpha = 0.6, label = 'Обучение', color = 'blue', edgecolors = 'black', linewidth = 0.5)
ax2.scatter(y_test_pred, residuals_test, alpha = 0.6, label = 'Тестирование', color = 'green', edgecolors = 'black', linewidth = 0.5)
ax2.axhline(y = 0, color = 'red', linestyle = '--', linewidth = 1.5, label = 'Нулевая ошибка')
ax2.set_xlabel('Предсказанные значения', fontsize = 12)
ax2.set_ylabel('Остатки', fontsize = 12)
ax2.set_title('Анализ остатков', fontsize = 14, fontweight = 'bold')
ax2.legend()
ax2.grid(True, alpha = 0.3)

# График № 3: Предсказанные против Фактические значения
ax3 = axes[1, 0]
all_y = np.vstack([y_train, y_test])
all_pred = np.vstack([y_train_pred, y_test_pred])
ax3.scatter(all_y, all_pred, alpha = 0.6, color = 'purple', edgecolors = 'black', linewidth = 0.5)
ax3.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 'r--', linewidth = 2, label = 'Идеальное предсказание')
ax3.set_xlabel('Фактические значения', fontsize = 12)
ax3.set_ylabel('Предсказанные значения', fontsize = 12)
ax3.set_title('Предсказанные vs Фактические', fontsize = 14, fontweight = 'bold')
ax3.legend()
ax3.grid(True, alpha = 0.3)

print("\n2. Гребневая регрессия")
print("*" * 40)

# Генерация данных с мультиколлинеарностью
np.random.seed(42)
n_samples = 100
n_features = 5

# Создаем сильно коррелированные признаки
X_ridge = np.random.randn(n_samples, n_features)
X_ridge[:, 1] = X_ridge[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1  # Корреляция с первым
X_ridge[:, 2] = X_ridge[:, 0] * 0.8 + np.random.randn(n_samples) * 0.1  # Корреляция с первым

# Истинные коэффициенты (только первые два признака важны)
true_coefficients = np.array([3, 2, 0, 0, 0])
y_ridge = X_ridge @ true_coefficients + np.random.randn(n_samples) * 0.5

X_ridge_train, X_ridge_test, y_ridge_train, y_ridge_test = train_test_split(
    X_ridge, y_ridge, test_size = 0.2, random_state = 42
)

# Создание моделей с разными alpha
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_models = {}
coefficients_history = []

for alpha in alphas:
    ridge_model = Ridge(alpha = alpha)
    ridge_model.fit(X_ridge_train, y_ridge_train)
    ridge_models[alpha] = ridge_model
    coefficients_history.append(ridge_model.coef_)

# Обычная линейная регрессия для сравнения
linear_ridge = LinearRegression()
linear_ridge.fit(X_ridge_train, y_ridge_train)

coefficients_history = np.array(coefficients_history)

print("\nСравнение коэффициентов:")
print(f"{'Альфа':<10} {'Коэффициенты':<60} {'Норма':<15}")
print("-" * 85)
print(f"{'Линейная':<10} {str(linear_ridge.coef_.round(3)):<60} {np.linalg.norm(linear_ridge.coef_):.3f}")
for alpha, coef in zip(alphas, coefficients_history):
    print(f"{alpha:<10} {str(coef.round(3)):<60} {np.linalg.norm(coef):.3f}")

# График № 4: Влияние альфа на коэффициенты
ax4 = axes[1, 1]
for i in range(n_features):
    ax4.semilogx(alphas, coefficients_history[:, i], marker = 'o', linewidth = 2, 
                label = f'Признак {i+1} (истинный коэф: {true_coefficients[i]})')
ax4.axhline(y = 0, color = 'black', linestyle = '-', linewidth = 0.5)
ax4.set_xlabel('Альфа (сила регуляризации)', fontsize = 12)
ax4.set_ylabel('Значение коэффициента', fontsize = 12)
ax4.set_title('Влияние Гребня на коэффициенты', fontsize = 14, fontweight = 'bold')
ax4.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
ax4.grid(True, alpha = 0.3)

plt.suptitle('Визуализация регрессионных моделей', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()

# Дополнительная визуализация для Гребня
fig2, axes2 = plt.subplots(1, 2, figsize = (14, 5))

# График № 5: Сравнение предсказаний Линейная "против" Гребневой
ax5 = axes2[0]
y_test_pred_linear = linear_ridge.predict(X_ridge_test)
y_test_pred_ridge = ridge_models[1.0].predict(X_ridge_test)

ax5.scatter(range(len(y_ridge_test)), y_ridge_test, alpha = 0.7, label = 'Фактические', 
           color = 'green', edgecolors = 'black', linewidth = 0.5, s = 50)
ax5.scatter(range(len(y_test_pred_linear)), y_test_pred_linear, alpha = 0.7, label = 'Линейная', 
           color = 'blue', edgecolors = 'black', linewidth = 0.5, s =30)
ax5.scatter(range(len(y_test_pred_ridge)), y_test_pred_ridge, alpha = 0.7, label = 'Гребневая (альфа = 1.0)', 
           color = 'red', edgecolors = 'black', linewidth = 0.5, s = 30)
ax5.set_xlabel('Номер образца', fontsize = 12)
ax5.set_ylabel('Значение', fontsize = 12)
ax5.set_title('Сравнение предсказаний на тестовой выборке', fontsize = 14, fontweight = 'bold')
ax5.legend()
ax5.grid(True, alpha = 0.3)

# График 6: Метрики для разных альфа
ax6 = axes2[1]
mse_scores = []
r2_scores = []

for alpha in alphas:
    y_pred = ridge_models[alpha].predict(X_ridge_test)
    mse_scores.append(mean_squared_error(y_ridge_test, y_pred))
    r2_scores.append(r2_score(y_ridge_test, y_pred))

ax6.semilogx(alphas, mse_scores, marker = 'o', linewidth = 2, color = 'red', label = 'MSE')
ax6.set_xlabel('Альфа', fontsize = 12)
ax6.set_ylabel('MSE', fontsize = 12, color = 'red')
ax6.tick_params(axis = 'y', labelcolor = 'red')
ax6.set_title('Зависимость MSE от Альфа', fontsize = 14, fontweight = 'bold')
ax6.grid(True, alpha = 0.3)

# Вторая ось для R^²
ax6_twin = ax6.twinx()
ax6_twin.semilogx(alphas, r2_scores, marker = 's', linewidth = 2, color = 'blue', label = 'R^²')
ax6_twin.set_ylabel('R^²', fontsize = 12, color = 'blue')
ax6_twin.tick_params(axis = 'y', labelcolor = 'blue')

lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc = 'center right')

plt.suptitle('Анализ Гребня регрессии', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()
print("\n" + "*" * 60)
print("Ключевые выводы:")
print("*" * 60)
print("""
1. Линейная регрессия:
   - Находит прямую, минимизирующую MSE
   - Хорошо работает при отсутствии мультиколлинеарности
   - Коэффициенты интерпретируемы

2. Гребневая регрессия:
   - Сжимает коэффициенты к нулю (L2-регуляризация)
   - При увеличении альфа коэффициенты уменьшаются
   - Помогает при мультиколлинеарности
   - Предотвращает переобучение
""")