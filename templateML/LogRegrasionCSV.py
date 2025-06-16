import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Логистическая регрессия: Классификация клиентов банка

data = pd.read_csv('bank_customers.csv')
X = data[['age', 'credit_score']]
y = data['approved']

# Обучение модели
model = LogisticRegression()
model.fit(X, y)

# Визуализация
sns.scatterplot(x='age', y='credit_score', hue='approved', data=data)
plt.title('Одобрение кредита по возрасту и кредитному рейтингу')
plt.show()

# Предсказание
new_client = [[30, 720]]
prob = model.predict_proba(new_client)[0][1]
print(f"Вероятность одобрения: {prob*100:.2f}%")