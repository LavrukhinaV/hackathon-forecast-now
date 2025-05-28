import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt




# === 3. Регрессионная модель Lasso для интерпретируемости
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Загружаем датафрейм с категориями
df = pd.read_csv('train_clustered_categories.csv')
X = df[['avg_temp_c', 'precipitation_mm', 'city_code', 'category']]
y = df['log_units']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Lasso-регрессия с масштабированием
from sklearn.linear_model import LassoCV

model = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42))

model.fit(X_train, y_train)

# Получение и вывод коэффициентов
Lasso = model.named_steps['lassocv']
coefs = pd.Series(Lasso.coef_, index=X.columns)

print("\n📌 Коэффициенты влияния погодных и географических признаков:")
print(coefs)

# === Визуализация коэффициентов
plt.figure(figsize=(8, 5))
sns.barplot(x=coefs.values, y=coefs.index)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Коэффициенты влияния погоды и географии на логарифм продаж")
plt.xlabel("Коэффициент (лог-продажи)")
plt.ylabel("Признак")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. Корректировка прогноза на основе погодных коэффициентов
df['adjusted_log_units'] = (
    df['log_units']
    + coefs['avg_temp_c'] * df['avg_temp_c']
    + coefs['precipitation_mm'] * df['precipitation_mm']
)
df['adjusted_units'] = df['adjusted_log_units'].apply(lambda x: max(0, round(np.expm1(x))))

# === 5. Сохранение результатов
coefs.to_csv('Lasso_weather_coefficients.csv')
df.to_csv('train_with_adjusted_forecast.csv', index=False)
print("\n✅ Данные с учётом погодной коррекции сохранены в train_with_adjusted_forecast.csv")

"""Мультиколлинеарность между температурой и осадками есть, но мы её контролируем через регуляризацию (Lasso) — это сохраняет интерпретируемость модели.
Температура и осадки действительно влияют на спрос, но слабо — важно учитывать их в сочетании с другими признаками.

Категория товара — ключевой фактор: товары с определённым паттерном продаж продаются существенно хуже/лучше.
Погодные факторы, такие как температура и осадки, имеют положительное, но слабое влияние.

Код города практически не влияет на результат, что может говорить о слабой его интерпретируемости в текущем виде.


"""