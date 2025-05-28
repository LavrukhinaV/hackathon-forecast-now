import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt





# Загрузка очищенных данных
df = pd.read_csv('train_with_adjusted_forecast.csv')
df['units'] = df['adjusted_units']

# Выбор только числовых признаков
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Удалим ID и категориальные коды, если они есть
exclude_cols = ['store_nbr', 'item_nbr', 'unique_id_code']
numeric_df = numeric_df.drop(columns=[col for col in exclude_cols if col in numeric_df.columns], errors='ignore')

# Расчет корреляционной матрицы
corr_matrix = numeric_df.corr()

# Построение тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Корреляционная тепловая карта признаков")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

"""Погодные признаки (avg_temp_c, precipitation_mm) не коррелируют сильно между собой.

log_units сильно коррелирует с units.

Другие признаки (category, city_code) слабо коррелируют с остальными.

"""

# Загрузка очищенного датасета (или подставь свой файл)
df = pd.read_csv('train_with_adjusted_forecast.csv')
df['units'] = df['adjusted_units']

# Выбираем только числовые признаки
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Удалим идентификаторы, если они есть (по желанию)
exclude_cols = ['store_nbr', 'item_nbr', 'unique_id_code']
numeric_df = numeric_df.drop(columns=[col for col in exclude_cols if col in numeric_df.columns], errors='ignore')

# Построение корреляционной матрицы
correlation_matrix = numeric_df.corr()

# Отображение матрицы
print("📊 Корреляционная матрица:")
print(correlation_matrix.round(2))

df = pd.read_csv('train_with_adjusted_forecast.csv')
df['units'] = df['adjusted_units']

# Агрегируем общее количество продаж по каждому товару
item_sales = df.groupby('item_nbr')['units'].sum().sort_values(ascending=False)

# Построение графика
plt.figure(figsize=(12, 6))
sns.histplot(item_sales, bins=50, kde=True)
plt.title("Распределение общего количества продаж по item_nbr")
plt.xlabel("Суммарные продажи товара")
plt.ylabel("Количество товаров")
plt.grid(True)
plt.tight_layout()
plt.show()

""" 1. Пик около нуля
Большинство товаров (практически все) имеют низкие суммарные продажи

Это может означать:

товары с низким спросом

товары, появляющиеся/продающиеся только в отдельных магазинах или периодах

2. Немного товаров с ОЧЕНЬ высокими продажами
Некоторые единицы имеют продажи свыше 100,000 — 400,000+

Это "хиты продаж", которые:

популярны во многих магазинах

продаются постоянно

могут быть сезонными, но с массовым спросом


"""

df = pd.read_csv('train_with_adjusted_forecast.csv')
df['units'] = df['adjusted_units']

# Агрегируем общее количество продаж по каждому магазину
store_sales = df.groupby('store_nbr')['units'].sum().sort_values(ascending=False)

# Построение графика
plt.figure(figsize=(12, 6))
sns.barplot(x=store_sales.index.astype(str), y=store_sales.values)
plt.title("Суммарные продажи по каждому магазину (store_nbr)")
plt.xlabel("store_nbr")
plt.ylabel("Общие продажи")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

pivot = pd.pivot_table(
    df,
    values='units',
    index='store_nbr',
    columns='item_nbr',
    aggfunc='sum',
    fill_value=0
)

# Отображение части таблицы
print(pivot.head())

import holidays

# === 1. Загрузка данных ===
df = pd.read_csv('train_with_adjusted_forecast.csv')
df['units'] = df['adjusted_units']

# === 2. Приведение колонки 'date' к datetime ===
df['date'] = pd.to_datetime(df['date'])

# === 3. Год, месяц, день ===
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# === 4. Сезон по месяцу ===
df['season'] = 'unknown'
df.loc[df['month'].isin([12, 1, 2]), 'season'] = 'winter'
df.loc[df['month'].isin([3, 4, 5]), 'season'] = 'spring'
df.loc[df['month'].isin([6, 7, 8]), 'season'] = 'summer'
df.loc[df['month'].isin([9, 10, 11]), 'season'] = 'fall'

# === 5. Выходные (суббота и воскресенье) ===
df['weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# === 6. Праздники США ===
us_holidays = holidays.US()
df['holidays'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)

# === 7. Проверка ===
print(df[['date', 'year', 'month', 'day', 'season', 'weekend', 'holidays']].head())

# === 8. Сохранение результата ===
df.to_csv('train_with_temporal_features.csv', index=False)

df = pd.read_csv('train_with_temporal_features.csv')
# === Определение сезона по месяцу ===
df['season'] = 'unknown'  # очищаем перед назначением
df.loc[df['month'].isin([6, 7, 8]), 'season'] = 'summer'
df.loc[df['month'].isin([9, 10, 11]), 'season'] = 'fall'
df.loc[df['month'].isin([12, 1, 2]), 'season'] = 'winter'
df.loc[df['month'].isin([3, 4, 5]), 'season'] = 'spring'

# === Признак праздников США ===
us_holidays = holidays.US()
df['holidays'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)

# === Подсчёт количества праздничных и обычных дней ===
print("Распределение по праздникам:")
print(df['holidays'].value_counts())

# === Сохраняем финальный результат ===
df.to_csv('train_final_temporal.csv', index=False)

df = pd.read_csv('train_final_temporal.csv')

# === Извлечение года, месяца и дня ===
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# === Признак выходного дня: 1 — выходной (Сб/Вс), 0 — будни ===
df['weekend'] = df['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# === Назначение сезона по месяцу ===
df['season'] = 'unknown'
df.loc[df['month'].isin([12, 1, 2]), 'season'] = 'winter'
df.loc[df['month'].isin([3, 4, 5]), 'season'] = 'spring'
df.loc[df['month'].isin([6, 7, 8]), 'season'] = 'summer'
df.loc[df['month'].isin([9, 10, 11]), 'season'] = 'fall'
# === Сохранение финального датафрейма ===
df.to_csv('train_enriched_temporal.csv', index=False)

df = pd.read_csv('train_enriched_temporal.csv')

# === One-hot кодирование признака 'season' ===
df_season = pd.get_dummies(df['season'], prefix='is')

# === Объединение с основным датафреймом ===
df = df.join(df_season)

# === Удаление оригинального признака 'season' ===
df.drop(columns=['season'], inplace=True)

# === Сохранение результата ===
df.to_csv('train_with_onehot_season.csv', index=False)

# === Проверка результата ===
print("Размерность после кодирования:", df.shape)
print(df.head())

df = pd.read_csv('train_with_onehot_season.csv')

# ===Создание признака "sale": 1 если была продажа, иначе 0 ===
df['sale'] = np.where(df['units'] > 0, 1, 0)

# === Проверка распределения продаж ===
print("Распределение продаж (0 = нет, 1 = была):")
print(df['sale'].value_counts())

# ===  Сохранение результата ===
df.to_csv('train_final_with_sale.csv', index=False)