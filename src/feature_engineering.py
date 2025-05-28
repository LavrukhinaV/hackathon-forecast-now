import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

"""## Обработка выбросов и заполнение пропусков"""

df = pd.read_csv('train_with_weather.csv')

# Типизация
df['date'] = pd.to_datetime(df['date'])
df['store_nbr'] = df['store_nbr'].astype(int)
df['item_nbr'] = df['item_nbr'].astype(int)
df['city'] = df['city'].astype(str).str.lower().str.strip()

# ===  ПРОПУСКИ ===
#восполняем пропуски усредняя по городу. Это сохраняет климатические особенности регионов и не искажает общую структуру данных.
df['avg_temp_c'] = df.groupby('city')['avg_temp_c'].transform(lambda x: x.fillna(x.mean()))
df['avg_temp_c'].fillna(df['avg_temp_c'].mean(), inplace=True)

df['precipitation_mm'] = df.groupby('city')['precipitation_mm'].transform(lambda x: x.fillna(x.mean()))
df['precipitation_mm'].fillna(df['precipitation_mm'].mean(), inplace=True)

# === ЛОГАРИФМИРОВАНИЕ ПРОДАЖ ===
#уменьшает влияние выбросов, нормализует распределение продаж. Это улучшает стабильность и интерпретируемость регрессионных моделей.
df['log_units'] = np.log1p(df['units'])

# ===  КОДИРОВАНИЕ КАТЕГОРИЙ ===
le_city = LabelEncoder()
df['city_code'] = le_city.fit_transform(df['city'])

le_uid = LabelEncoder()
df['unique_id_code'] = le_uid.fit_transform(df['unique_id'])
# === Выбросы через Z-score ===
#заменяем отклоняющиеся значения медианами по паре магазин + товар. Это сохраняет полноту выборки без искажения метрик.
numeric_cols = ['units', 'avg_temp_c', 'precipitation_mm', 'log_units']
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df['outlier'] = (z_scores > 3).any(axis=1)

# === Сколько выбросов всего ===
total_rows = len(df)
total_outliers = df['outlier'].sum()
print(f"Общее количество выбросов: {total_outliers}")
print(f"Процент выбросов: {100 * total_outliers / total_rows:.2f}%")

# === Распределение по признакам ===
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = ['units', 'avg_temp_c', 'precipitation_mm', 'log_units']
outliers_df = df[df['outlier'] == True]

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='outlier', y=col)
    plt.title(f'Распределение "{col}" с выбросами')
    plt.show()

# === Примеры выбросов с наибольшими значениями units, температуры и осадков ===
print("\n🔍 ТОП 10 выбросов по units:")
print(outliers_df.sort_values('units', ascending=False)[['date', 'store_nbr', 'item_nbr', 'units', 'avg_temp_c', 'precipitation_mm']].head(10))

print("\n🔍 ТОП 10 по температуре:")
print(outliers_df.sort_values('avg_temp_c', ascending=False)[['date', 'store_nbr', 'item_nbr', 'units', 'avg_temp_c', 'precipitation_mm']].head(10))

print("\n🔍 ТОП 10 по осадкам:")
print(outliers_df.sort_values('precipitation_mm', ascending=False)[['date', 'store_nbr', 'item_nbr', 'units', 'avg_temp_c', 'precipitation_mm']].head(10))

"""Сначала мы отфильтровали экстремальные выбросы по продажам (более 1000 единиц в день), чтобы исключить влияние редких аномальных пиков. Далее выбросы, определённые через Z-оценку (>3), были заменены медианными значениями по каждому товару и магазину. Это позволило сохранить структуру данных и снизить искажения при обучении модели."""

# Удалим экстремальные выбросы (например, > 1000 продаж в день)
#заменяем отклоняющиеся значения медианами по паре магазин + товар. Это сохраняет полноту выборки без искажения метрик.
df_cleaned = df[df['units'] <= 1000].copy()

#Заменим менее экстремальные выбросы (по Z-оценке) медианой по store + item
from scipy import stats

z_scores_units = np.abs(stats.zscore(df_cleaned['units']))
df_cleaned['units_z'] = z_scores_units

# Обнаружим оставшиеся выбросы по Z
unit_outliers_mask = df_cleaned['units_z'] > 3

# Рассчитаем медианы
median_map = df_cleaned.groupby(['store_nbr', 'item_nbr'])['units'].median()

# Заменим выбросы медианами
def replace_with_median(row):
    if row['units_z'] > 3:
        return median_map.get((row['store_nbr'], row['item_nbr']), row['units'])
    return row['units']

df_cleaned['units'] = df_cleaned.apply(replace_with_median, axis=1)
df_cleaned.drop(columns=['units_z'], inplace=True)

# Обновим log_units
df_cleaned['log_units'] = np.log1p(df_cleaned['units'])

# Сохраняем результат
df_cleaned.to_csv('train_no_extreme_outliers.csv', index=False)
