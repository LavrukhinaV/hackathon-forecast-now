import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


"""Анализ мультиколлинеарности с помощью коэффициента VIF выявил сильную зависимость между температурой и осадками (VIF > 10). Это может негативно повлиять на стабильность коэффициентов в интерпретируемых моделях.

Кластеризация для выделения категорий
"""

# === 1. Загрузка исходных данных
# Загрузка данных без выбросов
df = pd.read_csv('train_no_extreme_outliers.csv')
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

# 1. Сортируем все уникальные недели и находим границу
all_weeks = sorted(weekly_sales['week'].unique())
split_idx = int(len(all_weeks) * 0.8)
train_weeks = all_weeks[:split_idx]
test_weeks  = all_weeks[split_idx:]

# 2. Делим датафрейм weekly_sales на train и test по неделе
weekly_train = weekly_sales[weekly_sales['week'].isin(train_weeks)]
weekly_test  = weekly_sales[weekly_sales['week'].isin(test_weeks)]

# 3. Строим pivot-таблицы с одинаковыми колонками (все train_weeks)
sales_train = weekly_train.pivot(index='item_nbr', columns='week', values='units').reindex(columns=train_weeks, fill_value=0)
sales_test  = weekly_test .pivot(index='item_nbr', columns='week', values='units').reindex(columns=train_weeks, fill_value=0)

# 4. Масштабирование и кластеризация только на train, затем predict для test
scaler = StandardScaler()
train_scaled = scaler.fit_transform(sales_train)
test_scaled  = scaler.transform(sales_test)

kmeans = KMeans(n_clusters=10, random_state=42)
train_labels = kmeans.fit_predict(train_scaled)
test_labels  = kmeans.predict(test_scaled)

# 5. Собираем Series item_nbr → category и мапим обратно в df
train_cat = pd.Series(train_labels, index=sales_train.index)
test_cat  = pd.Series(test_labels,  index=sales_test.index)
all_cat   = pd.concat([train_cat, test_cat])

df['category'] = df['item_nbr'].map(all_cat.to_dict())

"""# Построение интерпретируемой модели Lasso и корректировка прогноза на погоду"""

# === 2. VIF-анализ для оценки мультиколлинеарности
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

vif_df = df[['avg_temp_c', 'precipitation_mm', 'city_code']].dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = vif_df.columns
vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])]
print("\n📊 VIF-анализ мультиколлинеарности:")
print(vif_data)

# Визуализация
plt.figure(figsize=(8, 4))
sns.barplot(x="VIF", y="feature", data=vif_data)
plt.title("VIF по признакам")
plt.grid(True)
plt.tight_layout()
plt.show()