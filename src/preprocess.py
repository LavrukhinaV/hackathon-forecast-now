import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


# Загрузка данных
train_df = pd.read_csv('data/train.csv')
key_df = pd.read_csv('data/key.csv')
weather_df = pd.read_csv('data/weather_by_city.csv')
store_coords_df = pd.read_csv('data/store_city_coordinates.csv')

# ШАГ 1: Отбор 20 магазинов и создание уникального ID
#уменьшение объёма данных для прототипа и ускорения обработки, c сохраннением географической и товарной репрезентативности.
selected_stores = train_df['store_nbr'].drop_duplicates().sample(n=20, random_state=42)
train_subset = train_df[train_df['store_nbr'].isin(selected_stores)].copy()
train_subset['unique_id'] = train_subset['store_nbr'].astype(str) + "_" + train_subset['item_nbr'].astype(str)

# ШАГ 2: Связь магазинов с городами
store_station_city_df = pd.merge(key_df, store_coords_df, on='store_nbr', how='left')

# Приведение городов к нижнему регистру и удаление пробелов
store_station_city_df['city'] = store_station_city_df['city'].str.lower().str.strip()
weather_df['city'] = weather_df['city'].str.lower().str.strip()

# ШАГ 3: Объединение по дате и городу
#это основная цель проекта — понять, как погода влияет на продажи. Использован ключ (дата + город).
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df['date'] = weather_df['date'] - pd.DateOffset(years=11)
weather_df['date'] = weather_df['date'].dt.strftime('%Y-%m-%d')

# Объединение train_subset с городами
train_merged = pd.merge(train_subset, store_station_city_df[['store_nbr', 'city']], on='store_nbr', how='left')

# Объединение train_merged с погодой
train_merged_weather = pd.merge(train_merged, weather_df, on=['date', 'city'], how='left')

# Сохранение результата
train_merged_weather.to_csv('train_with_weather.csv', index=False)
