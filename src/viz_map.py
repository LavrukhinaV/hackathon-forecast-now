import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap


# Визуализации и гео-аналитика (HeatMap)


# Группируем продажи по складу и городу
train = pd.read_csv("/content/train.csv")
coords = pd.read_csv("/content/store_city_coordinates.csv")

# Добавим город к каждому store_nbr
train = train.merge(coords[['store_nbr', 'city', 'latitude', 'longitude']], on='store_nbr', how='left')

# Добавим продажи по складу и городу
sales_by_store_city = (
    train.groupby(['store_nbr', 'city', 'latitude', 'longitude'])['units']
    .sum()
    .reset_index()
    .rename(columns={'units': 'total_units'})
)

sales_by_store_city.to_csv("/content/sales_region_map.csv", index=False)

sales_by_store_city.info()

import folium
from folium.plugins import HeatMap


# Центрируем карту по средним координатам
map_center = [sales_by_store_city['latitude'].mean(), sales_by_store_city['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=2)

# Формируем данные для тепловой карты: [lat, lon, вес]
heat_data = sales_by_store_city[['latitude', 'longitude', 'total_units']].values.tolist()

# Добавляем слой HeatMap
HeatMap(heat_data).add_to(m)

# Сохраняем в HTML
map_file = '/content/store_heatmap.html'
m.save(map_file)

# Показать карту в ноутбуке
