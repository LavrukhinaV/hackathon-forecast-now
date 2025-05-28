import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv('train_final_with_sale.csv')
df['date'] = pd.to_datetime(df['date'])


# Кол-во дней с последней продажей этого товара в этом магазине
df = df.sort_values(['store_nbr', 'item_nbr', 'date'])

df['days_since_last_sale'] = df.groupby(['store_nbr', 'item_nbr'])['date'].diff().dt.days
df['days_since_last_sale'].fillna(999, inplace=True)  # если первая продажа — заполняем 999





#  Сохраняем обогащённый датафрейм
df.to_csv('train_sale_with_features.csv', index=False)

print("✅ Feature engineering completed. New shape:", df.shape)

"""Обучение классификатора"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# === 1. Загрузка данных ===
df = pd.read_csv('train_sale_with_features.csv')

# === 2. Разделение признаков и целевой переменной ===
X = df.drop(columns=['units', 'sale'])  # признаки
y_class = df['sale']                    # целевая переменная

# === 3. Удаление datetime-признаков (если остались) ===
for col in X.columns:
    if np.issubdtype(X[col].dtype, np.datetime64):
        print(f"Удалён datetime-признак: {col}")
        X = X.drop(columns=[col])

# === 4. Оставляем только числовые признаки ===
X = X.select_dtypes(include=['number'])

# === 5. Разделение на обучающую выборку для оценки важности признаков ===
X_train_full, _, y_train_full, _ = train_test_split(X, y_class, test_size=0.3, random_state=42)

# === 6. Обучение модели Random Forest ===
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train_full, y_train_full)

# === 7. Получение важности признаков ===
clf_feat_imp = pd.Series(clf_rf.feature_importances_, index=X_train_full.columns)

# === 8. Визуализация Top-15 признаков ===
plt.figure(figsize=(14, 8))
clf_feat_imp.nlargest(15).plot(kind='barh')
plt.title("Feature Importance (for sale classification)")
plt.xlabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# === Загрузка данных ===
df = pd.read_csv('train_sale_with_features.csv')

df['date'] = pd.to_datetime(df['date'])
df['store_nbr'] = df['store_nbr'].astype(int)
df['item_nbr']  = df['item_nbr'].astype(int)

split_date = df['date'].quantile(0.8)
train_df = df[df['date'] <= split_date].copy()
test_df  = df[df['date'] >  split_date].copy()

# === Линейная модель Lasso для интерпретации погоды ===
train_df['avg_sales_last_7d']  = np.nan
train_df['avg_sales_last_30d'] = np.nan



for (store, item), grp in train_df.groupby(['store_nbr', 'item_nbr']):
    idx   = grp.index
    sales = grp['units']
    train_df.loc[idx, 'avg_sales_last_7d']  = sales.shift(1).rolling(7,  min_periods=1).mean().values
    train_df.loc[idx, 'avg_sales_last_30d'] = sales.shift(1).rolling(30, min_periods=1).mean().values

# === 5. Для теста подставляем последние значения из train (без утечки) ===
last7  = train_df.groupby(['store_nbr','item_nbr'])['avg_sales_last_7d'].last().to_dict()
last30 = train_df.groupby(['store_nbr','item_nbr'])['avg_sales_last_30d'].last().to_dict()

def fetch_last_feats(row):
    key = (row['store_nbr'], row['item_nbr'])
    return pd.Series({
        'avg_sales_last_7d':  last7.get(key, 0.0),
        'avg_sales_last_30d': last30.get(key, 0.0)
    })

test_feats = test_df.apply(fetch_last_feats, axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_feats.reset_index(drop=True)], axis=1)

# === 6. Подготовка данных и обучение модели Lasso (LogisticRegression с L1) ===
num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'sale' in num_features:
    num_features.remove('sale')

X_train = train_df[num_features]
y_train = train_df['sale']
X_test  = test_df[num_features]
y_test  = test_df['sale']

"""# Классификация и регрессия с XGBoost и LGBM, подбор гиперпараметров


"""

from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter

# Ресемплим вручную, чтобы посчитать scale_pos_weight
X_smote, y_smote = smote.fit_resample(X_train, y_train)
X_res, y_res = rus.fit_resample(X_smote, y_smote)

counter = Counter(y_res)
scale_pos_weight = counter[0] / counter[1]
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )),
])

# Сетка параметров для RandomizedSearch
param_dist = {
    'xgb__n_estimators': [100, 200, 300, 400],
    'xgb__max_depth': [3, 5, 7, 9],
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__min_child_weight': [1, 3, 5],
}

search = RandomizedSearchCV(
    pipe, param_distributions=param_dist,
    scoring='f1',
    n_iter=30,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)

# Предсказания на тесте (пробуем лучший pipeline)
best_pipe = search.best_estimator_
y_proba = best_pipe.predict_proba(X_test)[:, 1]

print("Лучшие параметры:", search.best_params_)
print(f"Лучший f1-score на кросс-валидации: {search.best_score_:.4f}")

# Предсказания на тесте (пробуем лучший pipeline)
best_pipe = search.best_estimator_
y_proba = best_pipe.predict_proba(X_test)[:, 1]

# Подбор оптимального порога для F1
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Оптимальный порог: {best_threshold:.2f}, F1-score на тесте: {f1_scores[best_idx]:.4f}")

# Финальный предсказания с оптимальным порогом
y_pred_final = (y_proba >= best_threshold).astype(int)

print(f"ROC-AUC на тесте: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=y_train)
plt.title("Распределение классов в обучающей выборке")
plt.show()

from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# ROC и PR кривые
fpr, tpr, _ = roc_curve(y_test, y_scores)
precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix

# Матрица ошибок с учётом подобранного порога
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказано')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (c threshold={:.2f})'.format(best_threshold))
plt.show()

# Получаем обученный XGBClassifier из pipeline
xgb_model = best_pipe.named_steps['xgb']

# Получаем важности признаков
importances = xgb_model.feature_importances_

# Можно отобразить в таблице
import pandas as pd

feature_names = X_train.columns  # если у тебя DataFrame
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)

print(feat_imp_df)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(20), x='importance', y='feature', palette='viridis')
plt.title('Топ 20 важных признаков по версии XGBoost')
plt.tight_layout()
plt.show()

"""Обучение регрессора"""

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# === 1. Загрузка данных ===
df = pd.read_csv('train_sale_with_features.csv')

# === 2. Разделение признаков ===
# Выделяем целевую переменную для классификации (продажа или нет) и регрессии (units)
y_class = df['sale']
y_units = df['units']

# Автоматически определяем числовые и категориальные признаки
num_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['sale', 'units']).tolist()
cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# === 3. Train/Test split ===
X_train, X_test, y_class_train, y_class_test, y_units_train_full, y_units_test_full = \
    train_test_split(
        df[num_features + cat_features], y_class, y_units,
        test_size=0.3, random_state=42, stratify=y_class
    )

# === 4. Pipeline для классификации ===
preprocessor_clf = ColumnTransformer([  # предобработка для классификации
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])
clf_pipeline = Pipeline([
    ('preproc', preprocessor_clf),
    ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42))
])
clf_pipeline.fit(X_train, y_class_train)

# Предсказания вероятности продажи и бинарного исхода
y_class_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]
# Выбираем порог по ROC AUC (по умолчанию 0.5)
y_class_pred = (y_class_pred_proba > 0.5).astype(int)

# === 5. Подготовка данных для регрессии ===
# Фильтруем по предсказанным продажам\mask_train = clf_pipeline.predict(X_train) == 1
mask_train = clf_pipeline.predict(X_train) == 1
mask_test  = y_class_pred == 1

X_train_reg = X_train.loc[mask_train]
X_test_reg  = X_test.loc[mask_test]
y_train_reg = y_units_train_full.loc[mask_train]
y_test_reg  = y_units_test_full.loc[mask_test]

# === 6. Pipeline для регрессии ===
preprocessor_reg = ColumnTransformer([  # предобработка для регрессии
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])
reg_pipeline = Pipeline([
    ('preproc', preprocessor_reg),
    ('reg', LGBMRegressor(random_state=42))
])

# Параметры для RandomizedSearch
param_dist = {
    'reg__learning_rate': [0.01, 0.05, 0.1],
    'reg__n_estimators': [100, 300, 500, 1000],
    'reg__num_leaves': [31, 63, 127],
    'reg__max_depth': [-1, 10, 20],
    'reg__subsample': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    reg_pipeline, param_dist, n_iter=20, cv=3, scoring='neg_root_mean_squared_error',
    n_jobs=-1, random_state=42
)
search.fit(X_train_reg, y_train_reg)

# Лучший регрессионный прогноз
y_pred_base = search.predict(X_test_reg)
rmse_base = np.sqrt(mean_squared_error(y_test_reg, y_pred_base))

# === 7. Корректировка прогноза по погоде ===
# Можно сразу включить weather-признаки в модель, но если нужна пост-коррекция:
coeffs = {'avg_temp_c': coefficients['avg_temp_c'],
          'precipitation_mm': coefficients['precipitation_mm']}
# вычисляем средние на трейне
temp_mean = df.loc[X_train_reg.index, 'avg_temp_c'].mean()
precip_mean = df.loc[X_train_reg.index, 'precipitation_mm'].mean()
# дельты на тесте
delta_temp   = df.loc[X_test_reg.index, 'avg_temp_c'] - temp_mean
ndelta_precip = df.loc[X_test_reg.index, 'precipitation_mm'] - precip_mean
adjustment_factor = np.exp(
    coeffs['avg_temp_c'] * delta_temp + coeffs['precipitation_mm'] * ndelta_precip
)

y_pred_adj = y_pred_base * adjustment_factor
rmse_adj = np.sqrt(mean_squared_error(y_test_reg, y_pred_adj))

# Вывод результатов ===
results = pd.DataFrame([{
    'Model': 'LGBM_REG',
    'RMSE Base': round(rmse_base, 14),
    'RMSE Adjusted (weather)': round(rmse_adj, 14)
}])
print(results)
