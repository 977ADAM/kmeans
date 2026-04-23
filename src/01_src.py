import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from rich.console import Console

c = Console()
df = pd.read_csv("data/online_retail_II.csv")
c.print(f'Размер датасета: {df.shape[0]:,} строк, {df.shape[1]} колонок')
# Убираем строки без покупателя
df = df.dropna(subset=['Customer ID'])
c.print(f'После удаления пропусков Customer ID: {len(df):,} строк')
# Убираем отменённые заказы (код начинается с 'C')
df = df[~df['Invoice'].astype(str).str.startswith('C')]
c.print(f'После удаления отмен: {len(df):,} строк')
# Убираем некорректные значения
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
c.print(f'После удаления Quantity/Price <= 0: {len(df):,} строк')
df['Revenue'] = df['Quantity'] * df['Price']
c.print(f'Осталось {len(df)/525462*100:.1f}% от исходных данных')
reference_date = df['InvoiceDate'].astype('datetime64[ns]').max() + pd.Timedelta(days=1)
c.print(f'Дата отсчёта: {reference_date}')
df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')
df["Customer ID"] = df["Customer ID"].astype(int)


rfm = df.groupby('Customer ID').agg(
    Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency = ('Invoice',     'nunique'),
    Monetary  = ('Revenue',     'sum')
).reset_index()

features = rfm[['Recency', 'Frequency', 'Monetary']].copy()

features_log = np.log1p(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_log)

K = 4  # можно поменять на другое значение

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, init='k-means++')
rfm['Cluster'] = kmeans.fit_predict(features_scaled)

cluster_summary = rfm.groupby('Cluster').agg(
    Покупателей  = ('Customer ID', 'count'),
    Recency_avg  = ('Recency',    'mean'),
    Frequency_avg= ('Frequency',  'mean'),
    Monetary_avg = ('Monetary',   'mean'),
).round(1)

c.print(f'Распределение покупателей по {K} кластерам:')
c.print('Средние значения RFM по кластерам:')
c.print(cluster_summary)
c.print()
c.print('Подсказка:')
c.print('  Recency_avg  — меньше = покупал недавно (лучше)')
c.print('  Frequency_avg — больше = покупал чаще (лучше)')
c.print('  Monetary_avg  — больше = потратил больше (лучше)')

cluster_names = {
    0: 'Сегмент A',
    1: 'Сегмент B',
    2: 'Сегмент C',
    3: 'Сегмент D',
}

rfm['Segment'] = rfm['Cluster'].map(cluster_names)

c.print('Итоговое распределение по сегментам:')
c.print(rfm['Segment'].value_counts())
c.print(rfm.head())