import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans, DBSCAN



X, y = make_blobs(n_samples=100, random_state=10, centers=3)

df = pd.DataFrame(X, columns=['x1', 'x2'])
df['target'] = y

print(df.head())

plt.figure(figsize=(8, 6))
plt.scatter(df["x1"], df["x2"])



