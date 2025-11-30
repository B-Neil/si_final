import numpy as np
from si.io.csv_file import read_csv

# 1.1 Load dataset
filename = '../../../datasets/iris/iris.csv'
dataset = read_csv(filename, sep=',', features=True, label=True)

# 1.2 Select the penultimate independent variable
penultima_coluna = dataset.X[:, -2]
print("Shape:", penultima_coluna.shape)
print("Dimension of the resulting array:", penultima_coluna.ndim)

# 1.3 Select the last 10 samples and calculate the mean per feature
ultimas_10 = dataset.X[-10:]
media_ultimas_10 = np.mean(ultimas_10, axis=0)
print("Mean of the last 10 samples:", media_ultimas_10)

# 1.4 Select samples with value <= 6 in all features
mask = np.all(dataset.X <= 6, axis=1)
amostras_filtry = dataset.X[mask]
print("Number of samples <= 6:", len(amostras_filtry))

# 1.5 Select samples with class different from 'Iris-setosa'
mask_labels = dataset.y != 'Iris-setosa'
amostras_diff_setosa = dataset.X[mask_labels]
print("Number of samples != Iris-setosa:", len(amostras_diff_setosa))