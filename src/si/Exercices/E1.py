import numpy as np
from si.io.csv_file import read_csv

# 1.1 Carregar dataset
filename = '../../../datasets/iris/iris.csv'
dataset = read_csv(filename, sep=',', features=True, label=True)

# 1.2 Selecionar a penúltima variável independente
penultima_coluna = dataset.X[:, -2]
print("Shape:", penultima_coluna.shape)
print("Dimensão da array resultante:", penultima_coluna.ndim)

# 1.3 Selecionar as últimas 10 amostras e calcular a média por feature
ultimas_10 = dataset.X[-10:]
media_ultimas_10 = np.mean(ultimas_10, axis=0)
print("Média das últimas 10 amostras:", media_ultimas_10)

# 1.4 Selecionar amostras com valor <= 6 em todas as features
mask = np.all(dataset.X <= 6, axis=1)
amostras_filtry = dataset.X[mask]
print("Número de amostras <= 6:", len(amostras_filtry))

# 1.5 Selecionar amostras com classe diferente de 'Iris-setosa'
mask_labels = dataset.y != 'Iris-setosa'
amostras_diff_setosa = dataset.X[mask_labels]
print("Número de amostras != Iris-setosa:", len(amostras_diff_setosa))