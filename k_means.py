#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster

#Lê a tabela
iris = pd.read_csv('./Iris.csv', usecols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm','Species'])

#Mostra algumas informações dela
print(iris.head())
print(iris.info())
sns.pairplot(iris, hue='Species')
plt.show()

#Aqui separa apenas duas dimensões dos dados.
iris2D = iris[['PetalWidthCm','PetalLengthCm']]

print(iris2D)

fig, axe = plt.subplots(figsize=(8,5))
axe = sns.scatterplot(data=iris2D, x='PetalWidthCm', y='PetalLengthCm')
plt.show()


# Gráfico do "cotovelo". Esse gráfico nos dá uma ideia da quantidade de 'k' adequada para aqueles dados.
from sklearn.cluster import KMeans
inertias = []
k_values = range(1, 11)  # Testar de k = 1 até k = 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(iris2D)
    inertias.append(kmeans.inertia_)  # Soma das distâncias ao centróide

# Plotar o gráfico do cotovelo

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.title('Método do Cotovelo para Selecionar o Número Ideal de Clusters')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Soma dos Erros Quadrados)')
plt.xticks(k_values)
plt.grid(True)
plt.show()


#Aqui começa o KMeans

kmeans = cluster.KMeans(3)
#Estamos usando a implementação interna da classe KMeans do scikit-learn, 
# que já contém o laço de repetição responsável por:
# 1-Atribuir os pontos ao cluster mais próximo;
# 2-Recalcular os centróides;
# 3-Repetir o processo até convergir (ou até atingir um número máximo de iterações).


clusters = kmeans.fit_predict(iris2D)
print("Número de iterações: ", kmeans.n_iter_)

print(pd.Series(clusters).value_counts())

x_center, y_center = kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1]

fig, (axe1, axe2) = plt.subplots(1,2, figsize=(14,5))

sns.scatterplot(data=iris2D, x='PetalWidthCm', y='PetalLengthCm', hue=clusters, ax=axe1)
axe1.scatter(x=x_center, y=y_center, s = 80, color='red')
sns.scatterplot(data=iris, x='PetalWidthCm', y='PetalLengthCm', hue='Species', ax=axe2)
plt.show()
