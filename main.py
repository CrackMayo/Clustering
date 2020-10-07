import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_excel("titanic.xlsx",sheet_name=0)
data.head()
data['Clase'] = data['Clase'].astype('category')
data['Edad'] = data['Edad'].astype('category')
data['Sexo'] = data['Sexo'].astype('category')
data['Sobrevivio'] = data['Sobrevivio'].astype('category')
data.info()
data.describe()
data['Sobrevivio'].value_counts().plot(kind='bar')
dummiesClase = pd.get_dummies(data['Clase'])
data = data.drop('Clase', axis=1)
data = data.join(dummiesClase)
data.head()

model = KMeans(n_clusters=5, max_iter=500)
model.fit(data)
print(f'inertia del modelo = {model.inertia_}')
clusters= model.predict(data)
sil=metrics.silhouette_score(data, clusters)
print(f'Silueta= {sil}')

centroides=pd.DataFrame(model.cluster_centers_, columns=data.columns.values)
centroides.round(0)
data['cluster']=model.predict(data)
data.head()
pd.value_counts(data["cluster"])

def elbow():
    ks = range(1, 20)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(data)
        inertias.append(model.inertia_)
    plt.plot(ks, inertias, '-o')
    plt.xlabel('Clusters num, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()