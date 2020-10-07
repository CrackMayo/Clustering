import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

#Cargamos los datos
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

#Aprendizaje y Evaluacion
model = KMeans(n_clusters=5, max_iter=500)
model.fit(data)
print(f'inertia del modelo = {model.inertia_}')
clusters= model.predict(data)
sil=metrics.silhouette_score(data, clusters)
print(f'Silueta= {sil}')
