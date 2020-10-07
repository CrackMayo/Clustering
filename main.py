import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
