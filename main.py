import pandas as pd

#Cargamos los datos 
data = pd.read_excel("titanic.xlsx",sheet_name=0) 
data.head()