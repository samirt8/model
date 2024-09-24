import pandas as pd

# Charger les données
df = pd.read_csv('test_data.csv')

# Transformer le champ SK_ID_CURR en int
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

# Sauvegarder le fichier
df.to_csv('C:/Users/lenovo/Desktop/api/app/model/test_data_int.csv', index=False)
